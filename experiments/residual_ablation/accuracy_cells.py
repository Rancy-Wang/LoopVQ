"""Exclusive ownership and checked evidence for resumable accuracy cells."""
import fcntl
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'loopvq_reproduction'))
from evaluation_io import atomic_json, digest, file_hash
from run_evaluation import COUNTS, collect_cell

ARMS = {'bf16': 'bf16', 'loopvq': 'loopvq_0.5bit', 'residual': 'loopvq_residual4',
        'plain-continued': 'loopvq_plain_continued', 'residual-ft': 'loopvq_residual_ft'}
LABELS = {'bf16': 'BF16 accuracy', 'loopvq': 'LoopVQ 0.5bit accuracy',
          'residual': '原码本＋INT4 残差', 'plain-continued': '普通 KL 续训对照',
          'residual-ft': '残差感知续训'}
BENCH_LABELS = dict(zip(COUNTS, ['AIME24', 'AIME25', 'GPQA-D', 'HumanEval+', 'MBPP+']))
BASE_SHA = '3df02ee379e75f24c4691d8641a2970bbbf8fb2d'


def key(arm, bench, seed):
    return f'{arm}__{bench}__{seed}'


def job_directory(root, worker, arm):
    return root / 'jobs' / arm if worker == 'main-' + arm else root / 'replicas' / worker / arm


def registry(root):
    directory = root / 'dispatch'
    for sub in ('owners', 'locks', 'done'):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    return directory


def initialize(root):
    directory = registry(root)
    with (directory / 'registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Existing partial AIME cells must resume at the original endpoint.
        for arm in ARMS:
            existing = {('aime24', 0)}
            for running in (root / 'jobs' / arm).glob('evaluation/**/full/**/running.json'):
                if not running.with_name('complete.json').exists():
                    record = json.loads(running.read_text())
                    existing.add((record['benchmark'], record['seed']))
            for bench, seed in existing:
                name = key(arm, bench, seed)
                path = directory / 'owners' / (name + '.json')
                if not path.exists():
                    atomic_json(path, {'worker': 'main-' + arm, 'arm': arm, 'benchmark': bench,
                        'seed': seed, 'job': str(root / 'jobs' / arm), 'preserved_original_cell': True})


def claim(root, worker, preferred, first_benchmark='aime24'):
    directory = registry(root)
    with (directory / 'registry.lock').open('a') as registry_lock:
        fcntl.flock(registry_lock, fcntl.LOCK_EX)
        options = []
        for arm in ARMS:
            for bench in COUNTS:
                for seed in range(3):
                    name = key(arm, bench, seed)
                    if (directory / 'done' / (name + '.json')).exists():
                        continue
                    path = directory / 'owners' / (name + '.json')
                    owner = json.loads(path.read_text()) if path.exists() else None
                    if owner and owner['worker'] != worker:
                        continue
                    priority = (owner is None, arm != preferred, bench != first_benchmark,
                                list(COUNTS).index(bench), seed, list(ARMS).index(arm))
                    options.append((priority, name, path, owner, arm, bench, seed))
        for _, name, path, owner, arm, bench, seed in sorted(options):
            lock = (directory / 'locks' / (name + '.lock')).open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            if owner is None:
                owner = {'worker': worker, 'arm': arm, 'benchmark': bench, 'seed': seed,
                         'job': str(job_directory(root, worker, arm))}
                atomic_json(path, owner)
            return owner, lock
    return None


def recipe_directory(job, arm):
    paths = list((job / 'evaluation' / ARMS[arm]).glob('*/recipe.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one immutable recipe in {job}: {paths}')
    return paths[0].parent


def checked_cell(cell, method, bench, seed):
    marker = json.loads((cell / 'complete.json').read_text())
    if (marker['benchmark'] != bench or marker['seed'] != seed or marker['audit']
            or marker['method'] != method or marker['exit_code'] != 0):
        raise ValueError(f'Incorrect completion identity: {cell}')
    stem = f'{method}_{seed}_{bench}'
    scores = collect_cell(cell, stem, bench, COUNTS[bench])
    if scores != marker['scores']:
        raise ValueError(f'Changed completion evidence: {cell}')
    protocol = json.loads((cell / f'{stem}_protocol.json').read_text())
    if any(protocol.get(k) != v for k, v in {
            'api_seed': seed, 'temperature': 1.0, 'top_p': 0.7,
            'max_gen_toks': 32768, 'limit': None}.items()):
        raise ValueError(f'Incorrect scoring protocol: {cell}')
    values = {}
    for row in map(json.loads, (cell / f'{stem}_samples.jsonl').read_text().splitlines()):
        identity = str(row.get('task_id', row.get('doc_id')))
        value = float(row['plus'] == 'pass') if bench.startswith('evalplus_') else float(row['exact_match'])
        if value not in (0.0, 1.0):
            raise ValueError('Expected binary sample-level correctness')
        values[identity] = value
    if not math.isclose(100 * sum(values.values()) / len(values), scores['accuracy_percent'], abs_tol=1e-8):
        raise ValueError(f'Sample outcomes disagree with aggregate score: {cell}')
    data_identity = {k: protocol[k] for k in ('dataset_documents_sha256', 'dataset_hash', 'task_ids') if k in protocol}
    if not data_identity:
        raise ValueError('Missing frozen dataset identity')
    return scores, values, data_identity


def register_done(root, owner):
    recipe = recipe_directory(Path(owner['job']), owner['arm'])
    cell = recipe / 'full' / owner['benchmark'] / str(owner['seed'])
    scores, _, data = checked_cell(cell, ARMS[owner['arm']], owner['benchmark'], owner['seed'])
    result = {**owner, 'cell': str(cell), 'recipe': str(recipe / 'recipe.json'),
              'recipe_sha256': file_hash(recipe / 'recipe.json'), 'scores': scores, 'data_identity': data}
    atomic_json(registry(root) / 'done' / (key(owner['arm'], owner['benchmark'], owner['seed']) + '.json'), result)
    return result
