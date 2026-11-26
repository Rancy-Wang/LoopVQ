"""Own one TP2 accuracy server plus audited clients, with durable evidence."""
import argparse
import fcntl
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/loopvq_reproduction'))
from evaluation_io import atomic_json, digest, file_hash

ARMS = {'bf16': 'bf16', 'loopvq': 'loopvq_0.5bit', 'residual': 'loopvq_residual4',
        'plain-continued': 'loopvq_plain_continued', 'residual-ft': 'loopvq_residual_ft'}


def output(command):
    return subprocess.check_output(command, text=True).strip()


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--original-root', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--build-python', required=True)
    ap.add_argument('--eval-python', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = ap.parse_args()
    root = args.root.resolve()
    job = root / 'jobs' / args.arm
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ.get('TMUX'), 'Run long jobs inside tmux'
    assert len(args.gpus.split(',')) == 2
    for arm in ARMS:
        assert (root / 'results' / f'r6-{arm}.exit').read_text().strip() == '0', arm
        result = json.loads((root / 'results' / f'r6-{arm}.json').read_text())['audit']
        assert result['graph_eager_max_error'] == 0 and result['boundary_greedy_exact'], arm
    assert json.loads((root / 'results/r6-validation.json').read_text())['passed']
    assert output(['git', '-C', str(REPO), 'branch', '--show-current']) == 'test'
    assert not output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'])
    revision = output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'])
    gpu = output(['nvidia-smi', '-i', args.gpus, '--query-gpu=index,uuid,name,memory.used,memory.total',
                  '--format=csv,noheader'])
    busy = output(['nvidia-smi', '-i', args.gpus, '--query-compute-apps=pid', '--format=csv,noheader'])
    if busy:
        raise RuntimeError(f'GPU pair became busy: {args.gpus}, PIDs={busy}')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    recipes = [json.loads(p.read_text().splitlines()[0])['config'] for p in (
        args.original_root / 'logs/finetune.jsonl', root / 'logs/plain-continued.jsonl',
        root / 'logs/residual-ft.jsonl')]
    changed = {'src_files', 'residual'}
    assert all({k:v for k,v in r.items() if k not in changed} ==
               {k:v for k,v in recipes[0].items() if k not in changed} for r in recipes)
    # Hash actual model files once, under a shared lock; later workers also
    # verify the stat fingerprint before using this immutable inventory.
    model_files = sorted(p for p in args.model.iterdir() if p.is_file() and
                         p.suffix in ('.safetensors', '.json', '.py', '.model'))
    stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in model_files}
    with (root / 'model-hashes.lock').open('w') as model_lock:
        fcntl.flock(model_lock, fcntl.LOCK_EX)
        hashes_path = root / 'model-hashes.json'
        if hashes_path.exists():
            hashes = json.loads(hashes_path.read_text())
            assert hashes['stats'] == stats, 'Model files changed'
        else:
            hashes = {'stats': stats, 'files': {p.name: file_hash(p) for p in model_files}}
            atomic_json(hashes_path, hashes)
    bundle = None
    if args.arm in ('loopvq', 'residual'):
        bundle = args.original_root / 'artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e'
    elif args.arm != 'bf16':
        bundle = root / 'artifacts' / args.arm
    bundle_hashes = {p.name: file_hash(p) for p in sorted(bundle.iterdir()) if p.is_file()} if bundle else None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4',
        HF_HOME=str(args.original_root / 'cache/huggingface'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        WANDB_BASE_URL='https://api.wandb.ai', WANDB_DIR=str(root / 'wandb'))
    command = [str(Path(args.build_python).with_name('torchrun')), '--standalone', '--nproc-per-node=2',
        str(Path(__file__).with_name('hf_reference.py')), '--model', str(args.model), '--cuda-graph',
        '--port', str(args.port), '--output', str(job / 'server-ready.json')]
    if bundle:
        command += ['--bundle', str(bundle)]
    if args.arm in ('residual', 'residual-ft'):
        command += ['--residual']
    manifest = {'method': ARMS[args.arm], 'arm': args.arm, 'implementation': 'transformers_reference',
        'inference_git_sha': revision, 'model_sha256': digest(hashes['files']),
        'bundle_sha256': digest(bundle_hashes) if bundle else None, 'bundle_files': bundle_hashes,
        'total_ut_steps': 4, 'tp': 2, 'host': platform.node(), 'gpus': gpu, 'command': command,
        'prefix_tokens': 8, 'recent_tokens': 64, 'page_size': 8, 'cuda_graph': True,
        'attention_backend': 'pytorch_efficient_sdpa', 'materialized_bf16_cache': True,
        'residual': {'bits':4, 'group_size':32, 'scale_dtype':'float16'} if '--residual' in command else None}
    atomic_json(job / 'server-manifest.json', manifest)
    server, client = None, None
    state = {'arm': args.arm, 'host': platform.node(), 'started_unix': time.time(), 'git_sha': revision,
             'gpu': gpu, 'host_memory': output(['free', '-h']), 'status': 'starting'}
    try:
        with (job / 'server.log').open('a', buffering=1) as log:
            server = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        state['server_pid'] = server.pid
        atomic_json(job / 'state.json', state)
        deadline = time.time() + 600
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'Server startup failed: {server.returncode}')
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError('Server startup exceeded 600 seconds')
        common = [args.eval_python, str(REPO / 'experiments/loopvq_reproduction/run_evaluation.py'),
            '--model', str(args.model), '--base-url', f'http://127.0.0.1:{args.port}',
            '--method', ARMS[args.arm], '--server-manifest', str(job / 'server-manifest.json'),
            '--root', str(job / 'evaluation'), '--num-concurrent', '1',
            '--wandb-entity', '529513043-tsinghua-university',
            '--wandb-project', 'loopvq-ouro-2p6b-thinking', '--wandb-group', '2026-11-26-residual-ablation']
        for phase, tail in [('audit', ['--audit', '--benchmarks', 'aime24', 'evalplus_humaneval', '--seeds', '0', '1']),
                            ('full', ['--seeds', *map(str, args.seeds)])]:
            state.update(status=phase, command=common + tail)
            with (job / f'{phase}.log').open('a', buffering=1) as log:
                client = subprocess.Popen(common + tail, cwd=REPO, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            state['client_pid'] = client.pid
            atomic_json(job / 'state.json', state)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f'Server failed during {phase}: {server.returncode}')
                time.sleep(5)
            if client.returncode:
                raise RuntimeError(f'{phase} client failed: {client.returncode}, see {job / (phase + ".log")}')
        state.update(status='complete', exit_code=0)
    except BaseException as exc:
        state.update(status='failed', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop_owned(client)
        stop_owned(server)
        state['finished_unix'] = time.time()
        atomic_json(job / 'state.json', state)


if __name__ == '__main__':
    main()
