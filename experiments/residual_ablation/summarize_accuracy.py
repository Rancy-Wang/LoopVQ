"""Produce the requested five-column table only from complete, checked cells."""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from accuracy_cells import (ARMS, BASE_SHA, BENCH_LABELS, COUNTS, LABELS,
                            atomic_json, checked_cell, file_hash, key)


def semantic_recipe(recipe):
    server = recipe['server']
    expected = {'implementation': 'transformers_reference', 'inference_git_sha': BASE_SHA,
                'total_ut_steps': 4, 'tp': 2, 'prefix_tokens': 8, 'recent_tokens': 64,
                'page_size': 8, 'cuda_graph': True, 'attention_backend': 'pytorch_efficient_sdpa',
                'materialized_bf16_cache': True}
    if any(server.get(k) != v for k, v in expected.items()):
        raise ValueError('Cell uses an incompatible inference protocol')
    for k, v in {'temperature': 1.0, 'top_p': 0.7, 'max_tokens': 32768,
                 'num_concurrent': 1, 'api_seed_mode': 'explicit'}.items():
        if recipe.get(k) != v:
            raise ValueError(f'Incompatible generation setting: {k}')
    return {k: recipe[k] for k in ('tokenizer_files', 'sources', 'packages')} | {
        'model_sha256': server['model_sha256'], **expected}


def summarize(root):
    results = {'updated_unix': time.time(), 'complete': False, 'cells': {}, 'table': {},
               'paired_vs_loopvq': {}, 'workers': {}}
    reference = None
    bundle_identity = {}
    datasets = {}
    outcomes = {}
    for path in sorted((root / 'dispatch/done').glob('*.json')):
        record = json.loads(path.read_text())
        arm, bench, seed = record['arm'], record['benchmark'], record['seed']
        if arm not in ARMS or bench not in COUNTS or seed not in (0, 1, 2):
            raise ValueError(f'Unknown completed cell: {path}')
        name = key(arm, bench, seed)
        if name in results['cells']:
            raise ValueError('Duplicate completed cell')
        recipe_path = Path(record['recipe'])
        if file_hash(recipe_path) != record['recipe_sha256']:
            raise ValueError('Recorded recipe changed')
        recipe = json.loads(recipe_path.read_text())
        current = semantic_recipe(recipe)
        if reference is None:
            reference = current
        elif current != reference:
            raise ValueError('Model, tokenizer, scorer code or packages differ across cells')
        server = recipe['server']
        if server['method'] != ARMS[arm]:
            raise ValueError('Recipe method differs from the arm')
        identity = {k: server.get(k) for k in ('bundle_sha256', 'residual')}
        if arm in bundle_identity and identity != bundle_identity[arm]:
            raise ValueError('Codebook or residual setting changed within an arm')
        bundle_identity[arm] = identity
        scores, values, data_identity = checked_cell(Path(record['cell']), ARMS[arm], bench, seed)
        if scores != record['scores'] or data_identity != record['data_identity']:
            raise ValueError('Registered completion evidence changed')
        data_key = (bench, seed)
        if data_key in datasets and datasets[data_key] != data_identity:
            raise ValueError('Benchmark examples differ across methods')
        datasets[data_key] = data_identity
        outcomes[name] = values
        results['cells'][name] = {**record, 'scores': scores}
    for bench in COUNTS:
        results['table'][bench] = {}
        for arm in ARMS:
            cells = [results['cells'].get(key(arm, bench, seed)) for seed in range(3)]
            entry = {'completed_seeds': [seed for seed, cell in enumerate(cells) if cell],
                     'n_per_seed': COUNTS[bench], 'mean_accuracy_percent': None,
                     'std_accuracy_percent_ddof1': None}
            if all(cells):
                scores = [cell['scores']['accuracy_percent'] for cell in cells]
                entry.update(mean_accuracy_percent=statistics.mean(scores),
                    std_accuracy_percent_ddof1=statistics.stdev(scores), seed_accuracies_percent=scores)
            results['table'][bench][arm] = entry
        for arm in ARMS:
            if arm == 'loopvq':
                continue
            paired = []
            for seed in range(3):
                a = outcomes.get(key('loopvq', bench, seed))
                b = outcomes.get(key(arm, bench, seed))
                if a is None or b is None:
                    continue
                if set(a) != set(b):
                    raise ValueError('Paired sample identities differ')
                paired.append({'seed': seed, 'improved_ids': [i for i in a if b[i] > a[i]],
                    'regressed_ids': [i for i in a if b[i] < a[i]],
                    'delta_percentage_points': 100 * (sum(b.values()) - sum(a.values())) / len(a)})
            if len(paired) == 3:
                results['paired_vs_loopvq'][f'{bench}/{arm}'] = paired
    for state_path in (root / 'workers').glob('*/state.json'):
        results['workers'][state_path.parent.name] = json.loads(state_path.read_text())
    results['complete'] = len(results['cells']) == len(ARMS) * len(COUNTS) * 3
    return results


def write_outputs(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'accuracy.json', result)
    columns = ['Benchmark', *LABELS.values()]
    rows = []
    for bench, label in BENCH_LABELS.items():
        row = [label]
        for arm in ARMS:
            x = result['table'][bench][arm]
            row.append(f"{x['mean_accuracy_percent']:.2f} ± {x['std_accuracy_percent_ddof1']:.2f}" if
                       x['mean_accuracy_percent'] is not None else f"待完成（{len(x['completed_seeds'])}/3 seeds）")
        rows.append(row)
    with (directory / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    text = ('# LoopVQ 独立推理准确率\n\n'
        '准确率单位为 %，完整单元格显示三个 seed 的均值 ± 样本标准差。'
        'AIME/GPQA 使用 exact match，HumanEval+/MBPP+ 使用完整 plus 测试的 pass@1。'
        '模型为 Ouro-2.6B-Thinking，TP2，四个 loop，HP8+64，temperature1、top-p0.7、输出上限32768。'
        '全部使用独立 Transformers 推理，不使用 SGLang。\n\n')
    text += '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    text += ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows)
    text += ('\n原码本＋INT4 残差复用原始码本；普通 KL 续训为相同额外训练步数的对照；'
             '残差感知续训采用已导出的 FP16 码本。审计短输出、部分样本和训练 KL 均不计入此表。\n\n'
             f"已核验完整单元：{len(result['cells'])}/75。每组需2,400个完整评分样本。\n")
    (directory / 'accuracy.md').write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.root)
    write_outputs(result, args.output)
    print('complete=', result['complete'], 'verified_cells=', len(result['cells']))


if __name__ == '__main__':
    main()
