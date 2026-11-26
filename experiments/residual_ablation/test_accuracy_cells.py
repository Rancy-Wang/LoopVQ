import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from accuracy_cells import (ARMS, BASE_SHA, atomic_json, claim, collect_cell,
                            file_hash, initialize, key, register_done)
from summarize_accuracy import summarize
from cell_worker import drain


class CellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        initialize(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def make_cell(self, arm, seed, correct, dataset='frozen-data'):
        method = ARMS[arm]
        job = self.root / 'jobs' / arm
        recipe = job / 'evaluation' / method / 'recipe'
        cell = recipe / 'full/aime24' / str(seed)
        cell.mkdir(parents=True, exist_ok=True)
        source = {'server': {'method': method, 'implementation': 'transformers_reference',
            'inference_git_sha': BASE_SHA, 'total_ut_steps': 4, 'tp': 2, 'prefix_tokens': 8,
            'recent_tokens': 64, 'page_size': 8, 'cuda_graph': True,
            'attention_backend': 'pytorch_efficient_sdpa', 'materialized_bf16_cache': True,
            'model_sha256': 'model', 'bundle_sha256': None if arm == 'bf16' else 'book', 'residual': None},
            'temperature': 1.0, 'top_p': 0.7, 'max_tokens': 32768, 'num_concurrent': 1,
            'api_seed_mode': 'explicit', 'tokenizer_files': {}, 'sources': {}, 'packages': {}}
        atomic_json(recipe / 'recipe.json', source)
        stem = f'{method}_{seed}_aime24'
        atomic_json(cell / (stem + '.json'), {'aime24': {'exact_match,none': correct / 30}})
        (cell / (stem + '_samples.jsonl')).write_text(''.join(json.dumps(
            {'doc_id': i, 'exact_match': float(i < correct)}) + '\n' for i in range(30)))
        atomic_json(cell / (stem + '_protocol.json'), {'api_seed': seed, 'temperature': 1.0,
            'top_p': 0.7, 'max_gen_toks': 32768, 'limit': None, 'dataset_documents_sha256': dataset})
        scores = collect_cell(cell, stem, 'aime24', 30)
        atomic_json(cell / 'complete.json', {'benchmark': 'aime24', 'seed': seed, 'audit': False,
            'method': method, 'exit_code': 0, 'scores': scores})
        owner = {'worker': 'main-' + arm, 'arm': arm, 'benchmark': 'aime24', 'seed': seed, 'job': str(job)}
        register_done(self.root, owner)
        return cell

    def test_ownership_is_exclusive_and_sticky(self):
        a, lock_a = claim(self.root, 'helper-a', 'bf16', 'evalplus_humaneval')
        b, lock_b = claim(self.root, 'helper-b', 'bf16', 'evalplus_humaneval')
        self.assertNotEqual((a['benchmark'], a['seed']), (b['benchmark'], b['seed']))
        lock_a.close()
        again, resumed_lock = claim(self.root, 'helper-a', 'bf16', 'aime25')
        self.assertEqual(a, again)
        resumed_lock.close()
        lock_b.close()

    def test_incomplete_table_waits_for_all_seeds(self):
        self.make_cell('bf16', 0, 15)
        self.make_cell('bf16', 1, 18)
        partial = summarize(self.root)
        self.assertIsNone(partial['table']['aime24']['bf16']['mean_accuracy_percent'])
        self.make_cell('bf16', 2, 21)
        for seed, n in enumerate((12, 15, 18)):
            self.make_cell('loopvq', seed, n)
        full = summarize(self.root)
        self.assertFalse(full['complete'])
        self.assertEqual(full['table']['aime24']['bf16']['mean_accuracy_percent'], 60)
        self.assertEqual(full['table']['aime24']['bf16']['std_accuracy_percent_ddof1'], 10)
        self.assertEqual([p['delta_percentage_points'] for p in full['paired_vs_loopvq']['aime24/bf16']], [10] * 3)

    def test_corrupted_score_and_different_data_are_rejected(self):
        cell = self.make_cell('bf16', 0, 15)
        self.make_cell('loopvq', 0, 15, dataset='different')
        with self.assertRaisesRegex(ValueError, 'examples differ'):
            summarize(self.root)
        (self.root / 'dispatch/done' / (key('loopvq', 'aime24', 0) + '.json')).unlink()
        atomic_json(cell / 'bf16_0_aime24.json', {'aime24': {'exact_match,none': 1.0}})
        with self.assertRaisesRegex(ValueError, 'Changed completion'):
            summarize(self.root)

    def test_drain_waits_until_response_is_saved(self):
        job = self.root / 'jobs/bf16'
        cell = job / 'evaluation/method/recipe/full/aime24/0'
        cell.mkdir(parents=True)
        atomic_json(cell / 'running.json', {'pid': 12346})
        atomic_json(job / 'state.json', {'client_pid': 12345})
        atomic_json(job / 'server-ready.json.progress.json', {'status': 'finished', 'updated_unix': 100.9})
        actions = []

        def command(pid):
            return f'run_evaluation.py {job / "evaluation"}' if pid == 12345 else f'run_reasoning_api.py {cell}'

        def sleep(_):
            if not list((cell / 'http/responses').glob('*.json')):
                actions.append('saved')
                atomic_json(cell / 'http/responses/answer.json', {'response': {'created': 100}})
            else:
                atomic_json(job / 'state.json', {'client_pid': 12345, 'finished_unix': 101})

        with patch('cell_worker.command_line', side_effect=command), \
             patch('cell_worker.os.kill', side_effect=lambda p, s: actions.append(('signal', p, s))), \
             patch('cell_worker.os.killpg', side_effect=lambda p, s: actions.append(('group', p, s))), \
             patch('cell_worker.time.sleep', side_effect=sleep):
            drain(self.root, 'bf16', self.root / 'worker-state.json')
        self.assertEqual(actions[0], ('signal', 12346, signal.SIGSTOP))
        self.assertLess(actions.index('saved'), actions.index(('group', 12345, signal.SIGTERM)))
        self.assertTrue((job / 'drain-complete.json').exists())


if __name__ == '__main__':
    unittest.main()
