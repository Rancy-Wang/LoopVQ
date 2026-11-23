"""CPU recovery tests; GPU model validation is a separate acceptance gate."""
import argparse
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration"))
from reproduction_tracking import (Tracking, atomic_json, atomic_torch_save,
                                   checkpoint_state, restore_checkpoint)


class TrackingTests(unittest.TestCase):
    def test_failed_atomic_json_preserves_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            atomic_json(path, {"completed_epoch": 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"completed_epoch": 1})

    def test_disabled_sdk_still_records_local_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            args = argparse.Namespace(metrics_jsonl=str(path), wandb_mode="disabled")
            tracking = Tracking(args, "finetune", {"seed": 0})
            tracking.log({"global_step": 1, "train/kl": 0.5})
            tracking.finish(0)
            tracking.finish(1)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r.get("event") for r in records], ["start", None, "finish"])
            self.assertEqual(records[-1]["exit_code"], 0)
            self.assertEqual(records[1]["train/kl"], 0.5)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is required for recovery verification")
    def test_adam_rng_resume_matches_continuous_training(self):
        import numpy as np
        import torch
        random.seed(81)
        np.random.seed(81)
        torch.manual_seed(81)
        parameters = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        recipe = {"epochs": 2, "seed": 81, "data_hash": "fixed-test-inputs"}

        def train(ps, opt):
            losses = []
            for _ in range(3):
                target = torch.randn(3, 4) + random.random() + float(np.random.rand())
                opt.zero_grad(set_to_none=True)
                loss = sum((p - target).square().sum() for p in ps)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            return losses

        train(parameters, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_torch_save(path, checkpoint_state(
                *parameters, optimizer, recipe, 1, 3, 0.25,
                tuple(p.detach().clone() for p in parameters), {"curve": [(3, 0.25)]}, 1.0, {}))
            expected_losses = train(parameters, optimizer)
            expected_parameters = [p.detach().clone() for p in parameters]
            resumed = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
            resumed_optimizer = torch.optim.Adam(resumed, lr=1e-3)
            state = restore_checkpoint(path, *resumed, resumed_optimizer, recipe)
            self.assertEqual((state["completed_epochs"], state["global_step"]), (1, 3))
            self.assertEqual(state["trace"], {"curve": [(3, 0.25)]})
            self.assertEqual(train(resumed, resumed_optimizer), expected_losses)
            for got, expected in zip(resumed, expected_parameters):
                self.assertTrue(torch.equal(got, expected))
            before = [p.detach().clone() for p in resumed]
            with self.assertRaisesRegex(ValueError, "mismatch"):
                restore_checkpoint(path, *resumed, resumed_optimizer, {**recipe, "seed": 99})
            for got, expected in zip(resumed, before):
                self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
