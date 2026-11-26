"""Verify quantization boundaries and the residual-frozen training derivative."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration"))
from residual_codec import encode_residual, decode_residual


class ResidualTest(unittest.TestCase):
    def test_round_trip_and_zero(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            actual = torch.linspace(-1, 1, 256, device=device).reshape(2, 128)
            base = torch.zeros_like(actual)
            code, scale = encode_residual(actual, base)
            result = decode_residual((code, scale))
            bound = scale.float().expand(2, 4, 32).reshape_as(actual) / 2 + 1e-3
            self.assertTrue(torch.all((result - actual).abs() <= bound))
            self.assertTrue(torch.equal(decode_residual(encode_residual(actual, actual)), base))
            self.assertEqual(code.dtype, torch.int8)
            self.assertEqual(scale.dtype, torch.float16)

    def test_gradient_and_refresh(self):
        actual = torch.linspace(-2, 2, 128).reshape(1, 128)
        base = torch.nn.Parameter(torch.zeros_like(actual))
        encoded = encode_residual(actual, base)
        reconstructed = base + decode_residual(encoded)
        loss = (reconstructed - actual).square().sum()
        loss.backward()
        torch.testing.assert_close(base.grad, 2 * (reconstructed.detach() - actual))
        self.assertGreater(base.grad.abs().sum().item(), 0)
        self.assertFalse(encoded[1].requires_grad)
        fresh = encode_residual(actual, base.detach() + 0.2)
        self.assertFalse(torch.equal(encoded[0], fresh[0]))

    def test_invalid_group(self):
        with self.assertRaises(ValueError):
            encode_residual(torch.zeros(31), torch.zeros(31))


if __name__ == "__main__":
    unittest.main()
