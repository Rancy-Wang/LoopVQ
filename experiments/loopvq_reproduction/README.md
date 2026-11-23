# Ouro-2.6B-Thinking: pure LoopVQ 0.5-bit reproduction

This runner uses the upstream recipe: GPQA main minus Diamond, 16 raw-text
sequences of 2048 tokens, seed 0, CE Fisher-weighted k-means with 15 iterations,
G16 / Tc4 / 256 entries for K and V, then joint KL refinement with 12 train and
4 held-out sequences, Adam 1e-3, 4 epochs, sink 8 and replay chunk 512. It does
not compute TaSQ weights or enable TaSQ/normalization.

Run from the dedicated build environment in an A10 **tmux** window after
`wandb login` and a GPU/resource check. Use an absolute storage path outside Git:

```bash
python experiments/loopvq_reproduction/run_training.py \
  --model /homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking \
  --root /homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction \
  --gpu 0
```

The default W&B project is
`529513043-tsinghua-university/loopvq-ouro-2p6b-thinking`.
The online preflight writes a small run, finishes it, and reads it back before
launching expensive work. Credentials must come from the user's login or private
environment, not command arguments. Raw examples and input token IDs are kept
locally; only metrics/configuration are sent to W&B.

`--stop-after calibration` or `--stop-after fit` supports staged validation.
Run the same command again to resume. A file lock prevents duplicate runners;
stage completion requires a zero exit code, validated tensors and matching
content hashes. Different model/code/environment manifests require a new root.
The first run resolves and locks the GPQA revision; `--gpqa-revision` can pin it
explicitly. Full model hashes, code SHA, data fingerprints and local calibration
IDs are recorded for provenance.

The stages write `logs/*.log`, append-only `logs/*.jsonl`, and atomic
`results/*.complete.json`. W&B gets calibration sequence and fit-layer progress,
per-update train KL, initial/epoch held-out KL, the selected best KL, timings and
stage ETA estimates. GPU metrics come from the SDK. Lost connectivity does not
replace the local record; SDK upload failures are reported and local W&B data
can be synchronized later with the installed SDK's `wandb sync` command.

FT saves `checkpoints/finetune-latest.pt` after initial validation and each full
epoch validation. It contains current FP32 K/V codebooks, best codebooks, Adam
state, all RNG states, recipe/input/source hashes, completed epochs, step count,
trace and previous W&B run identity. Interrupted partial epochs are recomputed
from the latest complete boundary. A resumed process creates a new linked W&B
segment so repeated computations are not silently appended as new training
updates. These checkpoints are for training recovery, not inference.

Inference uses the original model plus
`artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e/`, which contains the two FP16
codebook dictionaries, two identity-rotation compatibility files, and
`build_info.json`. The initial bundle remains in the corresponding directory
without the `e2e` suffix. Quantized indices are created by the server at runtime.

Validation commands:

```bash
python -m unittest discover -s experiments/loopvq_reproduction -p 'test_*.py' -v
python -m py_compile calibration/reproduction_tracking.py \
  calibration/build_ouro_bundle.py calibration/finetune_codebook.py \
  experiments/loopvq_reproduction/run_training.py
```

The CPU test checks Adam/RNG continuation against uninterrupted training and
rejects a changed recipe. It does not replace an actual model/GPU epoch-resume
comparison, full calibration, or service validation.
