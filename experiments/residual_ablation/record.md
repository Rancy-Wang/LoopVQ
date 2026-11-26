# INT4 residual accuracy ablation, 2026-11-26

User authorized stages 0, 1 and 2. Subsequent clarification defers efficiency measurement. Training and accuracy are the current deliverables. Reuse original GPQA main-minus-Diamond revision 83022cefff930aea54f654c0b282e74b9eeda5c6 and exact token IDs. Freeze Ouro-2.6B-Thinking R4, Fisher initialization, original KL objective and all training hyperparameters.

## Six-part plan

1. Purpose. Isolate adding INT4 residual reconstruction and making the original final-output KL fine-tuning residual-aware.
2. Steps. Verify the existing baseline artifacts. Add fixed symmetric INT4 group32 residuals with FP16 scales. Keep original nearest-centroid assignment. Recompute detached residual encodings each training iteration and backpropagate through the codebook gather only. Compare frozen original book, residual-compensated original book, residual-aware continued book, and an ordinary KL continuation from the same original book with the same update budget. Evaluate BF16 and the ablations using matching generation protocols.
3. Scope. LoopVQ/calibration/finetune_codebook.py, a small shared residual codec, and experiments/residual_ablation scripts/tests. Local changes sync using Rancy-Wang/LoopVQ test. Training logs and data remain outside Git. No serving efficiency changes in this task.
4. Success. Identical original 12 train/4 validation sequences, length2048, Adam lr0.001, four epochs, seed0, sink8, rep_chunk512, compound1, joint K/V final-token-distribution KL. Finite gradients, residual refresh, export/load equivalence and an independent reconstruction check. Complete five Thinking benchmarks at seeds0/1/2, report sample-level changes and mean/SD. Improvements are measured outcomes, never required assertions.
5. Risks. Surrogate gradient through fixed residual encodings; FP16 export versus BF16 training gather; pre-RoPE inversion rounding; quantization feedback across loops; long-generation runtime; distinguishing numerical protocol differences from residual gains. Additional ordinary continuation controls extra optimizer steps.
6. Validation. CPU codec tests and py_compile; GPU codec/gradient/cache tests; small R4 inference and benchmark-scoring audit; then full runs. All long jobs run in tmux with persistent per-cell outputs and atomic completion markers. W&B receives training configuration, per-step/epoch KL, validation and progress, never raw prompts or credentials. Use actual empty GPUs on A10>A11>A12>A13, through the persistent X5 master.

## Candidate 1/10

Introduce the opt-in fixed INT4 residual codec and residual-aware forward in the existing training script. Disabled mode retains the original training operations. Verify on a spare A10 GPU before full training. This is a distinct residual-training study following completed serving-alignment work (26/26).

## Connections

X5 PTY25245, master /tmp/loopvq-residual-20261126-x5.sock. A10 PTY10517, A11 PTY78624, A12 PTY82582, A13 PTY58348. All target masters use the same X5 channel. Remote resource snapshots are refreshed before each launch.
