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

Candidate1 99951b9 passed compilation. Local Python lacks torch; the three codec tests ran on A10 CPU and CUDA GPU4 and exited0. Residual-aware continuation started on A10 GPU4, ordinary continuation on A12 GPU4, each from the same final original bundle with the unmodified four-epoch optimizer recipe. Online W&B runs dcbebws8 and 9dvetlbu. Initial held-out KL approximately .0034 and .0478 respectively. These are cache-substitution KL values, not accuracy scores.

## Candidate 2/10

Add an independent Transformers TP2 accuracy reference with a preallocated materialized KV cache. Quantize only complete pages leaving the fixed 8+64 high-precision window. Encode all four loop states jointly only after a full forward has produced them. Reconstruction uses the same Coded BF16 gather and FP32 RoPE inverse/reapply as the original training. The codec is applied to the student's own generated KV. All ablation arms use the same model arithmetic and cache interface. Materializing decoded KV is deliberate for this accuracy-only experiment; it is not a compressed-memory or optimized-throughput measurement. Verify actual head partitioning and teacher logits against the native cache before launching evaluation.

Candidate2 1cb8c87: TP2 smokes passed on A12 (BF16 and original VQ) and A13 (VQ+residual), all exit0 and 8 local KV heads per rank. Training completed exit0 for both arms. Residual best validation KL .003168791386997327 at epoch1; ordinary continuation .0432494948618114 at epoch4. Training token SHA256 matches original: 2b6633839f961a1af46264bb50ae8b5fc877124dcaf2ab518d536986a90c1d6e.

## Candidate 3/10

Add independent codec-layout, exported-checkpoint and native-cache logit validation before full evaluation. Make evaluation manifest identify the actual Transformers reference rather than SGLang. Keep historical SGLang evaluation defaults compatible. Accuracy generation remains TP2, R4, HP8+64; this window differs from earlier HP64+256 accuracy experiments, so those outputs are not reused. Original GPQA-minus-Diamond calibration recipe remains sink8 (without a recent-window exception), as published and previously used.

Candidate3 39a2279 passed TP2 validation on A12 GPU0/1, exit0: native/custom cache logits bit-exact across prefill+4 decode steps; all 48 coupled layers K/V with/without residual match original Coded assign/decode bit-exact; FP16 exported books match selected best checkpoint bit-exact. W&B readback confirms both runs finished and matching training IDs.

## Candidate 4/10

Purpose: remove Python per-layer dispatch from the independent accuracy evaluator so long-generation evaluation is practical. Scope: hf_reference.py, graph validation and evaluation launch scripts only. Add reusable static KV storage and bucketed CUDA graphs, preserving R4, original projection/MLP/attention arithmetic, sampling and codec. Keep eager mode as an oracle; require captured versus uncaptured identical padded arithmetic to be bit-exact, and compare greedy generated IDs against dynamic eager for all three original arms. Record any native-padding rounding difference. Risks: TP collectives in graph capture, masking unused KV, graph reuse across requests, bucket transitions, and capture mutating cache positions. Test two requests and bucket crossings before launching the existing five-benchmark/three-seed harness. No efficiency measurements or SGLang edits.

Candidate4 0876de8: BF16 and original VQ graph smokes reached SMOKE_OK with native eager greedy agreement, bit-exact captured versus uncaptured padded logits, repeat-request agreement and 256-to-512 bucket transition agreement. The residual smoke uses the same checks; its completion marker is checked before full launch.

## Candidate 5/10

Add durable accuracy-job orchestration and explicit residual method names, model/bundle hashes, online W&B aggregate metrics, prompt-ID cross-checks, and current-generation progress records. Reuse existing audited benchmark clients and scorers. Each worker owns one free GPU pair, its server and client process groups, tmux window, logs and exit markers. Require all graph checks and frozen training-data checks before launching. No sample truncation in full runs; audit-only 64-token outputs are kept separate. Failed jobs retain responses for replay. Success requires complete 15 benchmark/seed cells per arm; until then report partial status only.

Candidate4 residual graph check failed at the boundary greedy-ID assertion on A13; retain graph-residual.log and exit1. Captured/uncaptured padded logits and repeated first request passed, but dynamic-shape eager boundary outputs differed. Candidate5 first adds forced-token logit diagnostics; full accuracy launch remains gated. Do not silently accept the failing graph path or mix eager and captured arms.

Candidate5 diagnostics: profiling a matching TP2-local attention shape shows dynamic no-mask SDPA dispatches aten::_scaled_dot_product_flash_attention, but graph padded-mask SDPA dispatches aten::_scaled_dot_product_efficient_attention (CUTLASS). This is an unintended backend variable. Candidate4 BF16/VQ numerical assertions printed SMOKE_OK but explicit distributed teardown remained alive; their known torchrun PIDs 2976467/2976470 were terminated after preserving outputs. This is not a clean successful exit.

## Candidate 6/10

Fix the confound by explicitly selecting PyTorch efficient SDPA for both dynamic eager and graph modes in the standalone accuracy reference. Re-run original strict greedy, repeated-request, boundary and captured/eager logit assertions for every arm; do not remove the failing assertion. Destroy graph objects before NCCL process groups to fix owned-worker cleanup. Native cache validation must also be rerun under this explicit backend. This changes the documented independent inference protocol; all arms share it, while original KL training remains unchanged. Full evaluation orchestration follows only after these checks pass.

Candidate6 bca9024: BF16, original VQ, original+INT4 and ordinary-continued VQ graph checks passed all original assertions and exited0. The residual-ft check is checked separately before launch. Native cache/export/layout validation under explicit efficient SDPA passed exit0 on A13 GPU6/7. A direct attention check at lengths250/255/256/257/300 is bit-exact with and without padding under this backend. Explicit graph reset resolved distributed teardown for completed checks.

## Candidate 7/10

Implement the previously planned accuracy worker and progress recording. Preserve the successful model path and original scorer protocols. Launch only after all five candidate6 graph outputs and the independent validation are present with exit0. Recheck free GPUs, host memory, ports, clean Git SHA, model/bundle hashes and frozen training recipe. Run short real HTTP/scoring audits, then full five benchmarks x seeds0/1/2. Accuracy cells and raw responses persist outside Git; W&B receives aggregates only. No efficiency metrics are measured. Review actual completed response evidence before reporting jobs started successfully.

Candidate7 3df02ee completed all five real HTTP/scoring audits (four audit cells per arm). Full generation started on A13 GPU2/3 (BF16), A10 GPU4/5 (original VQ), A12 GPU0/1 (frozen+residual), 4/5 (ordinary continuation), 6/7 (residual-aware continuation). A10 GPU2/3 became busy before startup; its rejected launch is preserved. All five candidate6 checks exited0. Exported-bundle KL revalidation exited0: ordinary approximately .0432, residual-aware approximately .0043 (4 decimal output), versus training-state best .0432495/.0031688. These are not accuracy scores. Per-hour followup automation ID loopvq. Current independent inference remains commit3df02ee.

## Candidate 8/10 — complete the requested accuracy table

1. Purpose: complete the five-method, five-benchmark table without SGLang; use spare GPU pairs and preserve all generated samples.
2. Steps: add a shared per-cell ownership/lock registry; drain existing single-concurrency clients by pausing only their owned benchmark subprocess, leaving the HTTP proxy/server alive until its current answer is durably saved. Then resume with cell workers using the identical original endpoint, manifest and evaluation recipe. Additional replicas claim unassigned cells. Retain failures and prior raw responses.
3. Scope: new scripts under experiments/residual_ablation for cell dispatch, draining, checked aggregation and focused CPU tests. Do not edit hf_reference.py, codec/training math, run_evaluation.py or evaluation_io.py; validate they match3df02ee. Use LoopVQ/test via Git; no SGLang.
4. Success: no duplicate cell assignment; partial cells stay with their owning worker and recipe; every table number requires all three complete seeds and exact expected counts (30/30/198/164/378). Every score is checked against saved per-example outcomes and result hashes. Table reports mean and sample SD in percent. Frozen+INT4 reuses the original book; only ordinary and residual-aware continuation trained new books.
5. Risks: long in-flight requests delay drain; stopped benchmark clients require explicit termination/continuation; completed versus partial output confusion; cross-replica provenance and leftover claims. Never kill unrelated jobs, discard successful answers, shorten generations, or combine incompatible protocols. A replica's host/port is recorded separately; no throughput comparison is made.
6. Validation: CPU tests for cell locking, resumable ownership, incomplete exclusion, score/hash checks and cross-arm protocol checks; compile and diff checks. Pilot extra workers on currently empty A13 GPU4/5 and6/7. Verify original workers drain cleanly and resumed proxies replay existing responses. Monitor tmux, PIDs, status and W&B. Deliver JSON/CSV/Markdown table plus per-seed sample-level deltas, synchronizing results locally; pending entries remain explicitly pending until full completion.

Candidate8 d74647a passed four CPU tests locally and remotely. Extra BF16/VQ workers on A13 GPU4/5 and6/7 passed real audits and entered HumanEval+ seed0. Original VQ and ordinary continuation drained successfully, each preserving three complete answers; old clients terminated only after durable response saving. Immediate rebinding preflight rejected the just-released localhost ports (Errno98), although no listener remained by inspection. Failed worker states/logs and drain-complete evidence are preserved. Other original workers continue draining their in-flight answer.

## Candidate 9/10

Correct the cell worker's port preflight to use SO_REUSEADDR, matching HTTPServer, while still rejecting an active listener. No inference, benchmark, recipe, ownership or output-budget changes. Add an actual local socket test covering an active listener and recently closed accepted connection. Compile and rerun CPU tests, synchronize through test, and restart only workers that stopped at this known port preflight after a verified drain. Preserve original recipes and confirm cached answers are replayed, not regenerated. Existing helpers and in-flight drains remain undisturbed.
