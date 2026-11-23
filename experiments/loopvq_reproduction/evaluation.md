# Thinking accuracy over an external SGLang server

`run_evaluation.py` uses the repository's two official scoring scripts, the checkpoint's chat template with `enable_thinking=True`, temperature 1, top-p 0.7 and 32,768 generation tokens. It evaluates AIME24 (30), AIME25 (30), GPQA-Diamond (198), HumanEval+ (164) and MBPP+ (378), with seeds 0, 1 and 2: 2,400 examples per method. Run the driver once for BF16 and once for pure LoopVQ 0.5-bit against their respective external servers. It does not launch a server or change its configuration.

The caller must verify the running server and provide a JSON manifest with `method` (`bf16` or `loopvq_0.5bit`), `sglang_git_sha`, `model_sha256`, and numeric `total_ut_steps: 4`. The quantized manifest also requires `bundle_sha256`. Include the actual launch command, context length 40,960, prefix window 64, recent window 256, GPU identity, and disabled speculative decoding in this manifest. These settings are the caller's responsibility: `/health` alone does not attest to them. Do not use the same manifest for different server configurations.

Use the same absolute `--root` outside Git for interrupted runs. Replace example paths with the target machine's verified paths. The endpoint is the server root, without `/v1/chat/completions`.

```bash
python experiments/loopvq_reproduction/run_evaluation.py \
  --model /absolute/path/Ouro-2.6B-Thinking \
  --base-url http://127.0.0.1:30000 --method bf16 \
  --server-manifest /absolute/path/bf16-server.json \
  --root /absolute/path/evaluation --num-concurrent 2 \
  --audit --benchmarks aime24 evalplus_humaneval --seeds 0 1
```

This makes four real short HTTP requests, one per client and seed, with a 64-token audit budget. It exercises official generation and scoring but its scores are not benchmark results. Only successful audited cells permit a full run with the same recipe. Inspect their saved payloads, prompt IDs, finish reasons and usage before starting the full run:

```bash
python experiments/loopvq_reproduction/run_evaluation.py \
  --model /absolute/path/Ouro-2.6B-Thinking \
  --base-url http://127.0.0.1:30000 --method bf16 \
  --server-manifest /absolute/path/bf16-server.json \
  --root /absolute/path/evaluation --num-concurrent 2
```

Repeat both commands with `--method loopvq_0.5bit` and its verified server manifest. `--benchmarks` and `--seeds` can select pending cells without changing the recipe. Full completion requires all five tasks and all three seeds for each method. Source, tokenizer, dependency, service identity or concurrency changes select a new recipe directory and require a fresh audit. `summary.json` reports each completed cell and three-seed mean/sample standard deviation (`ddof=1`); audit scores stay separate.

The default `--api-seed-mode explicit` sends 0/1/2 as the HTTP `seed` for both clients. `--api-seed-mode legacy` instead preserves the original reasoning client's hidden lm-eval default HTTP seed 1234; its nominal filenames and client RNG still use 0/1/2. EvalPlus continues to send 0/1/2 in both modes. These are distinct recipes; the driver never treats a filename as proof of the API seed. Exact server determinism across scheduling or batch changes is not implied.

Each cell stores `http/attempts/` with every request and successful or failed response, payload, timing, input IDs, stop strings, finish reason and token counts. `input_token_ids_source` explicitly identifies IDs reconstructed from the pinned checkpoint template; they are not claimed to be returned by the server. The proxy checks their length against server `usage.prompt_tokens`. A successful response is written to `http/responses/` before being returned to the client. On restart, the identical payload is served from that cache and the official scorer runs again. Failures remain in the attempt history; incomplete cells never receive `complete.json`. EvalPlus additionally preserves incremental generation and scoring rows, including failures. AIME/GPQA retain raw responses in the HTTP audit immediately and their per-example scores/documents in the final `_samples.jsonl`.

No prompt, generated code, token IDs, samples or logs are uploaded to W&B by this driver. Set `WANDB_PROJECT` and optionally `WANDB_ENTITY`, or corresponding flags, to enable online aggregate counters and scores. Raw evidence stays in the local evaluation directory. The driver's terminal capture is disabled in W&B.

CPU transport validation (not a model or real benchmark audit):

```bash
python experiments/loopvq_reproduction/eval_test_io.py
```
