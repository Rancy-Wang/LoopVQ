#!/usr/bin/env python3
"""Pure 0.5-bit Ouro-2.6B-Thinking recipe with auditable stage completion.

Run in the build environment inside an A10 tmux window. Credentials come from
the user's W&B/HF login, never from command-line arguments or this file.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "calibration"))
from reproduction_tracking import atomic_json, file_sha256, json_digest

BUNDLE_FILES = ("codebook.pt", "vq_v_codebook.pt", "k_rotation_qqt_r_h_pbr.pt",
                "v_rotation_sst_r_h_pbr.pt", "build_info.json")


def inventory(paths):
    return {str(path): {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in paths}


def validate_bundle(path):
    import torch
    info = json.loads((path / "build_info.json").read_text())
    expected = {"num_hidden_layers": 48, "total_ut_steps": 4, "bundle_layers": 192,
                "heads": 16, "head_dim": 128, "group_dim": 16, "abits": 8,
                "method": "loopvq", "v_method": "loopvq", "bits_per_channel": 0.5}
    if any(info.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Bundle metadata does not match the 0.5-bit recipe: {path}")
    if info.get("k_qweight") or info.get("k_cov") or info.get("k_norm") != "none":
        raise ValueError("TaSQ/normalization was unexpectedly enabled")
    for name in BUNDLE_FILES[:2]:
        blob = torch.load(path / name, map_location="cpu", weights_only=False)
        if blob.get("loop_coupled") != {"loops": 4, "channels": 4, "base_layers": 48}:
            raise ValueError(f"Unexpected loop coupling in {name}")
        if blob["forward"].shape != (192, 16, 128, 128):
            raise ValueError(f"Unexpected forward shape in {name}")
        for kind in ("forward", "inverse"):
            if not torch.equal(blob[kind], torch.eye(128).expand_as(blob[kind])):
                raise ValueError(f"Non-identity transform in pure LoopVQ: {name}/{kind}")
        if blob["mean"].shape != (192, 16, 128) or blob["mean"].any():
            raise ValueError(f"Invalid mean in {name}")
        if blob.get("pertoken_norm") or any(blob.get(key) is not None for key in ("chan_weight", "loop_scale")):
            raise ValueError(f"Unexpected normalization/scale in {name}")
        books = blob["codebooks"]
        if set(books) != {(slot, head) for slot in range(192) for head in range(16)}:
            raise ValueError(f"Missing/extra slot-head codebooks in {name}")
        for groups in books.values():
            if len(groups) != 8:
                raise ValueError(f"Unexpected group count in {name}")
            for tensor in groups:
                if tensor.shape != (256, 16) or tensor.dtype != torch.float16 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid codebook tensor in {name}")
        del blob
    for name in BUNDLE_FILES[2:4]:
        blob = torch.load(path / name, map_location="cpu", weights_only=False)
        if blob.get("objective") != "identity" or set(blob["layers"]) != set(range(192)):
            raise ValueError(f"Invalid rotation bundle: {name}")
        for entry in blob["layers"].values():
            if not torch.equal(entry["rotation"], torch.eye(128)):
                raise ValueError(f"Non-identity rotation: {name}")
    return inventory([path / name for name in BUNDLE_FILES])


def run_stage(root, name, command, recipe_hash, outputs, validate):
    marker = root / "results" / f"{name}.complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["recipe_sha256"] != recipe_hash or old["command"] != command:
            raise ValueError(f"Refusing to reuse a different {name} recipe")
        current = validate()
        if old["outputs"] != current:
            raise ValueError(f"Completed {name} outputs changed; do not silently reuse them")
        print(f"[runner] verified completed stage {name}", flush=True)
        return
    # FT may resume only from its own validated boundary; the other stages restart incomplete work.
    actual_command = list(command)
    checkpoint = root / "checkpoints" / "finetune-latest.pt"
    if name == "finetune" and checkpoint.exists():
        actual_command += ["--resume", str(checkpoint)]
    started = time.time()
    record = {"stage": name, "command": command, "actual_command": actual_command,
              "recipe_sha256": recipe_hash, "host": platform.node(), "started_unix": started}
    atomic_json(root / "results" / f"{name}.running.json", record)
    log_path = root / "logs" / f"{name}.log"
    with log_path.open("a", buffering=1) as logfile:
        logfile.write(f"\nRUN_START {started} {json.dumps(actual_command)}\n")
        process = subprocess.Popen(actual_command, cwd=REPO, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        record["pid"] = process.pid
        atomic_json(root / "results" / f"{name}.running.json", record)
        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)
        code = process.wait()
    record.update(exit_code=code, elapsed_seconds=time.time() - started,
                  ended_unix=time.time(), log=str(log_path))
    atomic_json(root / "results" / f"{name}.last-attempt.json", record)
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}; see {log_path}")
    record["outputs"] = validate()
    if set(record["outputs"]) != {str(path) for path in outputs}:
        raise ValueError(f"Unexpected completion manifest for {name}")
    atomic_json(marker, record)
    print(f"[runner] {name} complete in {record['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True, help="task storage directory outside Git")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpqa-revision", help="resolve and lock the current revision when omitted")
    parser.add_argument("--wandb-entity", default="529513043-tsinghua-university")
    parser.add_argument("--wandb-project", default="loopvq-ouro-2p6b-thinking")
    parser.add_argument("--wandb-group", default="2026-11-23-loopvq-reproduction")
    parser.add_argument("--stop-after", choices=("calibration", "fit", "finetune"), default="finetune")
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        parser.error("Run training inside a target-machine tmux window")
    root, model = Path(args.root).resolve(), Path(args.model).resolve()
    for directory in ("cache", "artifacts", "checkpoints", "logs", "results", "wandb"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    with (root / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["WANDB_DIR"] = str(root / "wandb")
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["LOOPVQ_V_LAYOUT"] = "contiguous"
        provenance_path = root / "results" / "training-provenance.json"
        previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        if args.gpqa_revision:
            revision = args.gpqa_revision
        elif previous:
            revision = previous["gpqa_revision"]
        else:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info("Idavidrein/gpqa").sha
        cfg = json.loads((model / "config.json").read_text())
        if any(cfg.get(key) != value for key, value in {
                "num_hidden_layers": 48, "total_ut_steps": 4, "num_key_value_heads": 16,
                "head_dim": 128}.items()):
            raise ValueError("Expected the complete Ouro-2.6B R4 geometry")
        tracked_clean = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode == 0
        if not tracked_clean:
            raise ValueError("Commit the candidate before running the recorded training recipe")
        model_files = sorted(p for p in model.iterdir() if p.is_file() and
                             (p.suffix in (".safetensors", ".json", ".py") or p.name.endswith(".model")))
        if not any(p.suffix == ".safetensors" for p in model_files):
            raise ValueError("No model safetensors found")
        provenance = {"model": str(model), "model_files": inventory(model_files),
                      "gpqa_revision": revision, "repository": str(REPO),
                      "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                      "python": sys.executable, "host": platform.node(),
                      "wandb_base_url": os.environ.get("WANDB_BASE_URL"),
                      "packages": {name: importlib.metadata.version(name) for name in
                                   ("torch", "transformers", "datasets", "wandb")},
                      "recipe": {"G": 16, "Tc": 4, "abits": 8, "n_calib": 16,
                                 "seq_len": 2048, "seed": 0, "kmeans_iters": 15,
                                 "n_val": 4, "epochs": 4, "lr": 0.001, "sink": 8,
                                 "compound": 1, "objective": "joint", "rep_chunk": 512}}
        if previous and previous != provenance:
            raise ValueError("Model/code/environment provenance changed; use a new task root")
        atomic_json(provenance_path, provenance)
        gpu_info = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader",
            "-i", args.gpu], text=True).strip()
        atomic_json(root / "results" / "training-launch.json", {
            "started_unix": time.time(), "pid": os.getpid(), "host": platform.node(),
            "tmux_pane": os.environ.get("TMUX_PANE"), "gpu": gpu_info,
            "cuda_visible_devices": args.gpu, "python": sys.executable,
            "git_sha": provenance["git_sha"], "wandb_base_url": os.environ.get("WANDB_BASE_URL")})
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         group=args.wandb_group, job_type="preflight", mode="online",
                         config={"provenance_sha256": json_digest(provenance), "git_sha": provenance["git_sha"]})
        if run.offline:
            raise RuntimeError("Online W&B is required before full training")
        run.log({"online_preflight": 1})
        url, run_id = run.url, run.id
        run.finish()
        wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{run_id}")
        atomic_json(root / "results" / "wandb-preflight.json", {"url": url, "run_id": run_id})
        print(f"[runner] verified W&B run: {url}", flush=True)
        data_manifest = root / "cache" / "calibration-data.json"
        cache = root / "cache" / "ouro_2p6b_think_n16_s2048_p2048_seed0_gpqa_nodiamond.pt"
        initial = root / "artifacts" / "ouro_2p6b_think_loopvq8b_g16_lvgq"
        final = root / "artifacts" / "ouro_2p6b_think_loopvq8b_g16_lvgqe2e"
        build = [sys.executable, str(REPO / "calibration" / "build_ouro_bundle.py"),
                 "--model", str(model), "--n-calib", "16", "--calib-seq-len", "2048",
                 "--calib-positions", "2048", "--calib-data", "gpqa_nodiamond",
                 "--gpqa-revision", revision, "--data-manifest", str(data_manifest),
                 "--calib-cache", str(cache), "--abits", "8", "--group-dim", "16",
                 "--method", "loopvq", "--loops-coupled", "4", "--v-method", "loopvq",
                 "--kmeans-iters", "15", "--seed", "0", "--grad-ckpt"]
        def tracking(stage):
            return ["--wandb-mode", "online", "--wandb-entity", args.wandb_entity,
                    "--wandb-project", args.wandb_project, "--wandb-group", args.wandb_group,
                    "--wandb-name", f"g16-{stage}", "--metrics-jsonl", str(root / "logs" / f"{stage}.jsonl")]
        recipe_hash = json_digest(provenance)
        def validate_cache():
            import torch
            blob = torch.load(cache, mmap=True, map_location="cpu", weights_only=True)
            for key in ("k", "v", "gk", "gv"):
                if blob[key].shape != (192, 16, 1, 16, 2048, 128) or blob[key].dtype != torch.bfloat16:
                    raise ValueError(f"Unexpected calibration cache shape/dtype: {key}")
                for layer in blob[key]:
                    if not torch.isfinite(layer).all():
                        raise ValueError(f"Nonfinite calibration cache: {key}")
            return inventory([cache, data_manifest])
        run_stage(root, "calibration", build + ["--calib-only", "--out-dir", str(initial)] + tracking("calibration"),
                  recipe_hash, [cache, data_manifest], validate_cache)
        if args.stop_after == "calibration":
            return
        run_stage(root, "fit", build + ["--out-dir", str(initial)] + tracking("fit"), recipe_hash,
                  [initial / name for name in BUNDLE_FILES], lambda: validate_bundle(initial))
        if args.stop_after == "fit":
            return
        finetune = [sys.executable, str(REPO / "calibration" / "finetune_codebook.py"),
                    "--model", str(model), "--src", str(initial), "--dst", str(final),
                    "--calib-data", "gpqa_nodiamond", "--gpqa-revision", revision,
                    "--data-manifest", str(data_manifest), "--n-calib", "16", "--n-val", "4",
                    "--seq-len", "2048", "--grad-ckpt", "--lr", "0.001", "--epochs", "4",
                    "--seed", "0", "--sink", "8", "--compound", "1", "--rep-chunk", "512",
                    "--objective", "joint", "--train", "kv", "--trace", str(root / "results" / "finetune-trace.json"),
                    "--checkpoint", str(root / "checkpoints" / "finetune-latest.pt"),
                    "--provenance-manifest", str(provenance_path)] + tracking("finetune")
        run_stage(root, "finetune", finetune, recipe_hash, [final / name for name in BUNDLE_FILES],
                  lambda: validate_bundle(final))
        print(f"[runner] completed pure LoopVQ 0.5-bit training: {final}", flush=True)


if __name__ == "__main__":
    main()
