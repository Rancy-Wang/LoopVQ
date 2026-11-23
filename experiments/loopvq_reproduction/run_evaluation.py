#!/usr/bin/env python3
"""Five Thinking tasks against an already running BF16 or pure 0.5-bit server.

First run --audit --benchmarks aime24 evalplus_humaneval --seeds 0 1.
Then omit --audit for all five tasks and all three seeds. Both phases use the
same --root and server manifest. HTTP responses are saved immediately and reused
after interruption; the official scorers are rerun over the complete sample set.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from evaluation_io import AuditProxy, atomic_json, digest, file_hash

REPO = Path(__file__).resolve().parents[2]
COUNTS = {"aime24": 30, "aime25": 30, "gpqa_diamond": 198,
          "evalplus_humaneval": 164, "evalplus_mbpp": 378}


def client_kind(bench):
    return "evalplus" if bench.startswith("evalplus_") else "lm_eval"


def collect_cell(directory, stem, bench, expected_count):
    path = directory / f"{stem}.json"
    result = json.loads(path.read_text())
    samples = [json.loads(line) for line in (directory / f"{stem}_samples.jsonl").read_text().splitlines()]
    if len(samples) != expected_count:
        raise ValueError(f"Incomplete {bench}: {len(samples)}/{expected_count} scored examples")
    identities = [row.get("task_id", row.get("doc_id")) for row in samples]
    if None in identities or len(set(identities)) != len(identities):
        raise ValueError(f"Missing or duplicated example IDs in {bench}")
    metrics = result[bench]
    if client_kind(bench) == "evalplus":
        score = metrics["pass@1,plus"]
    else:
        keys = [key for key in metrics if key.split(",")[0] == "exact_match"]
        if len(keys) != 1:
            raise ValueError(f"Ambiguous exact-match metric in {bench}: {keys}")
        score = metrics[keys[0]]
    return {"accuracy_percent": 100 * score, "n": len(samples),
            "result_sha256": file_hash(path),
            "samples_sha256": file_hash(directory / f"{stem}_samples.jsonl"),
            "protocol_sha256": file_hash(directory / f"{stem}_protocol.json")}


def record_audit(coverage, coverage_path, bench, seed, api_seed, marker):
    entry = {"kind": client_kind(bench), "seed": seed, "api_seed": api_seed,
             "marker": str(marker), "marker_sha256": file_hash(marker)}
    coverage[:] = [x for x in coverage if x["marker"] != str(marker)] + [entry]
    atomic_json(coverage_path, coverage)


def summarize(root, method, run):
    summary = {"method": method, "cells": {}, "benchmarks": {}}
    for bench in COUNTS:
        values = []
        for seed in (0, 1, 2):
            directory = root / "full" / bench / str(seed)
            marker = directory / "complete.json"
            if not marker.exists():
                continue
            cell = json.loads(marker.read_text())
            checked = collect_cell(directory, f"{method}_{seed}_{bench}", bench, COUNTS[bench])
            if checked != cell["scores"]:
                raise ValueError(f"Completed cell changed: {directory}")
            values.append(checked["accuracy_percent"])
            summary["cells"][f"{bench}/{seed}"] = checked
        if len(values) == 3:
            summary["benchmarks"][bench] = {"mean_accuracy_percent": statistics.mean(values),
                                           "std_accuracy_percent_ddof1": statistics.stdev(values),
                                           "seeds": [0, 1, 2], "n_per_seed": COUNTS[bench]}
            if run:
                run.log({f"{bench}/mean_accuracy_percent": statistics.mean(values),
                         f"{bench}/std_accuracy_percent_ddof1": statistics.stdev(values)})
    summary["complete"] = len(summary["cells"]) == 15
    atomic_json(root / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exact local checkpoint used by the server")
    parser.add_argument("--base-url", required=True, help="server root, e.g. http://127.0.0.1:30000")
    parser.add_argument("--method", required=True, choices=("bf16", "loopvq_0.5bit"))
    parser.add_argument("--server-manifest", required=True,
                        help="JSON: method, sglang_git_sha, model_sha256, total_ut_steps; "
                             "quantized server also requires bundle_sha256")
    parser.add_argument("--root", required=True, help="persistent evaluation directory outside Git")
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(COUNTS), default=list(COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--api-seed-mode", choices=("explicit", "legacy"), default="explicit")
    parser.add_argument("--num-concurrent", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=172800)
    parser.add_argument("--audit", action="store_true", help="one real HTTP example per cell; never a benchmark score")
    parser.add_argument("--audit-max-tokens", type=int, default=64)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-group", default="2026-11-23-loopvq-reproduction")
    args = parser.parse_args()
    if (any(seed not in (0, 1, 2) for seed in args.seeds) or args.num_concurrent < 1
            or args.audit_max_tokens < 1 or len(args.seeds) != len(set(args.seeds))):
        parser.error("Use unique seeds from 0/1/2, positive concurrency and positive audit budget")
    model = Path(args.model).resolve()
    server = json.loads(Path(args.server_manifest).read_text())
    required = ("method", "sglang_git_sha", "model_sha256", "total_ut_steps")
    if any(not server.get(k) for k in required) or server["total_ut_steps"] != 4 or server["method"] != args.method:
        raise ValueError("Server manifest missing identity fields or does not match full-depth method")
    if args.method != "bf16" and not server.get("bundle_sha256"):
        raise ValueError("Quantized server manifest needs bundle_sha256")
    with urlopen(args.base_url.rstrip("/") + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("External server is not healthy")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if "enable_thinking" not in (tokenizer.chat_template or ""):
        raise ValueError("Expected the Ouro-Thinking checkpoint's enable_thinking template")
    scripts = [REPO / "scripts" / name for name in
               ("run_reasoning_api.py", "run_evalplus_reasoning_api.py")]
    sources = scripts + [Path(__file__), Path(__file__).with_name("evaluation_io.py")]
    sources += sorted((REPO / "tasks").glob("aime*/*"))
    sources += sorted((REPO / "tasks" / "gpqa_diamond").glob("*"))
    recipe = {"server": server, "model": str(model), "base_url": args.base_url.rstrip("/"),
              "api_seed_mode": args.api_seed_mode, "temperature": 1.0, "top_p": 0.7,
              "max_tokens": 32768, "num_concurrent": args.num_concurrent,
              "tokenizer_files": {p.name: file_hash(p) for p in sorted(model.glob("*"))
                                  if p.is_file() and ("token" in p.name or p.name == "config.json")},
              "sources": {str(p.relative_to(REPO)): file_hash(p) for p in sources if p.is_file()},
              "packages": {name: importlib.metadata.version(name) for name in
                           ("lm_eval", "evalplus", "transformers", "datasets")}}
    root = Path(args.root).resolve() / args.method / digest(recipe)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root / "recipe.json", recipe)
        coverage_path = root / "audit-coverage.json"
        coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else []
        if not args.audit:
            for kind in {client_kind(b) for b in args.benchmarks}:
                for seed in (0, 1):
                    matches = [x for x in coverage if x["kind"] == kind and x["seed"] == seed
                               and Path(x["marker"]).exists()
                               and file_hash(x["marker"]) == x["marker_sha256"]]
                    if not matches:
                        raise RuntimeError(f"Missing real HTTP audit for {kind} client seed {seed}; "
                                           "run --audit --benchmarks aime24 evalplus_humaneval --seeds 0 1 first")
        run = None
        if args.wandb_project:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, job_type="http-audit" if args.audit else "accuracy",
                             name=f"{args.method}-{'audit' if args.audit else 'accuracy'}", mode="online",
                             config={"recipe_sha256": digest(recipe), "method": args.method,
                                     "sglang_git_sha": server["sglang_git_sha"],
                                     "api_seed_mode": args.api_seed_mode},
                             settings=wandb.Settings(disable_code=True, console="off"))
            if run.offline:
                raise RuntimeError("Requested online W&B run is offline")
        try:
            for bench in args.benchmarks:
                for seed in args.seeds:
                    budget = args.audit_max_tokens if args.audit else 32768
                    directory = root / (f"audit-{budget}" if args.audit else "full") / bench / str(seed)
                    directory.mkdir(parents=True, exist_ok=True)
                    stem = f"{args.method}_{seed}_{bench}"
                    marker = directory / "complete.json"
                    expected_count = 1 if args.audit else COUNTS[bench]
                    api_seed = 1234 if client_kind(bench) == "lm_eval" and args.api_seed_mode == "legacy" else seed
                    if marker.exists():
                        saved = json.loads(marker.read_text())
                        if collect_cell(directory, stem, bench, expected_count) != saved["scores"]:
                            raise ValueError(f"Completed cell changed: {directory}")
                        print(f"[eval] verified completed {bench}/{seed}", flush=True)
                        if args.audit:
                            record_audit(coverage, coverage_path, bench, seed, api_seed, marker)
                        continue
                    expected = {"model": str(model), "temperature": 1.0, "top_p": 0.7,
                                "max_tokens": budget, "seed": api_seed, "stop": None}
                    if client_kind(bench) == "lm_eval":
                        expected["stop"] = (["Question:"] if bench.startswith("aime") else []) + [
                            "</s>", "<|im_end|>", "<|eot_id|>"]
                    record = {"benchmark": bench, "seed": seed, "api_seed": api_seed,
                              "audit": args.audit, "method": args.method, "host": platform.node(),
                              "started_unix": time.time(), "expected_count": expected_count}
                    with AuditProxy(args.base_url, directory / "http", tokenizer, expected, args.timeout) as proxy:
                        command = [sys.executable, str(scripts[1] if client_kind(bench) == "evalplus" else scripts[0]),
                                   "--base_url", proxy.url, "--model_name", str(model),
                                   "--output_dir", str(directory), "--save_postfix", args.method,
                                   "--seed", str(seed), "--temperature", "1.0", "--top_p", "0.7",
                                   "--max_gen_toks", str(budget), "--num_concurrent", str(args.num_concurrent),
                                   "--timeout", str(args.timeout)]
                        if client_kind(bench) == "evalplus":
                            command += ["--bench", bench.removeprefix("evalplus_")]
                        else:
                            command += ["--task", bench, "--include_path", str(REPO / "tasks"),
                                        "--api-seed-mode", args.api_seed_mode]
                        if args.audit:
                            command += ["--limit", "1"]
                        record["command"] = command
                        atomic_json(directory / "running.json", record)
                        with (directory / "client.log").open("a", buffering=1) as logfile:
                            process = subprocess.Popen(command, cwd=REPO, stdout=logfile, stderr=subprocess.STDOUT)
                            record["pid"] = process.pid
                            atomic_json(directory / "running.json", record)
                            while process.poll() is None:
                                try:
                                    process.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    counters = proxy.snapshot()
                                    print(f"[eval] {bench}/{seed} {counters}", flush=True)
                                    if run:
                                        run.log({f"{bench}/seed{seed}/{k}": v for k, v in counters.items()})
                        record.update(exit_code=process.returncode, elapsed_seconds=time.time() - record["started_unix"],
                                      http=proxy.snapshot())
                    atomic_json(directory / f"attempt-{time.time_ns()}.json", record)
                    if record["exit_code"]:
                        raise RuntimeError(f"Client failed ({record['exit_code']}), see {directory / 'client.log'}")
                    if record["http"]["network_success"] + record["http"]["replayed"] != expected_count:
                        raise ValueError(f"HTTP request count differs from full scored set: {directory}")
                    scores = collect_cell(directory, stem, bench, expected_count)
                    record["scores"] = scores
                    atomic_json(marker, record)
                    if args.audit:
                        record_audit(coverage, coverage_path, bench, seed, api_seed, marker)
                    elif run:
                        run.log({f"{bench}/seed{seed}/accuracy_percent": scores["accuracy_percent"],
                                 f"{bench}/seed{seed}/n": scores["n"],
                                 f"{bench}/seed{seed}/elapsed_seconds": record["elapsed_seconds"]})
                    print(f"[eval] {'AUDIT' if args.audit else 'FULL'} {bench}/{seed}: {scores}", flush=True)
            if not args.audit:
                summarize(root, args.method, run)
        finally:
            if run:
                run.finish()


if __name__ == "__main__":
    main()
