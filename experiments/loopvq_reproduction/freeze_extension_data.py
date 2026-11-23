#!/usr/bin/env python3
"""Freeze extension benchmark data without generating prompts or model outputs.

Run ``online`` followed by ``offline`` with the same --root and --repo. Raw
datasets stay under the task cache; results contain hashes, counts and paths.
Dependencies are inspected, never installed. No W&B client is imported.
"""

import argparse
import collections
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
import zipfile


SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", [
        "main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"]),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                ["test.jsonl"]),
    "ruler": ("simonjegou/ruler", "24adceac8a0e6532936e8d721cd9e9084d2e4686",
              ["16384/test-00000-of-00001.parquet"]),
    "longbench": ("zai-org/LongBench", "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
                  ["data.zip"]),
}
LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
SOURCE_FILES = [
    "tasks/gsm8k_cot_3shot/gsm8k_cot_3shot.yaml",
    "tasks/niah/build_niah_rows.py", "scripts/run_math500_api.py",
    "experiments/loopvq_longbench/build_rows.py",
    "experiments/loopvq_longbench/lb/dataset2prompt.json",
    "experiments/loopvq_longbench/lb/dataset2maxlen.json",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_info(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def canonical(row):
    return (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def rows_digest(rows):
    digest, count = hashlib.sha256(), 0
    for row in rows:
        digest.update(canonical(row))
        count += 1
    return {"rows": count, "rows_sha256": digest.hexdigest()}


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def metric_import_checks():
    checks = {}
    for name in ("jieba", "fuzzywuzzy", "rouge"):
        try:
            importlib.import_module(name)
            checks[name] = {"importable": True}
        except Exception as error:
            checks[name] = {"importable": False, "error": repr(error)}
    return checks


def freeze(root, repo, mode):
    # These imports must follow main's cache/offline environment configuration.
    import pyarrow.parquet as parquet
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token, hf_hub_download

    online = mode == "online"
    manifest_path = root / "results/extensions-data-manifest.json"
    optional_metrics = metric_import_checks()
    observed = {
        "packages": {name: metadata.version(name) for name in
                     ("datasets", "huggingface-hub", "pyarrow", "PyYAML")},
        "sources": {},
        "source_files": {name: file_info(repo / name) for name in SOURCE_FILES},
    }
    token = get_token() if online else False
    api = HfApi(token=token) if online else None
    paths = {}
    for key, (hub_repo, revision, filenames) in SOURCES.items():
        if online:
            require(api.dataset_info(hub_repo, revision=revision).sha == revision,
                    f"Pinned revision unavailable: {hub_repo}@{revision}")
            require(api.dataset_info(hub_repo).sha == revision,
                    f"{hub_repo}: main moved; do not silently change the pinned revision")
        files = {}
        for filename in filenames:
            path = Path(hf_hub_download(
                hub_repo, filename, repo_type="dataset", revision=revision,
                token=token, local_files_only=not online))
            paths[(key, filename)] = path
            files[filename] = file_info(path)
        observed["sources"][key] = {"repo_id": hub_repo, "revision": revision, "files": files}

    def parquet_rows(path):
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()

    train = list(parquet_rows(paths[("gsm8k", "main/train-00000-of-00001.parquet")]))
    test = list(parquet_rows(paths[("gsm8k", "main/test-00000-of-00001.parquet")]))
    require(len(train) == 7473 and len(test) == 1319, "Unexpected GSM8K split sizes")
    config = yaml.safe_load((repo / SOURCE_FILES[0]).read_text(encoding="utf-8"))
    require(config["num_fewshot"] == 3 and config["fewshot_config"]["sampler"] == "first_n",
            "GSM8K no longer uses the expected three fixed examples")
    demos = config["fewshot_config"]["samples"][:3]
    require(len(demos) == 3, "Missing GSM8K fixed examples")
    normalize = lambda text: " ".join(text.split())
    observed["gsm8k"] = {
        "train": rows_digest(train), "test": rows_digest(test),
        "fewshot": {
            "count": 3, "source": "inline YAML first_n; not sampled from train",
            "sha256": hashlib.sha256(canonical(demos)).hexdigest(),
            "matching_train_question_indices": [
                [index for index, row in enumerate(train)
                 if normalize(row["question"]) == normalize(demo["question"])]
                for demo in demos],
        },
    }
    math_rows = list(json_rows(paths[("math500", "test.jsonl")]))
    require(len(math_rows) == 500, "Unexpected MATH500 split size")
    observed["math500"] = rows_digest(math_rows)

    # Prime and reread the existing entrypoints' exact, unversioned dataset calls.
    # Pinned downloaded files remain authoritative: cache contents must match them.
    gsm = load_dataset("gsm8k", "main")
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    require(rows_digest(gsm["train"]) == observed["gsm8k"]["train"], "GSM8K train loader mismatch")
    require(rows_digest(gsm["test"]) == observed["gsm8k"]["test"], "GSM8K test loader mismatch")
    require(rows_digest(math) == observed["math500"], "MATH500 loader mismatch")
    observed["entrypoint_loaders"] = {
        "gsm8k_main_train_test": "matched", "math500_test": "matched"}

    ruler_path = paths[("ruler", "16384/test-00000-of-00001.parquet")]
    selected, indices, counts = [], [], collections.Counter()
    digest, count = hashlib.sha256(), 0
    for index, row in enumerate(parquet_rows(ruler_path)):
        digest.update(canonical(row))
        count += 1
        counts[row["task"]] += 1
        if row["task"] == "niah_single_2" and len(selected) < 40:
            selected.append(row)
            indices.append(index)
    require(count == 6500 and len(selected) == 40, "Unexpected RULER source rows")
    # Identical transformation and ordering to tasks/niah/build_niah_rows.py.
    pieces = []
    for row in selected:
        body = row["context"][row["context"].index("\n"):]
        pieces.append(re.sub(r" ?One of the special magic numbers for [^\s]+ is: \d+\.",
                             "", body, count=1))
    corpus = " ".join(pieces).encode("utf-8")
    corpus_path = root / "cache/extensions/niah" / SOURCES["ruler"][1] / "niah_corpus_essays.txt"
    if online:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        if not corpus_path.exists():
            corpus_path.write_bytes(corpus)
    require(corpus_path.read_bytes() == corpus, "Existing NIAH corpus differs")
    observed["niah"] = {
        "source_rows": count, "source_rows_sha256": digest.hexdigest(),
        "task_counts": dict(counts), "selected_count": 40, "selected_source_indices": indices,
        "selected_rows_sha256": rows_digest(selected)["rows_sha256"],
        "corpus": file_info(corpus_path),
        "generation": "not run; pass this corpus path explicitly to the builder",
    }

    zip_path = paths[("longbench", "data.zip")]
    destination = root / "cache/extensions/longbench" / SOURCES["longbench"][1] / "data"
    longbench = {}
    with zipfile.ZipFile(zip_path) as archive:
        for task in LONGBENCH_TASKS:
            names = [name for name in archive.namelist() if name.split("/")[-1] == task + ".jsonl"]
            require(len(names) == 1, f"Expected one LongBench member for {task}: {names}")
            output = destination / (task + ".jsonl")
            if online:
                destination.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".jsonl.tmp")
                with archive.open(names[0]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(output)
            member_digest = hashlib.sha256()
            with archive.open(names[0]) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    member_digest.update(chunk)
            info = file_info(output)
            require(info["sha256"] == member_digest.hexdigest(), f"LongBench member mismatch: {task}")
            stats = rows_digest(json_rows(output))
            require(stats["rows"] > 0, f"Empty LongBench task: {task}")
            longbench[task] = {"zip_member": names[0], **info, **stats}
    require(sum(item["rows"] for item in longbench.values()) == 3750, "LongBench must contain 3750 rows")
    observed["longbench"] = {
        "tasks": longbench, "total_rows": 3750, "data_dir": str(destination),
        "generation": "not run; no truncation or prompt changes",
    }

    if online:
        manifest = {"status": "online_complete_offline_pending", "schema": 1,
                    "created_utc": utc_now(), "observed": observed,
                    "optional_metric_imports_online": optional_metrics}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest["observed"] == observed, "Offline files/rows/protocol/package mismatch")
        manifest.update(status="verified_offline", offline_verified_utc=utc_now(),
                        optional_metric_imports_offline=optional_metrics)
    write_json(manifest_path, manifest)
    print(json.dumps({"phase": mode, "status": "pass", "gsm8k_train": 7473, "gsm8k_test": 1319,
                      "math500": 500, "ruler": 6500, "niah_corpus_source_rows": 40,
                      "longbench_tasks": 16, "longbench_rows": 3750,
                      "optional_metric_imports": optional_metrics, "manifest": str(manifest_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("online", "offline"))
    parser.add_argument("--root", type=Path, default=Path(
        "/homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction"))
    parser.add_argument("--repo", type=Path, default=Path("/homes/ruoxi07/repo/LoopVQ"))
    args = parser.parse_args()
    root, repo = args.root.expanduser().resolve(), args.repo.expanduser().resolve()
    cache = root / "cache/huggingface"
    os.environ.update({
        "HF_HOME": str(cache), "HF_HUB_CACHE": str(cache / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "hub"), "HF_TOKEN_PATH": str(cache / "token"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "HF_HUB_OFFLINE": "1" if args.mode == "offline" else "0",
        "HF_DATASETS_OFFLINE": "1" if args.mode == "offline" else "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    manifest_path = root / "results/extensions-data-manifest.json"
    try:
        freeze(root, repo, args.mode)
    except Exception as error:
        failure = {"phase": args.mode, "status": "failed", "error_type": type(error).__name__,
                   "error": str(error), "utc": utc_now()}
        write_json(manifest_path.with_name("extensions-data-manifest.failure.json"), failure)
        if not manifest_path.exists():
            write_json(manifest_path, failure)
        raise


if __name__ == "__main__":
    main()
