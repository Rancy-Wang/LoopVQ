"""Opt-in experiment records; does not change calibration or training arithmetic."""
import atexit
import hashlib
import json
import os
import platform
import time
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def rng_state():
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def add_tracking_args(parser):
    parser.add_argument("--metrics-jsonl", help="append local metrics (no raw examples)")
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-group", default="2026-11-23-loopvq-reproduction")
    parser.add_argument("--wandb-name")


class Tracking:
    def __init__(self, args, job_type, config):
        self.started = time.monotonic()
        self.run = None
        self.closed = False
        self.path = Path(args.metrics_jsonl) if args.metrics_jsonl else None
        self.identity = {"job_type": job_type, "host": platform.node(), "pid": os.getpid()}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if args.wandb_mode != "disabled":
            import wandb
            self.run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                                  group=args.wandb_group, name=args.wandb_name,
                                  job_type=job_type, mode=args.wandb_mode, config=config)
            if args.wandb_mode == "online" and self.run.offline:
                raise RuntimeError("An online W&B run was required but the SDK is offline")
            self.identity.update(wandb_run_id=self.run.id, wandb_url=self.run.url)
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("validation/*", step_metric="global_step")
        self.log({"event": "start", "config": config})
        # Unexpected exits are failures; successful paths explicitly finish(0).
        atexit.register(self.finish, 1)

    def log(self, values):
        record = {"time_unix": time.time(), "elapsed_seconds": time.monotonic() - self.started,
                  **self.identity, **values}
        if self.path:
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
        if self.run:
            try:
                self.run.log({k: v for k, v in record.items() if k != "config"})
            except Exception as exc:
                # Local metrics remain authoritative; an upload outage must not discard training.
                print(f"[tracking] W&B log delayed/failed ({type(exc).__name__}); local record retained",
                      flush=True)

    def finish(self, exit_code=0):
        if self.closed:
            return
        self.closed = True
        self.log({"event": "finish", "exit_code": exit_code})
        if self.run:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                print(f"[tracking] W&B finish failed ({type(exc).__name__}); sync local run later", flush=True)


def checkpoint_state(qk, qv, optimizer, recipe, epoch, step, best, best_state, trace,
                     elapsed_seconds, tracking_identity):
    """Snapshot for immediate atomic serialization at a completed epoch boundary."""
    return {"format_version": 1, "recipe": recipe, "recipe_sha256": json_digest(recipe),
            "completed_epochs": epoch, "global_step": step, "best_kl": best,
            "codebooks": [qk.detach().cpu().clone(), qv.detach().cpu().clone()],
            "best_codebooks": [x.detach().cpu().clone() for x in best_state],
            "optimizer": optimizer.state_dict(), "rng": rng_state(), "trace": trace,
            "elapsed_seconds": elapsed_seconds, "tracking": tracking_identity}


def restore_checkpoint(path, qk, qv, optimizer, recipe):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state.get("recipe_sha256") != json_digest(recipe):
        raise ValueError("Checkpoint recipe/source/data mismatch; refusing to resume")
    if state.get("recipe") != recipe:
        raise ValueError("Checkpoint manifest differs from the current recipe")
    for label in ("codebooks", "best_codebooks"):
        if len(state[label]) != 2:
            raise ValueError("Checkpoint must contain both K and V codebooks")
        for parameter, value in zip((qk, qv), state[label]):
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid checkpoint codebook")
    with torch.no_grad():
        for parameter, value in zip((qk, qv), state["codebooks"]):
            parameter.copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state
