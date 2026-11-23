"""Durable HTTP audit/replay for the unchanged official benchmark clients.

Each successful request is committed before its response is returned to the client.
Restarting a cell replays those responses and redoes scoring, never silently dropping
failed requests. Files contain prompts/code and stay local; only counters are exported.
"""
import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w") as out:
        json.dump(value, out, ensure_ascii=False)
        out.flush()
        import os
        os.fsync(out.fileno())
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class AuditProxy:
    def __init__(self, upstream, directory, tokenizer, expected, timeout=172800):
        self.upstream = upstream.rstrip("/") + "/v1/chat/completions"
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.expected = expected
        self.timeout = timeout
        self.lock = threading.Lock()
        self.key_locks = {}
        self.stats = {"network_success": 0, "replayed": 0, "failed": 0,
                      "prompt_tokens": 0, "completion_tokens": 0}
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    status, body = proxy.process(payload)
                except Exception as exc:
                    status, body = 502, {"error": {"message": str(exc)}}
                raw = json.dumps(body, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}/v1/chat/completions"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    def snapshot(self):
        with self.lock:
            return dict(self.stats)

    def process(self, payload):
        key = digest(payload)
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        # A repeated request after a client timeout waits for the original attempt,
        # preventing an extra generation from replacing a successfully saved answer.
        with key_lock:
            return self._process(key, payload)

    def _process(self, key, payload):
        started = time.time()
        attempt = {"request_sha256": key, "payload": payload, "started_unix": started,
                   "upstream": self.upstream}
        status, body = 502, None
        try:
            for field, expected in self.expected.items():
                if payload.get(field) != expected:
                    raise ValueError(f"HTTP payload mismatch for {field}: "
                                     f"{payload.get(field)!r} != {expected!r}")
            if payload.get("stream"):
                raise ValueError("Benchmark audit only supports non-streaming requests")
            messages = payload["messages"]
            kwargs = payload.get("chat_template_kwargs", {})
            if not isinstance(kwargs, dict) or kwargs.get("enable_thinking") is not True:
                raise ValueError("Thinking evaluation requires enable_thinking=True")
            # These IDs are reconstructed from the pinned checkpoint's own template.
            # The server's usage validates the length; do not claim server-returned IDs.
            with self.lock:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs)
                ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            attempt.update(rendered_prompt=prompt, input_token_ids=ids,
                           input_token_ids_source="checkpoint_template_reconstruction",
                           tokenizer_eos_token_ids=self.tokenizer.eos_token_id,
                           tokenizer_eos_token=self.tokenizer.eos_token)
            if not prompt.rstrip().endswith("<think>"):
                raise ValueError("Rendered prompt does not end with the expected Thinking prefix")
            cache = self.directory / "responses" / f"{key}.json"
            if cache.exists():
                saved = json.loads(cache.read_text())
                if (saved["payload"] != payload or saved["input_token_ids"] != ids
                        or saved.get("upstream") != self.upstream):
                    raise ValueError("Cached request/tokenizer/upstream identity changed")
                body = saved["response"]
                status = 200
                attempt["replayed"] = True
            else:
                raw = json.dumps(payload).encode()
                request = Request(self.upstream, raw, {"Content-Type": "application/json"})
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        status = response.status
                        body = json.load(response)
                except HTTPError as exc:
                    status = exc.code
                    raw = exc.read().decode(errors="replace")
                    attempt["upstream_error_body"] = raw
                    raise RuntimeError(f"Upstream HTTP {status}") from exc
                choices = body.get("choices", [])
                if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                    raise ValueError("Expected exactly one complete chat response")
                if choices[0].get("finish_reason") not in ("stop", "length"):
                    raise ValueError(f"Unexpected finish_reason: {choices[0].get('finish_reason')}")
                usage = body.get("usage", {})
                if usage.get("prompt_tokens") != len(ids):
                    raise ValueError(f"Server prompt_tokens={usage.get('prompt_tokens')} "
                                     f"but reconstructed input has {len(ids)} tokens")
                if not isinstance(usage.get("completion_tokens"), int):
                    raise ValueError("Server did not report completion_tokens")
                attempt.update(response=body, http_status=status, prompt_token_count_verified=True)
                atomic_json(cache, attempt)
            attempt.update(response=body, http_status=status, ok=True,
                           finish_reason=body["choices"][0]["finish_reason"], usage=body["usage"])
            with self.lock:
                self.stats["replayed" if attempt.get("replayed") else "network_success"] += 1
                for field in ("prompt_tokens", "completion_tokens"):
                    self.stats[field] += body["usage"][field]
        except Exception as exc:
            attempt.update(ok=False, http_status=status, error=f"{type(exc).__name__}: {exc}",
                           response=body)
            with self.lock:
                self.stats["failed"] += 1
            status = status if status >= 400 else 502
            body = {"error": {"message": attempt["error"]}}
        finally:
            attempt["elapsed_seconds"] = time.time() - started
            atomic_json(self.directory / "attempts" / f"{key}-{uuid.uuid4().hex}.json", attempt)
        return status, body
