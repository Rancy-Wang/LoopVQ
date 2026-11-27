"""CPU HTTP integration checks; these do not satisfy the real model audit gate."""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evaluation_io import AuditProxy


class Tokenizer:
    eos_token, eos_token_id = "</s>", 2

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\n<think>\n"

    def encode(self, text, **kwargs):
        return list(text.encode())


class HTTPAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.requests = []
        self.failures = 0
        self.bad_count = False
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append(payload)
                if test.failures:
                    test.failures -= 1
                    status, body = 503, {"error": "retryable failure"}
                else:
                    tok = Tokenizer()
                    count = len(tok.encode(tok.apply_chat_template(payload["messages"])))
                    status, body = 200, {"choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "reasoning\n</think>\n42"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": count + int(test.bad_count),
                                  "completion_tokens": 7, "total_tokens": count + 7}}
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.upstream = f"http://127.0.0.1:{self.httpd.server_port}"
        self.expected = {"model": "/model", "seed": 0, "temperature": 1.0,
                         "top_p": 0.7, "max_tokens": 64, "stop": ["</s>"]}
        self.payload = {**self.expected, "messages": [{"role": "user", "content": "Question: 6 * 7?"}],
                        "chat_template_kwargs": {"enable_thinking": True}}

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def proxy(self):
        return AuditProxy(self.upstream, self.directory, Tokenizer(), self.expected, timeout=5)

    def post(self, url, payload=None):
        request = Request(url, json.dumps(payload or self.payload).encode(), {"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def attempts(self):
        return [json.loads(path.read_text()) for path in (self.directory / "attempts").glob("*.json")]

    def test_wire_payload_and_resume_after_process_restart(self):
        with self.proxy() as proxy:
            first = self.post(proxy.url)
            self.assertEqual(first[0], 200)
            self.assertEqual(proxy.snapshot()["network_success"], 1)
        with self.proxy() as proxy:
            second = self.post(proxy.url)
            self.assertEqual(second, first)
            self.assertEqual(proxy.snapshot()["replayed"], 1)
        self.assertEqual(self.requests, [self.payload])
        records = self.attempts()
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["payload"], self.payload)
            self.assertEqual(len(record["input_token_ids"]), record["usage"]["prompt_tokens"])
            self.assertEqual(record["finish_reason"], "stop")
            self.assertEqual(record["input_token_ids_source"], "checkpoint_template_reconstruction")

    def test_greedy_rejects_sampled_payload(self):
        self.expected.update(temperature=0.0, top_p=1.0)
        self.payload.update(temperature=0.0, top_p=1.0)
        with self.proxy() as proxy:
            self.assertEqual(self.post(proxy.url, {**self.payload, "temperature": 1.0})[0], 502)
            self.assertEqual(len(self.requests), 0)
            self.assertEqual(self.post(proxy.url)[0], 200)
        self.assertEqual(self.requests[0]["temperature"], 0.0)
        self.assertEqual(len(list((self.directory / "responses").glob("*.json"))), 1)

    def test_failed_attempt_survives_successful_retry(self):
        self.failures = 1
        with self.proxy() as proxy:
            self.assertEqual(self.post(proxy.url)[0], 503)
            self.assertEqual(self.post(proxy.url)[0], 200)
            self.assertEqual(proxy.snapshot()["failed"], 1)
            self.assertEqual(proxy.snapshot()["network_success"], 1)
        records = self.attempts()
        self.assertEqual(sorted(x["ok"] for x in records), [False, True])
        failure = next(x for x in records if not x["ok"])
        self.assertIn("retryable failure", failure["upstream_error_body"])
        self.assertEqual(failure["payload"], self.payload)

    def test_seed_and_token_count_mismatches_are_not_cached(self):
        with self.proxy() as proxy:
            self.assertEqual(self.post(proxy.url, {**self.payload, "seed": 1})[0], 502)
            self.assertEqual(len(self.requests), 0)
            self.bad_count = True
            self.assertEqual(self.post(proxy.url)[0], 502)
        self.assertEqual(len(list((self.directory / "responses").glob("*.json"))), 0)
        records = self.attempts()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(not x["ok"] for x in records))
        self.assertTrue(any(x.get("response", {}).get("usage") for x in records if x.get("response")))

    def test_concurrent_duplicate_only_generates_once(self):
        with self.proxy() as proxy, ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(lambda _: self.post(proxy.url), range(2)))
            self.assertEqual([x[0] for x in responses], [200, 200])
            self.assertEqual(proxy.snapshot()["network_success"], 1)
            self.assertEqual(proxy.snapshot()["replayed"], 1)
        self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
