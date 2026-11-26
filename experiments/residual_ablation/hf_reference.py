"""Independent TP2 accuracy reference with pagewise coupled KV reconstruction.

Stores reconstructed BF16 KV to isolate quality. No serving performance claim.
"""
import argparse
import importlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'calibration'))
from finetune_codebook import Coded
from ouro_cq import RopeInverter, rotate_half
from residual_codec import encode_residual, decode_residual
from reproduction_tracking import atomic_json


class CoupledCodec:
    def __init__(self, bundle, rank, world):
        self.books = []
        for filename in ('codebook.pt', 'vq_v_codebook.pt'):
            book = Coded(Path(bundle) / filename, 'cuda')
            if not book.ident or book.norm or book.ls is not None or book.cw is not None:
                raise ValueError('Expected pure identity LoopVQ')
            heads = book.H // world
            book.cb = torch.nn.Parameter(book.cb[:, rank * heads:(rank + 1) * heads].contiguous(),
                                         requires_grad=False)
            book.H = heads
            self.books.append(book)

    @torch.no_grad()
    def reconstruct(self, data, base, kind, residual):
        # [four loops, batch, heads, tokens, channels] -> joint 16-D code vectors.
        loops, batch, heads, tokens, channels = data.shape
        assert loops == 4 and channels == 128
        vectors = data.permute(1, 3, 2, 4, 0).reshape(batch * tokens, heads, 32, 16)
        target = vectors.permute(1, 2, 0, 3).reshape(heads * 32, batch * tokens, 16).float()
        table = self.books[kind].cb[base].reshape(heads * 32, 256, 16)
        codes = torch.cdist(target, table).argmin(-1)
        decoded = table.to(torch.bfloat16)[torch.arange(heads * 32, device=data.device)[:, None], codes].float()
        decoded = decoded.reshape(heads, 32, batch, tokens, 4, 4).permute(5, 2, 0, 3, 1, 4)
        decoded = decoded.reshape_as(data)
        if residual:
            decoded = decoded + decode_residual(encode_residual(data, decoded))
        return decoded


def cache_class(model):
    native = importlib.import_module(type(model).__module__).UniversalTransformerCache

    class PageCache(native):
        def __init__(self, capacity, codec=None, residual=False, prefix=8, recent=64):
            super().__init__(192)
            self.capacity, self.codec, self.residual = capacity, codec, residual
            self.prefix, self.recent = prefix, recent
            self.storage = {}
            self.lengths = {}
            self.quant_end = prefix
            self.graph_size = 0
            self.rope = RopeInverter(model.model.rotary_emb)

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if self.graph_size:
                keys, values = self.storage[layer_idx]
                positions = cache_kwargs['cache_position']
                keys.index_copy_(2, positions, key_states)
                values.index_copy_(2, positions, value_states)
                return keys[:, :, :self.graph_size], values[:, :, :self.graph_size]
            if layer_idx not in self.storage:
                shape = (*key_states.shape[:2], self.capacity, key_states.shape[-1])
                self.storage[layer_idx] = (key_states.new_zeros(shape), value_states.new_zeros(shape))
                self.lengths[layer_idx] = 0
            start = self.lengths[layer_idx]
            end = start + key_states.shape[2]
            if end > self.capacity:
                raise ValueError('Request exceeded declared cache capacity')
            keys, values = self.storage[layer_idx]
            keys[:, :, start:end].copy_(key_states)
            values[:, :, start:end].copy_(value_states)
            self.lengths[layer_idx] = end
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
            self.key_cache[layer_idx], self.value_cache[layer_idx] = keys[:, :, :end], values[:, :, :end]
            self._seen_tokens = end
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def reset(self):
            self.quant_end = self.prefix
            self.advance(0)

        def advance(self, length):
            for slot, (keys, values) in self.storage.items():
                self.key_cache[slot], self.value_cache[slot] = keys[:, :, :length], values[:, :, :length]
                self.lengths[slot] = length
            self._seen_tokens = length

        @torch.no_grad()
        def finish_forward(self):
            if self.codec is None:
                return
            length = self.get_seq_length()
            assert len(self.lengths) == 192 and set(self.lengths.values()) == {length}
            end = self.prefix + max(0, (length - self.prefix - self.recent) // 8) * 8
            for start in range(self.quant_end, end, 128):
                stop = min(end, start + 128)
                positions = torch.arange(start, stop, device='cuda')
                cos, sin = self.rope.cos_sin(positions, 'cuda')
                for base in range(48):
                    slots = [base + depth * 48 for depth in range(4)]
                    post = torch.stack([self.key_cache[s][:, :, start:stop] for s in slots]).float()
                    raw = self.rope.unrope(post, (cos, sin))
                    vals = torch.stack([self.value_cache[s][:, :, start:stop] for s in slots]).float()
                    kr = self.codec.reconstruct(raw, base, 0, self.residual)
                    vr = self.codec.reconstruct(vals, base, 1, self.residual)
                    kr = kr * cos + rotate_half(kr) * sin
                    for depth, slot in enumerate(slots):
                        self.key_cache[slot][:, :, start:stop].copy_(kr[depth])
                        self.value_cache[slot][:, :, start:stop].copy_(vr[depth])
            self.quant_end = end
    return PageCache


class GraphDecode:
    """Reusable full-depth decode graphs; quantization still runs outside capture."""
    def __init__(self, model, cache, verify=False):
        self.model, self.cache, self.verify = model, cache, verify
        self.tokens = torch.zeros((1, 1), dtype=torch.long, device='cuda')
        self.position = torch.zeros(1, dtype=torch.long, device='cuda')
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.max_error = 0.0

    def decode(self, tokens):
        length = self.cache.get_seq_length()
        bucket = next(n for n in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960) if n > length)
        assert bucket <= self.cache.capacity
        self.tokens.copy_(tokens)
        self.position.fill_(length)
        self.cache.graph_size = bucket
        if bucket not in self.graphs:
            indices = torch.arange(bucket, device='cuda')

            def forward():
                mask = torch.where(indices <= self.position, 0.0, torch.finfo(torch.bfloat16).min)
                mask = mask.to(torch.bfloat16).view(1, 1, 1, bucket)
                return self.model(self.tokens, past_key_values=self.cache, use_cache=True,
                    cache_position=self.position, position_ids=self.position.unsqueeze(0),
                    attention_mask={'full_attention': mask}, logits_to_keep=1).logits

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                result = forward()
            self.graphs[bucket] = (graph, result, forward)
        graph, result, forward = self.graphs[bucket]
        graph.replay()
        if self.verify:
            observed = result.clone()
            expected = forward()
            error = (observed.float() - expected.float()).abs().max().item()
            self.max_error = max(self.max_error, error)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.cache.graph_size = 0
        self.cache.advance(length + 1)
        return result


@torch.inference_mode()
def generate(model, tokenizer, cache_type, codec, residual, payload, runner=None):
    prompt = tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=True)
    maximum = int(payload.get('max_tokens', 32768))
    if len(prompt) + maximum > 40960:
        raise ValueError('Prompt plus output exceeds 40960')
    seed = int(payload.get('seed', 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cache = runner.cache if runner else cache_type(len(prompt) + maximum, codec, residual)
    if runner:
        cache.reset()
    tokens = torch.tensor([prompt], device='cuda')
    emitted = []
    temperature = float(payload.get('temperature', 1))
    top_p = float(payload.get('top_p', 0.7))
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    stops = payload.get('stop') or []
    if isinstance(stops, str):
        stops = [stops]
    finish = 'length'
    started = time.time()
    for _ in range(maximum):
        if runner and emitted:
            raw_logits = runner.decode(tokens)
        else:
            raw_logits = model(tokens, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        cache.finish_forward()
        logits = raw_logits[:, -1].float()
        if temperature == 0:
            chosen = logits.argmax(-1, keepdim=True)
        else:
            ordered, indices = (logits / temperature).sort(descending=True)
            probabilities = ordered.softmax(-1)
            remove = probabilities.cumsum(-1) - probabilities > top_p
            ordered.masked_fill_(remove, -float('inf'))
            pick = torch.multinomial(ordered.softmax(-1), 1)
            chosen = indices.gather(-1, pick)
        if dist.is_initialized():
            dist.broadcast(chosen, 0)
        value = int(chosen.item())
        emitted.append(value)
        tokens = chosen
        if value in eos:
            finish = 'stop'
            break
        if stops and any(stop in tokenizer.decode(emitted, skip_special_tokens=True) for stop in stops):
            finish = 'stop'
            break
    text = tokenizer.decode(emitted, skip_special_tokens=True)
    for stop in stops:
        text = text.split(stop, 1)[0]
    return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion',
            'created': int(time.time()), 'model': 'Ouro-2.6B-Thinking',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                         'finish_reason': finish}],
            'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(emitted),
                      'total_tokens': len(prompt) + len(emitted)},
            'audit': {'input_ids': prompt, 'output_ids': emitted, 'seconds': time.time() - started,
                      'tp': dist.get_world_size() if dist.is_initialized() else 1,
                      'local_heads': cache.key_cache[0].shape[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bundle')
    ap.add_argument('--residual', action='store_true')
    ap.add_argument('--port', type=int, default=32801)
    ap.add_argument('--output', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cuda-graph', action='store_true')
    ap.add_argument('--verify-graph', action='store_true')
    args = ap.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(rank)
    options = {'tp_plan': 'auto'} if world > 1 else {'device_map': 'cuda'}
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', **options).eval()
    assert model.config.total_ut_steps == 4
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codec = CoupledCodec(args.bundle, rank, world) if args.bundle else None
    cache_type = cache_class(model)
    runner = GraphDecode(model, cache_type(40960, codec, args.residual), args.verify_graph) if args.cuda_graph else None
    if args.smoke:
        payload = {'messages': [{'role': 'user', 'content': 'Explain why the sky is blue. ' * 12}],
                   'temperature': 0, 'max_tokens': 16}
        result = generate(model, tokenizer, cache_type, codec, args.residual, payload, runner)
        if args.verify_graph:
            eager = generate(model, tokenizer, cache_type, codec, args.residual, payload)
            assert result['audit']['output_ids'] == eager['audit']['output_ids']
            repeated = generate(model, tokenizer, cache_type, codec, args.residual, payload, runner)
            assert result['audit']['output_ids'] == repeated['audit']['output_ids']
            boundary = dict(payload)
            content = 'Explain why the sky is blue.'
            while len(tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=True)) < 250:
                content += ' blue'
            boundary['messages'] = [{'role': 'user', 'content': content}]
            padded = generate(model, tokenizer, cache_type, codec, args.residual, boundary, runner)
            direct = generate(model, tokenizer, cache_type, codec, args.residual, boundary)
            assert padded['audit']['output_ids'] == direct['audit']['output_ids']
            assert {256, 512}.issubset(runner.graphs)
            result['audit']['graph_eager_max_error'] = runner.max_error
            result['audit']['repeat_request_exact'] = True
            result['audit']['boundary_greedy_exact'] = True
        assert result['audit']['local_heads'] == 16 // world
        if rank == 0:
            atomic_json(args.output, result)
            print('SMOKE_OK', result['audit']['tp'], result['audit']['local_heads'], flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    requests = queue.Queue()
    if rank == 0:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path in ('/health', '/v1/models') else 404)
                self.end_headers()
                self.wfile.write(b'{"status":"ready"}')

            def do_POST(self):
                try:
                    payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if self.path != '/v1/chat/completions' or payload.get('stream'):
                        raise ValueError('Only non-streaming chat completions are supported')
                    reply = queue.Queue()
                    requests.put((payload, reply))
                    result = reply.get()
                    body = json.dumps(result).encode()
                    self.send_response(500 if 'error' in result else 200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        atomic_json(args.output, {'status': 'ready', 'tp': world, 'port': args.port,
                                 'model': args.model, 'bundle': args.bundle, 'residual': args.residual})
    while True:
        payload, reply = requests.get() if rank == 0 else (None, None)
        shared = [payload]
        if world > 1:
            dist.broadcast_object_list(shared, src=0)
        try:
            result = generate(model, tokenizer, cache_type, codec, args.residual, shared[0], runner)
        except Exception as exc:
            # Preserve the traceback and terminate all ranks on a model failure.
            if reply is not None:
                reply.put({'error': str(exc)})
            raise
        if rank == 0:
            reply.put(result)


if __name__ == '__main__':
    main()
