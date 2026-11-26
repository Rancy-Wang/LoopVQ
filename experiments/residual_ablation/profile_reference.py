"""Short synthetic-history timings of the unchanged accuracy reference.

Not an accuracy run or a serving throughput comparison. Unwritten historical
KV entries are zero; shapes, kernels, graph padding and codec are real.
"""
import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_reference import CoupledCodec, GraphDecode, cache_class, select_attention_backend
from reproduction_tracking import atomic_json


def stats(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    select_attention_backend()
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', tp_plan='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    codec = CoupledCodec(args.bundle, rank, dist.get_world_size())
    cache = cache_class(model)(40960)
    token = torch.tensor([[42]], dtype=torch.long, device='cuda')
    model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    runner = GraphDecode(model, cache)
    output = {'synthetic_history': True, 'tp': dist.get_world_size(),
              'local_heads': cache.key_cache[0].shape[1], 'points': [],
              'note': 'Timing pass excludes capture; separate profiler pass has overhead.'}

    def reset(length):
        cache.advance(length)
        cache.quant_end = cache.prefix + max(0, (length-cache.prefix-cache.recent)//8)*8

    def step(timed):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if timed else None
        start = time.perf_counter()
        if timed:
            events[0].record()
        raw = runner.decode(token)
        after_forward = time.perf_counter()
        if timed:
            events[1].record()
        cache.finish_forward()
        after_codec = time.perf_counter()
        if timed:
            events[2].record()
        ordered, indices = raw[:, -1].float().sort(descending=True)
        probabilities = ordered.softmax(-1)
        ordered.masked_fill_(probabilities.cumsum(-1)-probabilities > 0.7, -float('inf'))
        pick = torch.multinomial(ordered.softmax(-1), 1)
        chosen = indices.gather(-1, pick)
        dist.broadcast(chosen, 0)
        if timed:
            events[3].record()
        value = int(chosen.item())
        after_sample = time.perf_counter()
        if timed:
            return {'wall_ms': (after_sample-start)*1000,
                    'cpu_forward_submit_ms': (after_forward-start)*1000,
                    'cpu_codec_submit_ms': (after_codec-after_forward)*1000,
                    'cpu_sample_and_wait_ms': (after_sample-after_codec)*1000,
                    'gpu_forward_ms': events[0].elapsed_time(events[1]),
                    'gpu_codec_interval_ms': events[1].elapsed_time(events[2]),
                    'gpu_sample_interval_ms': events[2].elapsed_time(events[3]),
                    'chosen_id': value}

    for arm in ('bf16', 'loopvq', 'residual'):
        cache.codec = None if arm == 'bf16' else codec
        cache.residual = arm == 'residual'
        for length in (2048, 8192, 16384):
            reset(length)
            for _ in range(8):
                step(False)
            torch.cuda.synchronize()
            reset(length)
            rows = [step(True) for _ in range(16)]
            item = {'arm': arm, 'history': length,
                    'bucket': next(n for n in runner.graphs if n > length),
                    'steps': rows,
                    'summary': {k: stats([r[k] for r in rows]) for k in rows[0] if k != 'chosen_id'}}
            if length == 8192 and arm in ('bf16', 'residual'):
                reset(length)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(8):
                        step(False)
                    torch.cuda.synchronize()
                kernels = defaultdict(lambda: [0, 0.0])
                for event in prof.events():
                    if event.device_type == torch.autograd.DeviceType.CUDA:
                        kernels[event.name][0] += 1
                        kernels[event.name][1] += event.device_time_total / 1000
                item['profile_gpu_kernel_sum_ms'] = sum(v[1] for v in kernels.values())
                item['profile_gpu_kernels'] = [{'name': k, 'count': v[0], 'total_ms': v[1]}
                    for k, v in sorted(kernels.items(), key=lambda kv: -kv[1][1])]
            output['points'].append(item)
            if rank == 0:
                atomic_json(args.output, output)
                print(json.dumps({k: v for k, v in item.items() if k not in ('steps', 'profile_gpu_kernels')}), flush=True)
    runner.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
