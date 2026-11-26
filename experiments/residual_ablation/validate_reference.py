"""Independent layout/export and native-cache checks; run under torchrun TP2."""
import argparse
import importlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_reference import CoupledCodec, Coded, cache_class, encode_residual, decode_residual, atomic_json


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='sdpa', tp_plan='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    native = importlib.import_module(type(model).__module__).UniversalTransformerCache(192)
    custom = cache_class(model)(512)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explain gravity. ' * 32}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True)
    tokens = torch.tensor([ids], device='cuda')
    errors = []
    for step in range(5):
        a = model(tokens, past_key_values=native, use_cache=True, logits_to_keep=1).logits.float()
        b = model(tokens, past_key_values=custom, use_cache=True, logits_to_keep=1).logits.float()
        errors.append((a - b).abs().max().item())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        tokens = a[:, -1].argmax(-1, keepdim=True)
    assert custom.key_cache[0].shape[1] == 16 // world
    del native, custom, model
    torch.cuda.empty_cache()
    codec = CoupledCodec(args.bundle, rank, world)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    torch.manual_seed(491)
    raw = torch.randn(192, 6, 16, 128, device='cuda')
    checks = []
    for kind, name in enumerate(('codebook.pt', 'vq_v_codebook.pt')):
        book = Coded(Path(args.bundle) / name, 'cuda')
        torch.testing.assert_close(book.cb.cpu(), checkpoint['best_codebooks'][kind].half().float(), rtol=0, atol=0)
        reference = book.decode(book.assign(raw))
        compensated = reference + decode_residual(encode_residual(raw, reference))
        heads = slice(rank * 8, (rank + 1) * 8)
        for base in range(48):
            slots = [base + 48 * d for d in range(4)]
            data = raw[slots, :, heads].reshape(4, 2, 3, 8, 128).permute(0, 1, 3, 2, 4)
            for residual, ref in ((False, reference), (True, compensated)):
                expected = ref[slots, :, heads].reshape(4, 2, 3, 8, 128).permute(0, 1, 3, 2, 4)
                actual = codec.reconstruct(data, base, kind, residual)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checks.append(name)
        del book, reference, compensated
    if rank == 0:
        atomic_json(args.output, {'passed': True, 'tp': world, 'bf16_logit_max_errors': errors,
            'export_exact': checks, 'layout_exact': 'all 48 base layers, K/V, with/without residual, batch2',
            'input_ids': ids, 'training_recipe': checkpoint['recipe']})
        print('VALIDATION_OK', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
