"""Paired, single-load A/B for the exact local Inkling checkpoint."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument('--engine', choices=['k4', 'k2'], required=True)
parser.add_argument('--prompts', default='p_short,p_2k')
parser.add_argument('--tokens', type=int, default=128)
parser.add_argument('--pairs', type=int, default=4)
parser.add_argument('--candidate', choices=['conv','mask','stack'], default='conv')
parser.add_argument('--reuse-prefix', action='store_true')
parser.add_argument('--out', required=True)
parser.add_argument('--model', default='~/.cache/huggingface/hub/models--mlx-community--Inkling-Small-mlx-2bit/snapshots/f8a0af554e168077bdaf48e1fb6682fb395c807c')
args = parser.parse_args()

import mlx.core as mx
from mlx.utils import tree_flatten

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
log = out.open('w')

def emit(row):
    row['time'] = time.time()
    text = json.dumps(row)
    print(text, flush=True)
    log.write(text + '\n')
    log.flush()

if args.engine == 'k4':
    import mlx_vlm
    from mlx_vlm import load
    from mlx_vlm.generate.ar import generate_step, generation_stream
    from mlx_vlm.generate.common import wired_limit
    from mlx_vlm.prompt_utils import apply_chat_template
    import mlx_vlm.models.inkling.language as language
    optimized = language._sconv_kv
    optimized_mask = language._mask_kernel
    optimized_mask_v2 = language._mask_v2_kernel
    optimized_convnorm = getattr(language, '_sconv_kv_norm', None)
    def reference(kc, vc, k, v, cache, mask=None):
        return kc(k, cache=cache, mask=mask), vc(v, cache=cache, mask=mask)
    def reference_norm(kc,vc,norm,k,v,cache,mask=None):
        k,v=reference(kc,vc,k,v,cache,mask)
        return norm(k.reshape(*k.shape[:-1],-1,norm.weight.size)).reshape(k.shape),v
    def original_dispatch(kernel):
        def wrapper(**kwargs):
            B,H,L,S=kwargs['output_shapes'][0]
            kwargs['grid']=(language._rup(S,8),language._rup(L,8),B*H)
            kwargs['threadgroup']=(8,8,1)
            return kernel(**kwargs)
        return wrapper
    old_mask=original_dispatch(optimized_mask)
    old_mask_v2=original_dispatch(optimized_mask_v2)
    def set_variant(variant):
        fast=variant=='optimized'
        if args.candidate in ('conv','stack'):
            language._sconv_kv=optimized if fast else reference
            if optimized_convnorm is not None:
                language._sconv_kv_norm=optimized_convnorm if fast else reference_norm
        if args.candidate in ('mask','stack'):
            language._mask_kernel=optimized_mask if fast else old_mask
            language._mask_v2_kernel=optimized_mask_v2 if fast else old_mask_v2
    source = mlx_vlm.__file__
else:
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import generate_step, generation_stream, wired_limit
    from mlx_lm.sample_utils import make_sampler
    import mlx_lm.models.inkling_mm_model as language
    optimized = language._expert_gate_up_kernel
    optimized_combine=getattr(language,'_expert_combine_sconv_kernel',None)
    def set_variant(variant):
        language._expert_gate_up_kernel = optimized if variant == 'optimized' else None
        if optimized_combine is not None:
            language._expert_combine_sconv_kernel=optimized_combine if variant=='optimized' else None
    source = mlx_lm.__file__

emit({'kind':'metadata', 'engine':args.engine, 'source':source,
      'model':args.model, 'mlx':mx.__version__, 'device':mx.device_info(),
      'args':vars(args), 'loadavg':os.getloadavg(), 'tf32':os.environ.get('MLX_ENABLE_TF32')})
t0 = time.perf_counter()
model, processor = load(args.model, lazy=True)
params = tree_flatten(model.parameters())
chunk, nb, total = [], 0, 0
for _, value in params:
    chunk.append(value)
    nb += value.nbytes
    total += value.nbytes
    if nb >= 2 * 2**30:
        mx.eval(chunk)
        chunk, nb = [], 0
if chunk:
    mx.eval(chunk)
del params, chunk
model.eval()
emit({'kind':'load', 'seconds':time.perf_counter()-t0,
      'parameter_gib':total/2**30, 'active_gib':mx.get_active_memory()/2**30})

tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
prompt_root = Path('<workdir>/inkling-contrib-review-2026-10-29/bench/prompts')
prompts = {}
for name in args.prompts.split(','):
    text = (prompt_root/(name+'.txt')).read_text()
    if args.engine == 'k4':
        prompt = apply_chat_template(processor, model.config, text, num_images=0, reasoning_effort='none')
        prompts[name] = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        ids = tokenizer.apply_chat_template([{'role':'user','content':text}],
                                           tokenize=True, add_generation_prompt=True,
                                           reasoning_effort='none')
        prompts[name] = list(ids['input_ids']) if hasattr(ids, 'keys') else list(ids)
    emit({'kind':'prompt', 'name':name, 'tokens':len(prompts[name])})

base_caches={}

def run(name, variant, pair, warmup=False):
    set_variant(variant)
    prompt_cache=copy.deepcopy(base_caches[name]) if args.reuse_prefix else None
    prompt_ids=prompts[name][-1:] if args.reuse_prefix else prompts[name]
    if prompt_cache is not None:
        mx.eval([c.state for c in prompt_cache])
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    first = None
    tokens = []
    if args.engine == 'k4':
        generator = generate_step(mx.array([prompt_ids]), model, None, None,
                                  max_tokens=args.tokens, temperature=0.0,
                                  top_p=1.0, top_k=0, prefill_step_size=2048,
                                  prompt_cache=prompt_cache)
    else:
        generator = generate_step(mx.array(prompt_ids), model,
                                  max_tokens=args.tokens, sampler=make_sampler(temp=0.0),
                                  prefill_step_size=2048,prompt_cache=prompt_cache)
    for token, _ in generator:
        now = time.perf_counter()
        if first is None:
            first = now
        tokens.append(int(token))
        last = now
    mx.synchronize()
    end = time.perf_counter()
    assert len(tokens) == args.tokens, (len(tokens),args.tokens)
    row = {'kind':'run', 'prompt':name, 'variant':variant, 'pair':pair,
           'warmup':warmup, 'tokens':tokens, 'ttft_s':first-start,
           'decode_ms_per_token':1000*(last-first)/(len(tokens)-1),
           'decode_tps':(len(tokens)-1)/(last-first), 'total_s':end-start,
           'peak_gib':mx.get_peak_memory()/2**30, 'loadavg':os.getloadavg()}
    emit(row)
    return row

with wired_limit(model, [generation_stream]):
    for name in prompts:
        if args.reuse_prefix:
            set_variant('baseline')
            lm=model.language_model if args.engine=='k4' else model
            base_caches[name]=lm.make_cache()
            prefill_start=time.perf_counter()
            for start in range(0,len(prompts[name])-1,2048):
                ids=mx.array([prompts[name][start:min(start+2048,len(prompts[name])-1)]])
                kwargs={'logits_to_keep':1} if args.engine=='k4' else {}
                lm(ids,cache=base_caches[name],**kwargs)
                mx.eval([c.state for c in base_caches[name]])
                mx.clear_cache()
            emit({'kind':'shared_prefix','prompt':name,'tokens':len(prompts[name])-1,
                  'seconds':time.perf_counter()-prefill_start})
        for variant in ['baseline','optimized']:
            run(name, variant, -1, True)
        ratios, deltas = [], []
        reference_tokens = None
        all_equal = True
        for pair in range(args.pairs):
            rows = {}
            order = ['baseline','optimized'] if pair % 2 == 0 else ['optimized','baseline']
            for variant in order:
                rows[variant] = run(name, variant, pair)
                if reference_tokens is None:
                    reference_tokens = rows[variant]['tokens']
                all_equal &= rows[variant]['tokens'] == reference_tokens
            b, a = rows['baseline'], rows['optimized']
            ratios.append(b['decode_ms_per_token']/a['decode_ms_per_token'])
            deltas.append(b['decode_ms_per_token']-a['decode_ms_per_token'])
            emit({'kind':'pair', 'prompt':name, 'pair':pair,
                  'speedup':ratios[-1], 'saved_ms_per_token':deltas[-1],
                  'same_tokens':b['tokens']==a['tokens']})
        emit({'kind':'summary','prompt':name,'paired_speedup_median':statistics.median(ratios),
              'paired_speedup_min':min(ratios),'paired_speedup_max':max(ratios),
              'saved_ms_per_token_median':statistics.median(deltas),
              'all_tokens_identical':all_equal})
        base_caches.pop(name,None)
