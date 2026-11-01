"""(b)/(c) mx.fast.scaled_dot_product_attention at Inkling-Small shapes.

32 query heads, 8 kv heads, head_dim 128, bf16.
  decode : 1 query vs context T in {512, 4096, 32768}; mask None / additive
           bf16 [1,32,1,T] / bool [1,1,1,T] / additive broadcast [1,1,1,T];
           plus the relative-bias construction (rel @ proj, gather by distance)
           fused into the same graph, as the vendored inkling_mlx does it.
  prefill: T=2048 queries vs 2048 keys; mask "causal" / bool causal array /
           bool sliding-window(512) array / additive bf16 [1,32,T,T].
Derived: attention-only time per decoded token for the 42-layer model
(35 sliding layers at 512 keys + 7 global layers at T keys).
"""

import mlx.core as mx

from benchutil import Recorder, timeit

NQ, NKV, D = 32, 8, 128
SCALE = 1.0 / D  # Inkling: q/k are RMS-normed per head -> 1/d, not 1/sqrt(d)
REL = 1024
rec = Recorder("sdpa")


def kv_bytes(T):
    return 2 * NKV * T * D * 2


def sdpa(q, k, v, mask=None):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=mask)


# ---------------- decode ----------------
q = mx.random.normal((1, NQ, 1, D)).astype(mx.bfloat16)
proj = mx.random.normal((16, REL)).astype(mx.bfloat16)
rel = mx.random.normal((1, 1, NQ, 16)).astype(mx.bfloat16)
mx.eval(q, proj, rel)
decode_ms = {}
for T in (512, 4096, 32768):
    rec.note(section=f"decode T={T}")
    k = mx.random.normal((1, NKV, T, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, NKV, T, D)).astype(mx.bfloat16)
    add_mask = (mx.random.normal((1, NQ, 1, T)) * 0.1).astype(mx.bfloat16)
    bool_mask = mx.arange(T) >= max(0, T - 512)
    bool_mask = bool_mask.reshape(1, 1, 1, T)
    add_bcast = (mx.random.normal((1, 1, 1, T)) * 0.1).astype(mx.bfloat16)
    mx.eval(k, v, add_mask, bool_mask, add_bcast)
    r = rec.row(f"decode T={T} mask=None", lambda: sdpa(q, k, v), nbytes=kv_bytes(T), T=T)
    decode_ms[(T, "none")] = r["med_ms_r8"]
    r = rec.row(f"decode T={T} mask=additive bf16 [1,32,1,T]", lambda: sdpa(q, k, v, add_mask),
                nbytes=kv_bytes(T) + NQ * T * 2, T=T)
    decode_ms[(T, "add")] = r["med_ms_r8"]
    rec.row(f"decode T={T} mask=bool [1,1,1,T] (window)", lambda: sdpa(q, k, v, bool_mask), nbytes=kv_bytes(T), T=T)
    rec.row(f"decode T={T} mask=additive bf16 [1,1,1,T] broadcast", lambda: sdpa(q, k, v, add_bcast),
            nbytes=kv_bytes(T), T=T)

    # relative bias built the vendored way: rel_logits [1,32,1,REL]; distance = (T-1) - kv_pos
    def bias_then_sdpa():
        rel_logits = mx.swapaxes(rel @ proj, 1, 2)  # [1,32,1,REL]
        dist = (T - 1) - mx.arange(T)
        gather = mx.clip(dist, 0, REL - 1).reshape(1, 1, 1, T)
        gather = mx.broadcast_to(gather, (1, NQ, 1, T))
        bias = mx.take_along_axis(rel_logits, gather, axis=-1)
        bias = mx.where((dist < REL).reshape(1, 1, 1, T), bias, mx.array(0.0, mx.bfloat16))
        return sdpa(q, k, v, bias)

    r = rec.row(f"decode T={T} rel-bias build (vendored style) + sdpa", bias_then_sdpa,
                nbytes=kv_bytes(T) + NQ * T * 2, T=T)
    decode_ms[(T, "bias+sdpa")] = r["med_ms_r8"]

    # unfused reference for the same shapes
    def unfused():
        qq = q.reshape(1, NKV, NQ // NKV, 1, D) * SCALE
        s = qq @ mx.expand_dims(k, 2).swapaxes(-1, -2)
        s = s + add_mask.reshape(1, NKV, NQ // NKV, 1, T)
        p = mx.softmax(s, axis=-1, precise=True)
        return (p @ mx.expand_dims(v, 2)).reshape(1, NQ, 1, D)

    rec.row(f"decode T={T} UNFUSED matmul+softmax+matmul (additive mask)", unfused, nbytes=kv_bytes(T), T=T)
    del k, v, add_mask, bool_mask, add_bcast
    mx.clear_cache()

# 42-layer attention-only budget per decoded token (r8 medians)
for T in (4096, 32768):
    swa = 35 * decode_ms[(512, "none")]
    glob_add = 7 * decode_ms[(T, "add")]
    glob_bias = 7 * decode_ms[(T, "bias+sdpa")]
    rec.note(label=f"42-layer attention per token at ctx {T}",
             ms_swa35_at512_nomask=round(swa, 3), ms_global7_add_mask=round(glob_add, 3),
             ms_global7_bias_build=round(glob_bias, 3),
             total_ms_add=round(swa + glob_add, 3), total_ms_bias=round(swa + glob_bias, 3))

# ---------------- prefill ----------------
for T in (2048,):
    rec.note(section=f"prefill T={T}")
    qT = mx.random.normal((1, NQ, T, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, NKV, T, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, NKV, T, D)).astype(mx.bfloat16)
    li = mx.arange(T)[:, None]
    ri = mx.arange(T)[None]
    causal = (li >= ri).reshape(1, 1, T, T)
    window = ((li >= ri) & (li < ri + 512)).reshape(1, 1, T, T)
    add_full = (mx.random.normal((1, NQ, T, T)) * 0.1).astype(mx.bfloat16)  # 256 MB
    mx.eval(qT, k, v, causal, window, add_full)
    flops = 2 * 2 * NQ * T * T * D  # QK^T and PV, dense (no causal skipping)
    for label, m in (("'causal'", "causal"), ("bool causal array", causal), ("bool window512 array", window),
                     ("additive bf16 [1,32,T,T]", add_full)):
        r = rec.row(f"prefill T={T} mask={label}", lambda m=m: sdpa(qT, k, v, m), reps=(1, 4), iters=10, T=T)
        rec.note(label=f"prefill T={T} {label} dense-TFLOPS", tflops=round(flops / (r["med_ms_r4"] * 1e-3) / 1e12, 2))

    # bias construction cost at prefill (vendored style): [1,32,T,T] gather
    def bias_build():
        relT = mx.random.normal((1, T, NQ, 16)).astype(mx.bfloat16)
        rel_logits = mx.swapaxes(relT @ proj, 1, 2)  # [1,32,T,REL]
        dist = li - ri
        gather = mx.broadcast_to(mx.clip(dist, 0, REL - 1)[None, None], (1, NQ, T, T))
        bias = mx.take_along_axis(rel_logits, gather, axis=-1)
        valid = (dist >= 0) & (dist < REL)
        return mx.where(valid[None, None], bias, mx.array(0.0, mx.bfloat16))

    rec.row(f"prefill T={T} rel-bias build only [1,32,T,T] (vendored style)", bias_build, reps=(1, 4), iters=10, T=T)
    del qT, k, v, causal, window, add_full
    mx.clear_cache()
