# A1 adversarial review: K1 `metal-banded-fa` (Metal kernel for `GGML_OP_FLASH_ATTN_EXT_BANDED` + `q_offset`)

Reviewer A1, 2026-10-28/29. Reviewed read-only: branch `metal-banded-fa` (`90290efad`, four commits on
`pr-25731` = `3582c5ee0`) in `upstream/llama.cpp-inkling`. My own builds, test cases and logs live in the
worktree `upstream/llama.cpp-a1` (branch `a1-review`, one commit `845ec0484` on top of K1's head) and in
`package/k1-metal-banded-fa/review/`. Nothing pushed. The protocol followed is `research/00-method.md` section 3;
the semantics checked against are `research/R5-reference-semantics.md` 3c/6.1 and the op definition in
`research/R2-llamacpp-metal.md` 7.1 (`bias = rel[q + q_offset - k]` iff `0 <= dist < E`, else 0; visibility by
the mask only; scale `1/128` applied before the bias; masked scores `-inf`; tau outside the op).

## Verdict per claim

| claim (PR.md / README.md / commit messages) | verdict | evidence |
|---|---|---|
| The Metal kernels implement the op's semantics (bias = `rel[q + q_offset - k]` for `0 <= dist < E`, masked keys excluded, GQA by q head, F32/F16/BF16 table) in all three kernel families | **CONFIRMED** | code review (section 1); 26/26 K1 cases + 145/145 + 8/8 + 12/12 A1 cases vs the CPU op (section 2), NMSE 1e-13..7e-8 at every Inkling shape incl. arbitrary `q_offset`, padded tails, windows, batch, sparse, 1M keys; `test-flash-attn-bias` 16/16 on MTL0 |
| `q_offset` is honoured consistently by CPU, CUDA (scalar + MMA) and Metal | **CONFIRMED** (CPU, Metal) / **PLAUSIBLE** (CUDA, by reading only; untested) | section 1 |
| The model-side change (256-padded cache views, raw mask, `q_offset = pos_max + 1 - n_tokens`) is safe | **CONFIRMED** | section 1 (contiguity contract), 4c layer 0 (1.7e-4 from the CPU op on real inputs), 4d (32K/64K correct) |
| The plain `FLASH_ATTN_EXT` is untouched (5020 OK) | **CONFIRMED** | section 2 |
| Banded within 1.4-1.6x of the plain op at the same shapes | **CONFIRMED** (and pessimistic: 0.82x-1.24x at the min of 3) | section 3 |
| Prefill no longer collapses with context: ~340 tok/s at 32K | **CONFIRMED** (270 tok/s at 32.8K and 235 tok/s at 65.5K through the server on a loaded machine; vs 15.6 reported for the CPU fallback) | 4d |
| Kernel +6% prefill vs the dense chain | **CONFIRMED** (+3-10%) | 4f |
| Kernel +28-30% decode vs the dense chain | **NOT CONFIRMED** (every pair in my window shows the opposite sign; load-dominated; needs a quiet A/B) | 4e, 4f |
| Greedy output "equivalent" to `-fa off`; divergences are near-tie flips | **BROKEN as stated, CONFIRMED in substance**: the comparison baseline was wrong (the dense chain is itself 4% / KLD 0.15 from the reference), the divergences vs the reference are 0.5-0.7-nat disagreements, not near-ties; but the cause is the model's amplification of f16-level attention differences, shown by the layer-by-layer dump, and the kernel is the closest implementation to the reference | 4a, 4b, 4c |
| PPL `-fa on` vs `-fa off` within 0.5% | **BROKEN** (2.9% apart; kernel +1.1% from the reference, dense +4.0%); the kernel is the better of the two | 4b |
| e2e numbers measured on "this PR" | **BROKEN**: measured on build 11985 (`4ddc6ed6d`, before `q_offset` and the final kernel commit) | section 5 |
| Peak RSS ~93 GiB either way | **CONFIRMED** (90.7-94.9 GiB depending on the tool and `-c`, same for both paths) | 4d |
| 26/26 + 13 cases cover "every kernel family and variant the model can hit" | **CONFIRMED** (and the sparse path, bf16/q8_0/q4_0/f32 KV, F16/BF16 tables, batch, 1M keys beyond it) | section 2 |
| PR text ready to submit | **BROKEN**: AI-written PR description and commit messages are prohibited by CONTRIBUTING.md/AGENTS.md; no AI-disclosure line; `Co-Authored-By` instead of `Assisted-by`; the `ggml.h` API change belongs to the op's PR, not a Metal follow-up | section 6 |

**Overall: the kernel is correct and fast (CONFIRMED); the package is not submittable as is (BROKEN on process, on the
e2e evidence, and on the decode claim).** No kernel defect found.



## 1. Code review (the diff `pr-25731..metal-banded-fa`, 19 files, +361/-92)

Every hazard listed in the brief was checked by reading the final kernels in full (not the hunks only):

| hazard | finding |
|---|---|
| distance when `q_offset != n_kv - n_q` | half8x8 (`fa_common.metal:444-456`): `d0 = iq1 + j + q_off - (ick + 2*tiisg)` for key `ick + 2*tiisg`, `d0 - 1` for `+1`; tensor (`fa_tensor.metal:198-206`): `d = iq1 + j + q_off - (ic + ii*NW + tiisg)`; vec (`fa_vec_common.metal:417-428`): `d = iq + q_off - ik`. All three match the CPU definition `rel_dist = iq1 + q_offset - ic` (`ops.cpp:8994`). Verified at op level with offsets 0, 4096, 7780, 13192, 2^20, -300 and the padded-tail offsets (section 2). |
| kv-pad path, last partial block (`ic` reset to 0) | `ick = ic` is captured before the pad switch in both the half8x8 (`:205`) and vec (`:200`) kernels and the bias uses `ick`; the tensor kernel never pads (host requires `ne11 % 64 == 0`). Verified with n_kv 4100, 8209, 8193, 16385, 65537, 70000 and the vec kvpad variants (`kvpad=1_rel=*` pipelines in the log). |
| sparse-index decode path | the vec kernel uses the gathered absolute index `spidx[NE*tx + ty]` and skips `-1` padding (`ik >= 0`), and tests every block (`has_rel_blk` forced on under `has_sparse`). The model can never reach this path (`ggml_flash_attn_ext_set_n_kv_max` asserts the plain op, so `op_params[4]` stays 0 for the banded op) but the Metal host reads `op_params[4]` for the banded op anyway; I exercised it by writing the hint directly: 4 cases OK (`sparse=1` pipelines). |
| `blk_cur == 2` mask shortcut | the bias is added before the `blk_cur != 2` branch in both the half8x8 (`:444` vs `:458`) and tensor (`:198` vs `:208`) kernels; `blk_cur == 0` blocks are skipped before anything, which is correct (fully masked). |
| GQA head mapping | the bias row pointer is formed from `iq2` (the query head) with `nb_rel1`; K/V use `ikv2 = iq2/(ne02/ne_12_2)`. Verified with 32/8, 16/1, 8/8, 1/1 heads. |
| bias dtype | `fa_rel_load` (`kernels/common.h:51-57`): F16 via `half`, BF16 as raw bits `<< 16` (exact), F32 direct; selected by a function constant per pipeline (`rel=0/1/2` in the pipeline name, `-1` = plain op, so no pipeline-cache collision). Verified rel F16/BF16 at 512x4096, 1x4096, 64x4100, 7x4100. |
| band test when `n_kv < E`, blocks straddling dist 0 or dist E | `has_rel_blk` = `dist_max(last query, first key) >= 0 && dist_min(first query, last key) < E` is conservative and the per-score test is exact; verified n_kv 1..1000 with E = 1024, every n_kv around 64-multiples, the E = 512 window cases and the band edge at 8192 with E = 512. |
| batch dimension | `iq3 % ne_rel3` with `ne_rel3 in {1, ne03}` from the host (`ne53`); verified ne3 = 2 with rel ne3 1 and 2 (prefill, decode, unaligned) and ne3 = 3. |
| int32 overflow up to 1M keys | kernel-side sums are `int` from `ushort`/`int32` terms (`iq1 + Q - 1 + q_off - ick` < 2^31 for n_kv <= 1M, q_off <= 2^24); byte offsets are `int * uint64 -> uint64`; the CUDA side asserts `|q_offset| <= 2^24` (float ABI). Verified n_kv = 1,048,576 (n_q 1 and 32) and 262,144 (n_q 512), q_off = 2^20. |
| `supports_op` vs the model | requires `src[5]` present, no sinks, rel type F32/F16/BF16, `nb[0] == type_size` (strided rows allowed, verified), `ne1 == n_head`, `ne2 == n_q`, `ne3 in {1, ne03}`, then the plain-op checks (head 64/128, K/V types, bfloat capability). The model passes rel = reshape of an f32 `mul_mat` (contiguous), an f16 contiguous mask (`ggml_flash_attn_ext_banded` asserts contiguity; the raw `get_kq_mask()` tensor is contiguous), f16 K/V cache views with head 128. The mask `ne[1] == n_tokens` (`llama-graph.cpp:3120/3127`), so the `bc_mask` variant is taken when `n_tokens % 8 != 0` (verified n_q 7, 20, 2). |
| alloc-size hook | `ggml_backend_metal_buffer_type_get_alloc_size` adds pad/blk/tmp/kv_f16/idx scratch for the banded op with the same formulas as the plain op (`ggml-metal.cpp:230-231`). |
| CPU `q_offset` | `op_params[5]` read only when `rel != NULL` (`ops.cpp:8860`); the tiled CPU kernel is still disabled for the banded op (`:9564`), so the scalar path is the only one that matters. |
| CUDA `q_offset` (not compiled here) | scalar kernel: a kernel argument replacing the local `n_kv - n_q` (`fattn-banded.cu:42,112,230-243`), consistent with the CPU. MMA path: `q_offset` is smuggled through the unused ALiBi `m0` slot (`fattn-common.cuh:1236-1257`), decoded `int(m0)` in `flash_attn_ext_f16` (`fattn-mma-f16.cuh:1896`) and plumbed positionally through `process_tile`/`iter`; `get_alibi_slope()` returns 1 for `max_bias <= 0` (`common.cuh:962-966`), so a non-unit `m0` cannot leak into the slopes; the dist expression matches the CPU at both MMA call sites (`:769`, `:852`). Reviewed by hand only: it is **untested** (no NVIDIA GPU on this machine) and must be run through `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT_BANDED` before any submission. |
| model side (padded views, dropped mask `cont`) | safe. The banded branch is only taken when `get_n_kv_pos_contiguous()` returns `pos_max + 1` (`llama-kv-cache.cpp:1271-1313`: one stream, one sequence, the ubatch is the monotonic tail, and every cell `0..pos_max` is non-empty, holds position `p` at cell `p` and belongs to that sequence); `q_offset = pos_max + 1 - n_tokens` therefore equals the query position minus the cell index for every visible cell. Cells in `[pos_max + 1, n_kv)` are empty, other-sequence or future cells and the KQ mask marks all of them `-inf`; the cache is zero-initialised so their K/V rows are finite. The reserve context reports the full cache (`:2946-2950`), so the reserve graph still sizes the banded layout and `n_kv_flash <= k_fa->ne[2]` holds. Multi-sequence or non-identity layouts fall back to the dense path exactly as before K1 (`sinfo.n_stream() != 1 || n_seqs_unq != 1` -> 0). The removed `ggml_cont` of the mask view and the removed `n_kv_flash` K/V views were the only model-side changes; both masks are `[n_kv, n_tokens, 1, 1]` so `n_stream == 1` always holds for the retained assert. |
| meta backend (multi-GPU split) | `handle_flash_attn_ext_banded` splits by heads only (`ggml-backend-meta.cpp:797-805`), the KV extent is never sharded, so `q_offset` needs no per-shard adjustment. |

Non-defect code observations:

- `ggml_metal_op_flash_attn_ext` reads `op_params[4]` (n_kv_max) for the banded op although no API can set it for this op; harmless (tested) but the sparse branch is dead code for Inkling.
- The half8x8 kernel does two scalar device gathers per lane per query row inside the softmax loop and recomputes `prel + (iq1 + j)*nb_rel2` per use (the fourth commit); the tensor kernel does `NC` gathers per lane per row. This is the measured 1.3-1.6x cost over the plain op on the in-band blocks (section 3); it is a performance note, not a correctness one.
- `test-backend-ops` tolerance for the banded cases (`5e-4`, `1e-3` above 8K) is four orders of magnitude above what the kernels achieve (measured NMSE `2e-8`..`7e-8` for f16 KV at every Inkling shape, `4e-5` only at n_kv = 1M where the output is a mean over 1M random rows). The comment "fp16 VKQ accumulation error grows with the KV length" does not apply: the CPU reference accumulates in f32 when `rel != nullptr` (`ops.cpp:8928`). A tighter bound (`1e-6` for f16 KV, say) would make a single wrong block in a 16-block band fail instead of hiding at ~2e-3; recommended, not required.
- `v_trans` can only be false under `-fa on` (llama sets `v_trans = !flash_attn`), so the retained `ggml_transpose` branch is dead but harmless.

## 2. Op-level tests (MTL0 against the CPU reference, my build of `metal-banded-fa` + my test cases)

1. **K1's 26 cases**, rebuilt from the committed sources: 26/26 OK (`review/test-banded-k1cases-a1build.log`). CONFIRMED.
2. **The PR's standalone `test-flash-attn-bias` on MTL0** (its `cuda` backend selector picks the first GPU, which is
   Metal here; it compares against a naive float oracle with a per-element max-abs tolerance of 2e-5 f32 / 2e-3 f16):
   `small` 10/10 PASS, `medium` 5/5 PASS (incl. `large_v_16k`, the f16-accumulator saturation case, and `decode_8k_e1024`),
   `overflow` kernel PASS (`review/test-flash-attn-bias-mtl0.log`). K1 only ran it on the CPU.
3. **Plain `FLASH_ATTN_EXT` regression** (the same three kernel families with `has_rel = false`): 5020 OK, 359 not supported
   on MTL0, 0 FAIL (`review/test-plain-fa-a1build.log`), identical to K1's `bench/test-plain-fa.log`. CONFIRMED.
4. **145 adversarial cases of my own** (`review/a1-tests.patch`; the helper gained `q_offset`, `window`, `ne3`, `rel_ne3`
   and `n_kv_max`, and the compare callback prints the NMSE and the largest element difference of every banded case):
   **145/145 OK, 0 FAIL, 0 not supported** (`review/test-banded-a1cases.log`, 171 cases including K1's; plus the 8 cases of group P below). Groups:
   - A: n_q in {1, 2, 7, 20, 64, 512, 1024} at n_kv 4100 (unaligned) and n_q in {20, 64, 512, 1024} at 4096 (aligned), 32/8 heads, E 1024, causal.
   - B: n_kv sweep at n_q 512: 512, 513, 575, 576, 577, 639, 640, 641, 1023..1025, 1535..1537, 2047..2049, 4095..4097, 8191..8193, 16383..16385, and 65535, 65536, 65537, 70000 (8/2 heads); at n_q 1: 1, 2, 31..33, 63..65, 127..129, 1023..1025, 2047..2049, 4095..4097, 65535..65537, 70000.
   - C: E = 512 with the 512 window at n_kv 512/1024/1536 (n_q 512) and 1024 (n_q 1); E = 512 under a causal mask at 8192 (band edge inside the KV, prefill and decode).
   - D: n_kv < E: 1, 7, 100, 500, 1000 (n_q 1) and 64, 100, 500, 1000 (n_q 64).
   - F: q_offset other than the tail: 0, 4096, 7780, 13192 (queries entirely past the band) and 2^20 at 512x8192 causal; -300 with no mask at 512x4096 (queries before the KV); 0, 1023, 1024, 2^20 at 1x4096 without a mask; 2048 at 1x4096 causal; kv_pad 239 with q_offset 0; 1000 at 20x4100 (half8x8 + pad) and at 7x4100 with kv_pad 60 (vec + pad).
   - G: no mask at 1x8192, 512x8209 (pad path without a mask), 7x4100, 64x4100, 512x4096 (tensor kernel without a mask).
   - H: sliding window 512 under E = 1024 at 512x4096, 1x8192, 7x4100.
   - I: heads 1/1 (512 and 1 queries), 8/8, 16/1, and d = 64 at 32/8 (512, 1, 7 queries).
   - J: K/V bf16 (512x4096, 1x4096, 7x4100, 512x8209), q8_0 (1x4100, 7x4100, 512x8209, 512x4096), q4_0 (512x4096, 1x4096), f32 (1x4096, 512x1024, 64x1024).
   - K: bias table F16 and BF16 at 512x4096, 1x4096, 64x4100, 7x4100.
   - L: ne3 = 2 with rel ne3 1 and 2 at 64x1024, 1x2048, 512x2048, 512x2049; ne3 = 3 with rel ne3 1.
   - M: strided K/V and table rows at 512x4096, 1x4096, 7x4100.
   - N: the n_kv_max hint (sparse vec path) at 1x4096, 7x2048, 1x8192 with the 512 window, 2x1024.
   - O: n_kv = 1,048,576 (n_q 1 and 32) and 262,144 (n_q 512).
   - Q (run separately, `review/test-banded-a1cases-Q.log`, 12/12 OK; summary `review/groupQ-summary.txt`): realistic input
     magnitudes, because the harness's U(-0.25, 0.25) q/k give logits of std 0.002 and near-flat softmaxes that cannot
     expose f16 rounding of the probabilities: unit-RMS q/k (U(-1.7, 1.7)) with a table up to +-4, and q/k U(-3, 3) with
     a table up to +-8 (peaked softmax, lazy-rescale events), at 512x8192, 512x8209, 1x8192, 7x4100, 512x1024/E=512/window,
     512x512. NMSE 1.5e-9..5.6e-8; the largest single-element differences 3.4e-4 (+-1.7/+-4) and 8.5e-4 (+-3/+-8) on values of
     ~0.15-0.3 (2-3e-3 relative, the f16 rounding of one probability), the same order as the dense chain's deviation from
     the CPU op on real data (4c).
   - P (run separately, `review/test-banded-a1cases-P.log`, 8/8 OK): the exact shapes the model sends, i.e. 256-padded cache
     views with the padding masked and `q_offset = pos_max + 1 - n_tokens`: 512 x 32794 + 230 pad, 1 x 32795 + 229 pad
     (the p_32k decode step), 512 x 1000 + 24 pad and 1 x 777 + 247 pad with the 512 window and E = 512 (SWA layers),
     226 x 226 + 30 pad for both layer kinds (p_short), and ubatch 2048 (2048 x 2048 and 2048 x 8192).

   Error statistics across the 171 cases: NMSE 1e-13..7e-8 for every f16/f32/q8_0/q4_0 KV case, 2e-6 and 2e-5 for bf16 KV at
   512x4096 / 512x8209 (the bf16 half8x8/tensor kernels' own precision, the plain op is the same), 4e-5 at n_kv = 1M (the output
   is a mean over 1M random rows, ref RMS 1.4e-4, max-abs 4e-6). Largest element difference: 4.8e-5 on f16 KV at the Inkling
   shapes (values ~0.02), 9e-5 on bf16 KV, 1.1e-4 on the bf16-KV/bf16-table E = 8 case. The f32-KV cases are at 1e-13..2e-12
   against the 2e-6 bound. No case is anywhere near the test's tolerance, and no shape class behaves differently from its
   neighbours, which is the signature one expects from an exact index computation plus f16 P rounding.
5. Kernel variants exercised (pipeline names compiled in the log): tensor f16/bf16 with table F32/F16/BF16, with and without
   a mask; half8x8 f16/bf16/f32 with kvpad 0/1, bc_mask 0/1, table F32/F16/BF16; vec f16/bf16/f32/q8_0/q4_0 with Q = 1 and 2
   (ne 2/4), kvpad 0/1, sparse 0/1, table F32/F16/BF16, nsg 1/2/4, nwg 32 with the reduce kernel; the pad (mask 0/1, C 32/64),
   blk (Q 8/32), kv_f16 (q4_0/q8_0) and vec_idx auxiliaries.



## 3. Op-level performance (my GPU window, `test-backend-ops perf`, three interleaved repeats, min)

Same shapes as K1's table (32/8 heads x 128, E = 1024, f16 KV, f32 table, causal mask; plain = `FLASH_ATTN_EXT` at
`hsk=hsv=128, nh=8, nr23=[4,1]`). Load average 13-30 from other jobs during the window, so only the minimum
is meaningful; the spread of the three repeats is shown (`review/e2e/perf-{banded,plain}-{1,2,3}.log`).

| shape (n_q x n_kv) | banded min | banded repeats | plain min | plain repeats | banded / plain |
|---|---:|---|---:|---|---:|
| 512 x 8192 | 6.02 ms | 6.02, 8.31, 8.80 | 7.31 ms | 7.31, 8.59, 10.72 | 0.82x |
| 512 x 32768 | 33.0 ms | 33.0, 39.1, 61.6 | 26.5 ms | 26.5, 29.6, 39.0 | 1.24x |
| 1 x 8192 | 113 us | 113, 141, 359 | 96 us | 96, 103, 106, 110, 149, 151 | 1.18x |
| 1 x 32768 | 535 us | 535, 539, 763 | 495 us | 495, 502, 555, 556, 565, 573 | 1.08x |

K1's "within 1.4-1.6x of the plain op" is CONFIRMED as an upper bound and is pessimistic: at the minimum the banded op is
0.82x-1.24x of the plain op on these four shapes. The minimum plain numbers (7.3 ms at 512 x 8192, 26.5 ms at 512 x 32K)
are still 1.8-2x above R2's idle-GPU ceiling (3.47 / 14.4 ms), so the machine was not quiet in my window either; the
ratios are what can be trusted, not the absolute values. The cost of the bias is bounded by its in-band gathers, so the
relative overhead shrinks as n_kv grows past E (1.08x at 32K decode).



## 4. End to end (my GPU window 04:32-04:58Z, 25 min incl. three chained extensions, `review/e2e/`, build `90290efad (11987)` = the final branch; reference = pristine
`3582c5ee0` with `-fa on`, i.e. the CPU banded op for every layer that takes the banded path)

Common flags for the servers: `-np 1 --no-kv-unified -ngl 99 --jinja --temp 0`, raw `/completion`, `n_predict` as stated,
`top_k 1`, `n_probs 5`, `cache_prompt false` (`review/a1_greedy.py`). All runs on the UD-IQ3_XXS GGUF.

### 4a. Greedy equivalence against the reference (`-c 8192`, 96 tokens, p_short 226 tokens, p_2k 2072 tokens)

| prompt | first divergence | kernel's margin (top1 - top2) there | reference's margin there | chosen-token logprob difference there | mean / max abs logprob difference on the common prefix |
|---|---:|---:|---:|---:|---:|
| p_short | token 11 of 96 | 0.071 (prefers 200008 over 200001) | 0.851 (prefers 200001) | 0.504 | 0.032 / 0.138 (11 steps) |
| p_2k | token 21 of 96 | 0.302 (prefers " on" over " Based") | 0.616 (prefers " Based") | 0.698 | 0.065 / 0.392 (21 steps) |

Tokens do not match exactly. Both divergences are at low-margin steps, but by the brief's criterion (margin < ~1e-2 nats)
they are **not** near-ties: the kernel's logprobs differ from the reference by 0.5-0.7 nats at those steps, and by
0.03-0.06 nats on average (up to 0.39) on the steps before them (`review/e2e/greedy-kernel-vs-ref.txt`). On p_short
the kernel diverges from the reference at the same token where K1 saw `-fa on` diverge from `-fa off`, and the
reference's text continues like K1's dense-chain text; i.e. at that step the dense chain agrees with the reference and
the kernel does not. The p_2k continuation degenerates into a "Based on the table of contents, Based ..." loop on
**both** the kernel and the reference (greedy decoding on an unformatted raw prompt), so that loop is the model's, not
the kernel's. Whether the 0.03-0.7-nat differences are the expected f16-level noise of a different attention
implementation or a systematic error is decided in 4b/4c: they are the model's amplification of an op-level difference
of 1.7e-4 (and the dense chain, which K1 used as the baseline, is further from the reference than the kernel is).

### 4b. Perplexity on p_8k (8218 tokens, `-c 4096 -b 2048`, 2 chunks; `review/e2e/ppl-*.err`)

| build / mode | chunk 1 | final PPL |
|---|---:|---:|
| final branch `-fa on` (this kernel) | 7.1517 | 8.3361 +- 0.4238 |
| final branch `-fa off` (dense Metal chain) | 7.3931 | 8.5764 +- 0.4387 |
| pristine `-fa on` (CPU banded reference) | 6.8526 | 8.2437 +- 0.4155 |

The kernel and the dense chain disagree by 2.9% in PPL (0.028 nats per token), outside the brief's 0.5% band, with the
kernel on the *lower* side. Against the reference the kernel is +1.1% (8.3361 vs 8.2437; chunk 1 +4.4%) and the dense
chain is +4.0% (chunk 1 +7.9%): the kernel is the closer of the two GPU paths, by a factor of ~3.6 on the final
number, but it is not inside a 0.5% band of the reference either. In all three runs the 35 SWA layers run the same
dense Metal chain for every position past the SWA cache (1024 cells), so the differences come from the 7 global layers
(and the SWA layers' first two ubatches per chunk) only. The dense chain, which R2 found identical to the CPU op at the
op level (NMSE 2e-7 on identical inputs), deviates 4% at the model level, so the model-level deviation is not a
measure of op-level arithmetic alone (the dense path's `rel_idx` gather and `GET_ROWS` plumbing is a different
code path from both the CPU and the Metal banded op); it is consistent with the field report that `-fa 0` degrades
Inkling's quality on ROCm. Section 4c puts per-token numbers on this.

### 4c. KL divergence against the reference (p_8k, `-c 2048 -b 2048`, 4 chunks; `llama-perplexity --kl-divergence`
with the reference log-probabilities written by the pristine build; `review/e2e/kld-*.err`)

| comparison (p_8k, 4 chunks of 2048, ~8K scored tokens) | mean KLD | median KLD | 90% / 95% / 99% KLD | max KLD | RMS dp | same top-1 token | PPL(Q)/PPL(base) |
|---|---:|---:|---:|---:|---:|---:|---:|
| kernel (`-fa on`, final branch) vs reference | 0.1366 +- 0.0092 | 0.0055 | 0.207 / 0.623 / 2.77 | 11.4 | 8.5% | 90.49 +- 0.46% | 1.0074 +- 0.0107 |
| dense chain (`-fa off`) vs reference | 0.1506 +- 0.0106 | 0.0059 | 0.237 / 0.632 / 2.75 | 11.2 | 9.1% | 90.35 +- 0.46% | 1.0480 +- 0.0122 |

Both GPU attention paths differ from the CPU banded op by the same, large amount: median KLD 0.006 but a heavy tail (10%
of tokens above 0.2, 1% above 2.7) and only 90% top-1 agreement, i.e. far more than the f16 rounding of one op can
explain (Q8_0-vs-f16 weight quantisation gives ~0.002-0.005 mean KLD and 98-99% top-1 in this tool). The kernel is
slightly closer to the reference than the dense chain on every statistic. Since the two independent GPU implementations
are as far from each other as from the CPU, the next experiment localised the source.

**Per-layer tensor dump on a real prompt** (`llama-eval-callback`, p_short 226 tokens, one ubatch, the three paths;
`review/e2e/evalcb-*.log`, table `review/e2e/evalcb-growth.txt`, script `review/a1_evalcb_compare.py`; the relative
difference is max |a - b| / max |b| over the 36 values the tool prints per tensor):

| layer | attention output `inkling_attn_o-N`: kernel vs CPU op | dense vs CPU op | kernel vs dense | residual `l_out-N`: kernel vs CPU op |
|---:|---:|---:|---:|---:|
| 0 | 1.7e-4 | 4.2e-4 | 3.4e-4 | 2.1e-4 |
| 1-5 | 3.4e-4 .. 8.5e-4 | 4.8e-4 .. 1.5e-3 | 3.2e-4 .. 1.2e-3 | 6e-5 .. 4.8e-4 |
| 8 / 11 | 5.0e-3 / 2.5e-3 | 2.2e-3 / 2.4e-3 | 4.3e-3 / 1.8e-3 | 7.3e-4 / 3.9e-4 |
| 14 / 17 | 9.4e-3 / 1.5e-2 | 1.2e-2 / 2.1e-2 | 1.0e-2 / 1.4e-2 | 2.0e-3 / 6.3e-3 |
| 20 / 26 | 5.8e-2 / 5.0e-2 | 3.1e-2 / 7.9e-2 | 4.8e-2 / 1.1e-1 | 7.5e-2 / 1.1e-1 |
| 35 / 41 | 1.0e-1 / 8.8e-2 | 2.3e-1 / 1.3e-1 | 1.8e-1 / 9.5e-2 | 1.1e-1 / 3.8e-1 |
| logits `result_output` | 1.4e-2 | 5.9e-2 | 7.0e-2 | |

At layer 0 every upstream op is the same Metal op in all three runs, so the attention inputs are bit-identical
(`inkling_attn_q_norm-0`: 0 difference) and the attention-output difference is the op itself on real data: the kernel
is 1.7e-4 from the CPU op, the dense chain 4.2e-4, the two GPU paths 3.4e-4 from each other, all at the f16 level.
From there the three runs drift apart from *each other* at the same exponential rate (x10 every ~8 layers), which is
the model amplifying attention-precision differences through its MoE routing (near-tie expert selection flips, as R5
noted for other implementation pairs), not an error that one path has and the others do not. The greedy divergences
(4a), the 1-4% PPL offsets (4b) and the KLD tail (4c) are all that amplification measured in different ways; the
kernel is the closest of the three implementations to the CPU op at the op level (4c, layer 0), at the model level
(KLD 0.137 vs 0.151) and in perplexity (+1.1% vs +4.0%).

What this means for the brief's criteria: token-exact greedy equivalence with the reference is not achievable for this
model by any f16 attention implementation (the dense chain, the PR's own Metal path today, fails it the same way), and
the 0.5% PPL band is not met by either GPU path against the CPU op; the kernel does meet it in the sense that matters
(it is within the arithmetic noise of the alternatives, and closer than the path users run today).



### 4d. Stability (`-c 70000`, 64 new tokens, the kernel)

| prompt | tokens | prefill | decode at that depth | first sentence | rest |
|---|---:|---:|---:|---|---|
| p_32k | 32,794 | 121.6 s = 270 tok/s | 64 tok / 3.53 s = 18.2 tok/s | "The narrator is Ishmael, and the ship is the Pequod." | reproduces the source ("Returning to the Spouter-Inn, I found Queequeg had already left, but had left behind him his heavy grego, or wrap-all, ..."), identical to K1's `-fa on` text |
| p_64k | 65,561 | 279.2 s = 235 tok/s | 64 tok / 6.08 s = 10.5 tok/s | "The narrator is Ishmael, and the Pequod is the ship." | then repeats the question ("In one sentence, who is the narrator of these chapters and who is the ...") twice; a repetition pattern like the reference's own p_2k loop, no garbage, no NaN |

No garbage at 32K or 64K; the answer is right at both depths. Peak RSS (`/usr/bin/time -l`): 100.7 GB = 93.8 GiB for
the `-c 70000` server, 97.4 GB = 90.7 GiB for the `-c 8192` server, 98.7-98.8 GB = 92 GiB for llama-bench with either
`-fa` setting, 101.8-101.9 GB = 94.9 GiB for llama-perplexity with either setting: the kernel path and the dense chain
have the same footprint at these context sizes (the dense chain's extra scratch is small at `-ub 512` and `-c 8192`).

### 4e. llama-bench, `-p 512 -n 128 -d 0,4096 -r 3`, final branch

| test | `-fa 1` (kernel) | `-fa 0` (dense chain) |
|---|---:|---:|
| pp512 @ d0 | 244.1 +- 22.0 | 267.3 +- 32.9 |
| tg128 @ d0 | 21.2 +- 2.4 | 22.2 +- 0.2 |
| pp512 @ d4096 | 247.3 +- 15.5 | 308.6 +- 7.7 |
| tg128 @ d4096 | 17.0 +- 1.7 | 22.0 +- 0.04 |

These two runs were one minute apart; the `-fa 1` run had other jobs' CPU jobs start during it (its spreads are
10-60x those of the `-fa 0` run), so this pair says nothing reliable on its own and it contradicts K1's +6% / +28%.
The interleaved A/B is in 4f.

### 4f. Interleaved llama-bench A/B (`-fa 1`, `-fa 0`, `-fa 1`, `-fa 0`, each `-d 0,4096 -r 2`)

Load average 18-32 during these rounds (`review/a1_extra2.log`; two renderer processes at ~100% CPU plus other
agents' CPU jobs), falling monotonically through the four runs, which is visible in the decode numbers:

| round (load avg) | test | `-fa 1` | `-fa 0` |
|---|---|---:|---:|
| 1 (32 / 29) | pp512 @ d0 | 247.8 +- 80.1 | 250.4 +- 66.6 |
| 1 | tg128 @ d0 | 13.5 +- 0.3 | 15.2 +- 2.0 |
| 1 | pp512 @ d4096 | 314.8 +- 2.4 | 304.6 +- 1.4 |
| 1 | tg128 @ d4096 | 12.2 +- 1.1 | 17.5 +- 0.2 |
| 2 (25 / 18) | pp512 @ d0 | 300.9 +- 16.5 | 273.1 +- 4.2 |
| 2 | tg128 @ d0 | 19.9 +- 2.9 | 21.3 +- 1.1 |
| 2 | pp512 @ d4096 | 358.2 +- 2.0 | 335.3 +- 12.8 |
| 2 | tg128 @ d4096 | 15.2 +- 1.1 | 21.7 +- 0.1 |

Prefill: the kernel is +3% / +7% at d4096 in the two rounds (and +10% at d0 in round 2), consistent with K1's +6%
and with the op-level picture (the attention is a small part of a 512-token ubatch; the gain is the removed dense
scratch traffic). Decode: every pair I have (4e and both rounds here) has `-fa 1` *slower* than `-fa 0` at d4096
(12.2 vs 17.5, 15.2 vs 21.7, 17.0 vs 22.0 tok/s) and slightly slower at d0, the opposite of K1's +28-30%. The
attention op cannot account for a 5-6 ms per-token gap at 4K context (7 global layers x ~0.1 ms, section 3), so if it
is real it is CPU-side or scheduling, and the load during my window (which halved between round 1 and round 2 and
moves decode by 50%) makes the sign uncertain; the server timings in a quieter moment gave 17.1 tok/s at 226 and 2K
depth and 18.2 at 32.8K for the kernel. K1's decode claim is therefore NOT CONFIRMED, and the decode-slope item is
open in both directions: a quiet-machine interleaved A/B at d0/4096/16384/32768 is required before any decode number
goes into a PR.





## 5. Benchmark skepticism (K1's numbers re-examined, my re-runs in section 4)

- **The end-to-end numbers in `PR.md` are not from the final branch.** Both llama-bench tables carry
  `build: 4ddc6ed6d (11985)` (`bench/e2e/llama-bench-fa-{on,off}.md`), i.e. the tests commit: before the `q_offset`
  commit (`97ba1bf48`, which changes the model's K/V views and the kernel family taken at every unaligned n_kv) and
  before the single-pointer kernel commit (`90290efad`). The greedy runs (`bench/e2e/greedy.log`, 03:01Z-03:17Z =
  23:01-23:17 local) also predate the final commit (23:21). Only the op-level `test-4.log` was run on the final
  sources. The PR text presents the e2e table as "`-fa on` (this kernel)" without this caveat.
- **Baseline handicap check.** The `-fa off` runs used identical flags (`-ngl 99 -p 512 -n 128 -d ... -r 2`, default
  `-ub 512`, 6 threads) and the same loaded machine, so the dense chain was not handicapped by flags. The one
  asymmetry is the opposite: `pp512 @ d0 = 218 +- 14` for `-fa on` is annotated "first touch of the mmap" in
  `PR.md`, i.e. the `-fa on` run paid the cold-page cost in its first test and `-fa off` (run second, model pages warm)
  did not; `-fa off` at d0 also shows `+- 38`, so both d0 rows are noise-dominated. llama-bench's warm-up is one
  run per test, which does not pull 91 GiB through the page cache.
- **Spread.** K1's `-r 2` gives no usable spread; the reported `+-` values (`+- 2.2` on `tg128 @ d16384`, `+- 1.6` on
  the dense side) are of the same size as the claimed decode gains at depth. The "decode slope" (19.5 -> 12.8 tok/s)
  is read off single-sample points taken minutes apart under a load average of 15-19 from other jobs; the
  `-fa off` side loses a similar fraction (15.1 -> 12.7 at d16384), so whatever the slope is, it is not specific to
  this kernel (the dense chain has a different attention cost profile) and it is confounded by machine load.
- **Op-level perf.** The two runs K1 quotes disagree with each other by up to 1.75x on the plain op
  (`512 x 32768`: 27.9 ms in `bench/perf-mtl0-fa.log`, 48.7 ms in `bench/e2e/perf-quiet-fa.log`) and the ratio
  banded/plain moves from 1.38x to 0.99x at that shape and from 0.85x to 0.73x at `512 x 2048` between the runs.
  Neither run had a quiet GPU (K1 says so). The table's "1.4-1.6x" is therefore a loose statement; my interleaved
  min-of-3 is in section 3.
- **The `-fa off` row at d32768 is missing** (lock budget), so the headline "340 tok/s at 32K vs 15.6 reported" compares
  the kernel to the field's CPU-fallback number, not to the dense chain a Mac user actually gets with `-fa auto`.
  R2 predicts the dense chain at ~300-400 tok/s at 32K; the honest comparison at 32K is roughly a wash on prefill
  and a decode win, not a 20x.
- **The greedy "equivalence" compares the kernel to the dense Metal chain, not to the reference.** Both are
  approximations of the CPU banded op; a divergence between them says nothing about which one is right. K1's
  "every divergence is a near-tie flip" was asserted without the probability readout (the `run_probs.sh` step was
  never run). Section 4 compares against the reference and reads the margins.



## 6. Upstream fit

- The diff is confined to the op (`ggml.h/ggml.c`, CPU, CUDA, Metal, the model's call site, tests); no unrelated edits; ggml-metal conventions are followed (kargs fields, function constants `FC_... + 5/6/4` and `+ 30`, pipeline-name suffixes, `GGML_TENSOR_LOCALS`, the `supports_op` fall-through). The new `test-backend-ops` cases are in the right place and use the existing helper. `tests/test-flash-attn-bias.cpp` (the PR's own standalone test) was updated for the signature; it still only exercises `q_offset = n_kv - n_q`.
- **The `ggml.h` API change cannot ship in a Metal follow-up.** `ggml_flash_attn_ext_banded` does not exist on master; its signature is owned by #25731 (or the CPU-first split #29491). Adding `q_offset` is a change to that PR's op definition across three backends, so it must go to the op's author as a separate suggestion (or be made in that PR) before the Metal kernel can be a self-contained follow-up. The Metal kernel itself does not need `q_offset` for correctness, only for the aligned-view speedup (1.5x at 8K); without the API change the kernel would run on the exact `pos_max + 1` views via the half8x8 + pad path, which my tests also cover (every unaligned case).
- The CUDA MMA ABI now carries `q_offset` in the `m0` slope slot (on top of the PR's existing `sinks`-pointer and negative-`max_bias` smuggling). It is correct by reading but it is the kind of thing the CUDA maintainer will push back on; K1's own open question 3 says as much. It is untested here.
- **CONTRIBUTING.md / AGENTS.md compliance (blocking for submission, not for the code):**
  - `AGENTS.md:50` lists "AI-written PR descriptions, commit messages, or reviewer responses" under *Prohibited AI Usage (results in immediate PR closure)*, and `CONTRIBUTING.md:25` repeats it. `PR.md` is an AI-written PR description and the four commit messages are AI-written and verbose (the `AGENTS.md:186-212` example marks exactly this style as BAD). The submitter must write the PR description and the commit messages himself; `PR.md` can only serve him as notes.
  - `AGENTS.md:96`: when the user asks the agent to commit, use `Assisted-by: <assistant name>`, **not** `Co-authored-by:`. All four commits carry `Co-Authored-By: Claude Fable 5.1`; they need rewording/rewriting (`git rebase -i` by the submitter).
  - The PR template (`.github/pull_request_template.md`) requires the *Requirements* section with "AI usage disclosure: YES/NO - if yes, describe how AI was used". `PR.md` has no such section.
  - `CONTRIBUTING.md:21-24`: explicit disclosure of how AI was used, a comprehensive manual review ("one hour per 200-400 LOC ... at least that much time on code review alone"), and the ability to explain every line. 361 added lines means the submitter owes roughly an hour of his own review before submitting.
  - `CONTRIBUTING.md` "Before you start": one open PR per new contributor; "Features must begin with an issue". The thread of #25731 is where this belongs first.
- `PR.md` accuracy: the design and test sections are accurate. The end-to-end section is **not** from the final branch (section 5) and compares against the wrong baseline (section 4); its "near-tie" classification was unverified at the time of writing.

## 7. Defects found

No correctness defect in the Metal kernels, the host glue, the `q_offset` plumbing or the model-side views was found
by code reading, by 191 op-level cases, or by the real-model dumps. The findings are in the evidence and the package:

| # | severity | finding | repro |
|---|---|---|---|
| D1 | major (evidence) | The end-to-end tables in `PR.md` (llama-bench, greedy) were produced on build 11985 = commit `4ddc6ed6d`, two commits before the branch head; the final kernel and the `q_offset` model change never had an e2e run before packaging | `bench/e2e/llama-bench-fa-{on,off}.md` last line `build: 4ddc6ed6d (11985)`; `bench/e2e/greedy.log` timestamps 03:01-03:17Z vs commit `90290efad` at 03:21Z |
| D2 | major (claim) | "+28-30% decode vs the dense chain" is not reproducible: three interleaved pairs on the final build show `-fa 1` decode 10-30% *slower* at d4096 under load | `review/e2e/abab-fa{1,0}-{1,2}.md`, `review/e2e/llama-bench-fa{1,0}.md` |
| D3 | major (claim) | The greedy "equivalence" compares two approximations (kernel vs dense chain) and classifies divergences as near-ties without measuring them; against the reference they are 0.5-0.7-nat disagreements with mean 0.03-0.06-nat differences on the common prefix | `review/e2e/greedy-kernel-vs-ref.txt` |
| D4 | medium (claim) | PPL `-fa on` vs `-fa off` differ by 2.9% (kernel +1.1% from the reference, dense +4.0%); `PR.md` makes no PPL statement and `llama-perplexity` was not even built | `review/e2e/ppl-*.err` |
| D5 | medium (process) | AI-written PR description and commit messages (prohibited: `AGENTS.md:50`, `CONTRIBUTING.md:25`), `Co-Authored-By` trailers instead of `Assisted-by` (`AGENTS.md:96`), no AI-usage disclosure line (PR template) | `git log pr-25731..metal-banded-fa`, `PR.md` |
| D6 | medium (scope) | The `ggml.h` signature change of a function that only exists in the unmerged #25731 cannot be a Metal follow-up PR; the CUDA MMA part is untested here | diff of `ggml/include/ggml.h`, `ggml/src/ggml-cuda/*` |
| D7 | low (robustness) | The decode-slope item is confounded by machine load; the 19.5 -> 12.8 slope quoted is from two-sample points minutes apart at load 15-19; my server timings show 17.1 -> 18.2 tok/s from 2K to 32.8K and 10.5 at 65.5K | `review/e2e/greedy-k1-faon*.txt` |
| D8 | low (tests) | `test-backend-ops` tolerance (5e-4 / 1e-3) is 4 orders of magnitude above the kernels' actual error; a wrong block in a 16-block band would pass; the helper cannot express a `q_offset` other than the tail, a window independent of E, a batch > 1 or the sparse hint | `tests/test-backend-ops.cpp` (K1) vs `review/a1-tests.patch` |



## 8. Required before this could be submitted

1. Re-run every end-to-end measurement on the branch head (or on whatever is submitted) and replace the `PR.md` tables:
   llama-bench `-fa 1` vs `-fa 0` as an interleaved A/B with `-r 3` or more on a quiet machine (no other jobs, no
   renderer at 100%), at d0 / 4096 / 16384 / 32768, including the `-fa 0` row at 32768; report the spread. Drop or
   invert the decode claim according to what that A/B says, and state the decode slope only from that data.
2. Replace the greedy section by what the brief actually asks for: equivalence against the CPU banded reference
   (pristine `-fa on`) with the probability readout, plus the perplexity and KLD numbers (sections 4b/4c), with the
   layer-0 dump as the explanation of why token-exact agreement is not attainable; do not call divergences near-ties.
3. Split the submission: (a) the `q_offset` op change (ggml.h/ggml.c, CPU, CUDA scalar + MMA, model call site, test
   helper) goes to the #25731 author / the CPU-first split as a suggestion, with the CUDA part compiled and run on an
   NVIDIA machine (`test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT_BANDED`); (b) the Metal kernel + tests as the follow-up
   once the op (with or without `q_offset`) is on master. If `q_offset` is rejected, the Metal kernel still works on the
   exact views (every unaligned case is covered) at ~1.5x the prefill attention time at 8K.
4. The submitter writes the PR description (with the "AI usage disclosure: YES, ..." line from the template) and rewrites the four
   commit messages (concise, `Assisted-by: Claude Fable 5.1`), after the manual review CONTRIBUTING.md asks for (~1 hour
   for +361 lines). `PR.md` becomes his notes, not the body.
5. Keep the added `test-backend-ops` cases but add at least: an explicit non-tail `q_offset` case, a window-independent
   -of-E case, a batch > 1 case, and tighten `max_nmse_err` for f16 KV to ~1e-6 (`review/a1-tests.patch` has all of
   these; the sparse-hint and 1M-key cases are optional).
6. Optional but cheap: the half8x8 bias gather could read `half2`/`float2` pairs for the two adjacent keys of a lane
   (dist d0 and d0-1 are adjacent table entries) instead of two scalar gathers; not required.



## 9. Raw logs (`package/k1-metal-banded-fa/review/`)

- `build-a1*.log`, `build-evalcb.log`: my builds. Worktrees left in place for inspection: `upstream/llama.cpp-a1` (branch `a1-review`, three test commits `845ec0484`, `114a8aae1`, `ad0825f1c` on top of K1's head; build dir `build-a1`) and `upstream/llama.cpp-a1-base` (detached at the pristine `3582c5ee0`; `build-a1-base`, `llama-perplexity` and `llama-eval-callback` only). K1's clone and `build/` were not modified.
- `test-banded-k1cases-a1build.log`: K1's 26 cases on my build. `test-banded-a1cases.log`: the 171-case run (K1's 26 + my 145), `test-banded-a1cases-P.log` (8, the model's exact shapes), `test-banded-a1cases-Q.log` (12, realistic magnitudes; `groupQ-summary.txt`), each case with its NMSE / max-abs line (`[A1 nmse=... max_abs=...]`).
- `test-plain-fa-a1build.log`: the plain `FLASH_ATTN_EXT` suite on my build (regression check of the shared kernels).
- `test-flash-attn-bias-mtl0.log`: the PR's standalone test run on MTL0 (`small`, `medium`, `overflow` suites).
- `a1-tests.patch`: my test-backend-ops changes (the helper's new parameters and the 145 cases).
- `e2e/`: `perf-{banded,plain}-{1,2,3}.log`, `greedy-*.json/.txt` + `greedy-kernel-vs-ref.txt`, `server-*.err`, `ppl-*.{log,err}`, `kld-*.{log,err}`, `llama-bench-fa{1,0}.md/.err`, `abab-fa{1,0}-{1,2}.md/.err`, `evalcb-*.log` + `evalcb-growth.txt`. Scripts: `a1_e2e.sh` (the window; its timeline in `a1_e2e.log`), `a1_extra.sh` / `a1_extra2.sh` / `a1_extra3.sh` (the three chained extensions: KLD, interleaved bench, eval-callback dumps), `a1_greedy.py` / `a1_compare.py` (greedy driver and divergence analysis), `a1_evalcb_compare.py`. The 1.6 GB reference log-prob file was deleted after use.
