# Ranking of candidate contributions (final, 2026-10-29 03:40Z)

Scored with research/00-method.md section 1 (speed x3, users x2, upstream x2, deliverable x2, provable x1).

| # | Candidate | Target | Speed | Users | Upstream | Deliv. | Prov. | Score | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| A | Metal banded flash-attention kernel (+ explicit q_offset op param) | llama.cpp (ggml-metal) | 5 | 5 | 3 | 4 | 5 | 44 | R2 s5-7: op runs on the CPU scalar path on Metal; kernel ceiling 7-8x over the dense path, 200-500x over `-fa on`; removes GiB scratch; 13 test cases exist |
| B | Server prompt-cache reuse for hybrid+recurrent memory (sconv state) | llama.cpp (tools/server) | 4 | 5 | 4 | 5 | 3 | 43 | R2 s8a: every request re-prefills the whole prompt; field: cached turns 5x faster with the fork fix |
| C | First-class Inkling model (op-count design, bounded SWA caches, 8-bit attention predicate, server/prompt-cache for free) | mlx-lm | 4 | 5 | 4 | 3 | 4 | 40 | R3 s4-7: decode dispatch-bound (~4,200 ops/token); bf16 attention+unembed are 66% of bytes/token; mlx_lm.server snippets on the cards cannot work today |
| D | Inkling fast-path + robustness set (bounded SWA caches, 10 fewer kernels/layer, tau skip, logits_to_keep, kv_bits fix, fp32 global_scale fix, chunked loader) | mlx-vlm | 3 | 5 | 4 | 5 | 4 | 42 | R1 s"Opportunities" 2,3,5,9 and hazards H1,H2,H6; loader crash reproduced here (bench/results/smoke-mlx-2bit.log) |
| E | Relative-bias table inside fused SDPA kernels | MLX core | 3 | 3 | 2 | 2 | 4 | 28 | R3 s7 item 3, R1 item 6: 5-10% prefill, removes 1-4 GB masks; needs maintainer buy-in across 3 kernel families |
| F | MTP drafter rework (trim-based rollback, no replay) | mlx-vlm / mlx-lm | 3 | 3 | 2 | 2 | 2 | 24 | R1 item 4 (1.4x if acceptance >= 60%), but R5 s4: three different MTP semantics in the wild, mlx-vlm's unvalidated; PR #2206 in flight; no Mac checkpoint ships the drafter |
| G | 8-bit attention/unembed re-quantized checkpoints + card fixes | HF mlx-community | 4 | 4 | 3 | 2 | 3 | 33 | R1 item 1 (+22% measured on synthetic), R3 s3.1; needs a conversion run; code part lands inside C |

## R4 facts that settle the reach column
- GGUF downloads of Inkling dwarf MLX downloads (~1.0M vs ~2.2k across 41 MLX repos), so the llama.cpp items reach the most users even though the architecture PR is still a draft (R4 s"Engines").
- mlx-vlm is the only MLX implementation and is vendored by oMLX (23k stars) and pinned (v0.7.3) by LM Studio's MLX engine, so mlx-vlm fixes propagate beyond the repo (R4 s"Engines"); D's reach score raised to 5.
- No one has upstreamed the fork's prompt-cache fix; two closed server PRs (#24003/#24004) touched disk-restored KV for recurrent/hybrid models (K3 checks them).
- oMLX reports 36-39 tok/s flat to 32K on an M3 Ultra and 51 tok/s with an 8-depth MTP; mlx-vlm's own MTP path is slower than plain decode (R4 s"Measured Apple numbers"), consistent with R1 item 4 and R5's semantics finding.

## Decision (three opportunities, one per Mac engine; four PRs)

1. **llama.cpp on Metal**: A (kernel) + B (prompt-cache reuse) as two PRs, plus a note for the PR thread that `-fa off` is 31-61x faster than `-fa on` on Metal today. K1 builds A; K3 builds B after R4 confirms nobody is upstreaming the fork fix.
2. **mlx-lm**: C, with the 8-bit attention recipe as its quant_predicate (the code half of G). K2.
3. **mlx-vlm**: D as one PR (loader change possibly split out). K4.

Not pursued now: E (follow-up once C/D land and MLX's new SDPA window argument ships), F (semantics unresolved; PR #2206 in flight), G's artifact half (needs the 532 GB source or a local re-quantization run; K2 produces a local 8-bit-attention variant for measurement only).

## Acceptance for each package (research/00-method.md section 3)
Independent adversarial review with the oracle (research/oracle/inkling_ref.py, MLX_ENABLE_TF32=0 for fp32 checks), greedy-token equivalence on real weights before/after, long-context stability, like-for-like benchmarks with raw logs, the project's own test suite and style checks.
