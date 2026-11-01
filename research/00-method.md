# Method: how candidates are ranked and proven

## 1. Opportunity scoring (0-5 each, weighted)

| Criterion | Weight | What earns a 5 |
|---|---|---|
| Speed impact on a Mac Studio | 3 | >=2x on a metric users feel (TTFT at 8K-64K context, decode tok/s at depth, or "fits and runs at all") with the arithmetic shown and a measurement behind it |
| Users reached | 2 | lands in the engine Mac users actually run (llama.cpp/LM Studio/Ollama, mlx-lm/LM Studio MLX, mlx-vlm) rather than a fork |
| Upstream acceptability | 2 | matches the project's stated policy and PR shape; tests the project expects; no maintainer objection on record |
| Deliverability in this campaign | 2 | can be built, tested against an oracle, and benchmarked on this M5 Max; bounded risk |
| Provable correctness | 1 | an executable oracle exists (research/oracle) and the change is bit-for-bit or tolerance-checkable |

Score = sum(weight x points). Ties break toward the smaller, more reviewable change.

## 2. Evidence rules

- A speed claim needs a before/after measurement on the same machine, same quant, same prompt, same tokens, warm (first run discarded), median of >=3, with the raw logs under bench/.
- Like for like only: never compare a 2-bit MLX run against a 3-bit GGUF run as if it were an engine comparison.
- Extrapolations from synthetic per-layer benchmarks are labelled "predicted" until the full model confirms them.
- Every statement about code cites file:line in the clone under upstream/.

## 3. Adversarial evaluation protocol (per implemented candidate)

An reviewer agent receives the branch and the oracle, not the author's conclusions, and must try to break it:
1. Numeric equivalence vs the oracle at random shapes, including: context shorter than the window, contexts crossing rel_extent (1024) and multiples of the kernel block size, prompt lengths not a multiple of the chunk/ubatch size, batch > 1 where supported, quantized KV cache, GQA head mapping, the global vs local layer types, log-scaling positions past n_floor.
2. End-to-end: for a change designed to be bit-exact, identical greedy tokens AND identical per-token logprobs before vs after. For any change that alters accumulation order or precision (a new kernel, a different chunking, f16 vs f32 paths), token-exact agreement is NOT a valid criterion on Inkling-Small: A1 showed (package/k1-metal-banded-fa/REVIEW.md section 4) that per-layer differences of ~1e-4 at layer 0 are amplified ~10x every 8 layers through MoE routing, so any two f16 paths diverge in greedy tokens with 0.5-0.7-nat disagreements at the divergence step. Use instead: teacher-forced per-token logprob comparison on the same tokens (mean/max abs difference, KL divergence per token, top-1 agreement) against the reference path, compared with the noise floor measured between two runs of the unmodified implementation under a different chunk/ubatch size; plus perplexity on the 8K text. A change passes when its KL/PPL gap to the reference is no worse than the implementation it replaces (and the noise floor), for short, 8K and 32K prompts; multi-turn with prefix cache judged the same way.
3. Stability: a 48K+ prompt and a 2K-token generation finish without garbage; memory does not grow unbounded; repeated runs are deterministic.
4. Benchmark skepticism: re-run the author's benchmark, check the warmup, the token counts, the variance, and whether the baseline was handicapped.
5. Upstream fit: run the project's own test suite and style checks; read CONTRIBUTING; compare with how the last similar PR was reviewed.

The reviewer's verdict (CONFIRMED / PLAUSIBLE / BROKEN with evidence) goes into package/<n>/REVIEW.md. A BROKEN verdict sends the candidate back to its author.

## 4. Package contents (per contribution)

package/<n>-<slug>/
- README.md: what, why, measured gain, how to apply, review status
- PR.md: the pull-request title and body as it would be submitted (target repo and branch named)
- <slug>.patch: `git format-patch` output from the branch in upstream/<repo>
- bench/: before/after scripts, raw logs, summary table
- tests/: the tests added (also present in the patch)
- REVIEW.md: the adversarial reviewer's report

Nothing was pushed, filed or uploaded during the campaign; each item waits for the maintainer to approve it specifically.
