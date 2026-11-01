# Pull request material for ggml-org/llama.cpp

> **Read before posting.** llama.cpp's CONTRIBUTING.md and AGENTS.md prohibit AI-written PR descriptions and commit
> messages, require an AI-usage disclosure, a reproducible issue plus a regression test for bug fixes, and ask new
> contributors to keep one open PR and to comment on an existing PR that addresses the same change instead of opening
> a duplicate. The code, the test and the commit message on `server-hybrid-prompt-cache` were written by Claude
> under human direction, so before submission:
>
> 1. rewrite the commit message (`git commit --amend`) and the PR description in your own words - the text below is
>    a fact sheet and a suggested structure, not a description to paste. Trailers: AGENTS.md requires
>    `Assisted-by: <assistant>` for AI assistance and forbids `Co-authored-by:` for the assistant; the commit
>    `9b8b6bd93` carries `Assisted-by: Claude Fable 5.1 <noreply@anthropic.com>` and, as human credit for the fork
>    commit the fix is ported from, `Co-authored-by: fezzik <fezzik@fezzik.dev>` (keep that one; the project allows
>    crediting a human author in the body or trailer);
> 2. review every line (`patches/0001-*.patch`, 3 files, ~130 lines; the server change is 7 lines) and be ready to
>    explain the gate arithmetic to a reviewer without help (A3's adversarial review, `REVIEW.md`, found no defect:
>    22 server cases pass on this build and on master + patch, dense and SWA-only models unchanged);
> 3. decide between a new PR and a comment on #24035 (Regrad, open since 2026-06-02, no reviews, 6 comments): it
>    reworks checkpoint selection for exact-state (FULL/RS) memories and includes "direct reuse when tokens are
>    appended to an exact full state", so it covers this bug within a much larger change; this fix is the minimal
>    gate correction that also unblocks the existing checkpoint path. A comment on #24035 offering it as the small
>    first step is the policy-compliant move; a standalone PR should link #24035 and say why the smaller change
>    is preferable (one condition, no new selection logic).

## Suggested title

    server : do not apply the SWA back-off to a recurrent-tail pos_min (prompt cache for hybrid models with SWA)

## Facts for the description

### Problem

For a model whose memory is recurrent or hybrid *and* has sliding-window attention (Inkling in #25731: 35 SWA
layers with window 512 plus a shortconv state on every layer; also any `llama_memory_hybrid_iswa` model such as
LFM2 with `attention.sliding_window` set), `llama-server` re-processes the whole prompt on every request, including
a request that merely extends the cached prompt:

    slot update_slots: ... forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)

Field report in #25731 (wowo101, gfx1151, `-ub 256`): `/completion` with A = 4000 tokens then A + 1500 tokens with
`cache_prompt: true` took 39.8 s and reported `prompt_n = 5500`; chat turn 2 re-processed 3925 tokens; a Qwen3.5
control on the same build reused the prefix. F3zz1k fixed it on his fork
(https://github.com/F3zz1k/llama.cpp/commit/371c74a0553663d9d6e28b4ebecac9542f076c19); brywil measured cached turns
~5x faster with it on a GB10.

### Cause

`llama_memory_hybrid[_iswa]::seq_pos_min()` returns `max(attn_min, recr_min)`, and the recurrent memory keeps a
single live cell per sequence whose `pos` is the last decoded position. The `pos_min` the server reads for such a
memory is therefore a tail, not the oldest position the state still covers, and so is the `pos_min` recorded in its
context checkpoints (`pos_min == pos_max == n_tokens - 1` in the logs).

In the reuse gate of `update_slots()` a state or checkpoint is useful when `pos_min < pos_min_thold`, with
`pos_min_thold = pos_eval - n_swa + 2` for a model with a sliding window (a SWA state starting at
`pos_eval - n_swa + 1` is enough for the first evaluated token). That is right for a window start and wrong for a
tail: a tail is usable exactly when it ends right before `pos_eval`. With the back-off, `pos_min >= pos_min_thold`
holds on every request (strict extension: `pos_next - 1 >= pos_next - n_swa + 2` for any `n_swa >= 3`), and the
checkpoint search then needs a checkpoint whose tail lies more than `n_swa - 2` tokens before `pos_eval`; with the
default checkpoint placement (`n_ubatch + 4` and `4` tokens before the end of the prompt) that exists only when
`n_swa < n_ubatch + 7` (Inkling: `n_swa 512`, so at `-ub 512` the server falls back to the checkpoint 516 tokens
back and re-evaluates 516 + the new tokens; at `-ub 256` and below `n_past` is reset to 0).

### Fix (tools/server/server-context.cpp, 7 lines)

Apply the `n_swa` back-off only when `pos_min` is a genuine window start, i.e. not when the target context's
seq_rm type is FULL or RS (`pos_min_is_tail`). Strict extensions then pass the gate and reuse the state as-is; an
identical prompt or a rewind still fails it and restores the newest checkpoint before the first changed token, as a
hybrid model without SWA already does; `n_swa == 0` models are unchanged; PART models keep the back-off. The
predicate rests on FULL/RS arising only from a recurrent memory today (`llama_memory_recurrent::seq_rm` is the only
one that refuses a partial range; `llama_kv_cache` and `llama_kv_cache_iswa` never do); a future non-recurrent
FULL memory would need its own case. Say that in the PR.

Ported from the fork commit above (the gate has since moved to `pos_eval` / `pos_min_thold`); credited with
`Co-authored-by: fezzik <fezzik@fezzik.dev>`.

### Regression test (tools/server/tests)

`test_completion_cache_hybrid_memory[n_swa]` on `ggml-org/LFM2-test-ci-80M` (new `ServerPreset.lfm2_hybrid`,
50 MB Q4_K_M, 3 shortconv + 1 attention layer, FULL seq_rm), `n_ubatch 32`, parametrized `n_swa = 0` (as-is) and
`n_swa = 64` (`--override-kv lfm2.attention.sliding_window=int:64`, which makes the memory `llama_memory_hybrid_iswa`;
`64 > n_ubatch + 4` so the unfixed server cannot fall back to an older checkpoint). Cached requests on slot 0,
uncached references on slot 1; asserts `(cache_n, prompt_n)` and the top-5 logprobs of the first predicted token
(within 5e-2: the cached request evaluates fewer ubatches than the reference) for: prime `(0, n_base)`; strict
extension `(n_base, n_ext)`; identical prompt `(n_full - 4, 4)`; rewind within the last 4 tokens `(n_full - 4, 4)`;
rewind into the extension `(n_base, 2 + n_alt)`; rewind before every checkpoint `(0, n)`.

Output check: same greedy token, and the logprobs of the candidates both top-5 lists share within 0.2 nats (the
cached request evaluates its suffix in fewer ubatches than the reference; 1-2 token suffixes of this Q4_K_M model
deviate by up to 0.19 nats on Metal, baseline included). Results of the committed test version: see README.md
("The test"), with the pass counts on the fixed and the baseline binaries.

### Measurements (Inkling-Small UD-IQ3_XXS, Metal, M5 Max 128 GB, #25731 at 3582c5ee0 + this patch)

Server: `-np 1 --no-kv-unified -ngl 99 -fa 0 -c 70000 --jinja --temp 0 -ub {512,256}`, greedy, `n_predict 64`.
`-ub 256` is the field's setting; with `-ub 512` the unfixed server can still fall back to the checkpoint 516
tokens before the end of the prompt because `516 > n_swa - 2 = 510`, so the two columns differ before the fix.
The fixed binary measured was built from `7251878de`, server code byte-identical to the submitted commit; the
baseline from `3582c5ee0`. The runs were unequally loaded (`load1` 6.5 before, 17.9 after, 26.7 before `-ub 256`:
other jobs on the machine), so quote the token counts (deterministic, reproduced independently by A3 on the same
model) and treat the times as indicative (+-15%: the uncached 8218-token reference took 29.9 s in the before run,
34.5 s in the after run).

| request (`timings.prompt_n` evaluated, then `prompt_ms`) | before `-ub 512` | after `-ub 512` | before `-ub 256` |
|---|---|---|---|
| p_8k (8218 tok), no cache | 8218 tok, 29.9 s | 8218 tok, 34.5 s | - |
| p_8k + 200 tok, cache on, after a p_8k request | 716 tok, 3.60 s | 204 tok, 1.43 s | 8418 tok, 42.3 s |
| p_8k + its 63 generated tok + 200, cache on (strict extension) | 779 tok, 4.27 s | 200 tok, 1.42 s | 8481 tok, 35.4 s |
| p_8k with the last 100 tok replaced, cache on (rewind) | 8218 tok, 30.3 s | 253 tok, 1.90 s | 8218 tok, 35.1 s |
| chat turn 2 / 3 / 4 after a 2085-tok turn 1 | 550 / 592 / 574 tok, 3.1-3.5 s | 18 / 23 / 16 tok, 0.8 s | 2110 / 2163 / 2217 tok, 8.4-8.7 s |

(Prompt evaluation ran at 3.5-4 ms per token at this depth, 5-7 ms per token for the short suffixes.) Before the fix at `-ub 512` the server falls back to the checkpoint 516
tokens before the end of the previous prompt (516 > n_swa - 2); at `-ub 256`, the field's setting, it logs
`forcing full prompt re-processing` on every cached request. After the fix a strict extension reuses the live
state (`cache_n = 8281`), the other cases restore the checkpoint right before the first changed token. The rewind's
64 greedy tokens are identical with and without the cache; the extension cases diverge at a near-tie (same top
candidates within 0.3 nats, both continuations coherent) exactly as they do before the fix through its checkpoint
path - the batched-vs-incremental drift `test_cache_vs_nocache_prompt` is skipped for.

The patch is independent of #25731 (it applies to master, where the gate is identical); the Inkling numbers only
need #25731 for the model.

### Related

- #13194 (the SWA/hybrid full re-processing note the log points at), #21831, discussion #19264
- #24035 (Regrad): checkpoint selection rework for recurrent/hybrid prompts (open, unreviewed) - supersedes this gate
  change within a larger diff
- #24797 (AntmanLFEz, closed 2026-06-27): "server: fix checkpoint handling for hybrid memory models" - the same
  diagnosis (the recurrent `pos_min` is the current position, `max()` conflates the two semantics) addressed from
  the library side by changing `llama_memory_hybrid::seq_pos_min` and the recurrent memory, plus server changes;
  a reviewer will ask how this differs: this change keeps the memory API as is and corrects the server's use of it
- #22826 (jacekpoplawski, closed 2026-07-15): "server: preserve context checkpoint coverage" - a checkpoint
  eviction policy change motivated by the same log line
- #24785 (Moltes94): recurrent state shrink/expand around prompt cache save/load (open) - different mechanism
- #22932 (leon7609): checkpoint matching by n_tokens for FULL memories (closed)
- #24003 / #24004 (F3zz1k): disk restore reuse for recurrent/hybrid models (closed)

## Requirements section (fill in yourself)

- I have read and agree with the contributing guidelines
- AI usage disclosure: YES - state how (the code and the test were produced with Claude under your direction, the
  fix's design is F3zz1k's fork commit, you reviewed every line and ran the tests and the measurements)
