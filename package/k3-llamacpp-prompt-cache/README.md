# K3: llama-server prompt-cache reuse for hybrid models with a sliding window (Inkling)

Branch `server-hybrid-prompt-cache` in `~/code/inkling-contrib/upstream/llama.cpp-k3` (a git worktree of
`upstream/llama.cpp-inkling`), one commit on top of `pr-25731` at `3582c5ee0`:

    9b8b6bd93 server : do not apply the SWA back-off to a recurrent-tail pos_min

`patches/0001-*.patch` is its `git format-patch`; it applies cleanly to upstream `master` (`2e0d86beb`,
2026-10-28), because the server code it touches is the same there. Nothing was pushed. Commit history: `7251878de`
(first version, measured on the model), `5edd1f152` (comment shortened, reviewed by A3), `9b8b6bd93` (commit
message with the `Assisted-by:` trailer, test comparison loosened); the server code is byte-identical in all three.

Review: `REVIEW.md` (A3's adversarial review of `5edd1f152`): no code defect, 22 adversarial server cases pass on
this build and on master + patch, the real-model tuples reproduced exactly; its required changes are applied here
(trailer, provenance, test tolerance, the `--cache-reuse` statement, benchmark presentation).

## What

`llama-server` re-processed the whole prompt on every request for Inkling (and any other model whose memory is
recurrent/hybrid *and* uses a sliding window), even when the request only extended the cached prompt. Multi-turn
chats and agent loops re-prefilled their entire context each turn (the field report in PR #25731: 27 s per turn at
4K tokens on gfx1151; brywil: the fork fix made cached turns ~5x faster).

The fix is one condition in the prompt-cache reuse gate of `tools/server/server-context.cpp` (+11 lines) plus a
server test (`tools/server/tests/unit/test_completion.py::test_completion_cache_hybrid_memory`, with a new
`lfm2_hybrid` preset in `tools/server/tests/utils.py`). It is backend-independent and model-independent.

## Why (the mechanism)

- `llama_memory_hybrid[_iswa]::seq_pos_min()` returns `max(attn_min, recr_min)`
  (`src/llama-memory-hybrid-iswa.cpp:178`). The recurrent side keeps one live cell per sequence whose `pos` is the
  last decoded position (`src/llama-memory-recurrent.cpp:391`), so the `pos_min` the server sees for such a memory is
  a **tail**, not the oldest position the state still covers. A checkpoint of such a memory records a tail too
  (`created context checkpoint (pos_min = 11636, pos_max = 11636, n_tokens = 11637)` in the logs).
- The server's reuse gate (`server-context.cpp` around line 4105): a state or checkpoint is useful when
  `pos_min < pos_min_thold`, with `pos_min_thold = pos_eval - n_swa + 2` when the model has a sliding window (the
  first evaluated token only attends to the last `n_swa` positions), else `pos_eval`. `pos_eval` is the position of
  the first token to evaluate (`pos_next`, or `pos_next - 1` when the prompt is identical to the cache).
- For a tail the back-off is wrong: a tail is usable exactly when it ends right before `pos_eval`, i.e. `tail ==
  pos_eval - 1`. Backing the threshold off by `n_swa` makes `pos_min >= pos_min_thold` true on every request
  (including a strict extension, where `tail = pos_next - 1 >= pos_next - n_swa + 2` for any `n_swa >= 3`), and the
  checkpoint search then needs a checkpoint whose recorded tail lies more than `n_swa - 2` tokens before `pos_eval`.
  With the default checkpoint placement (`n_ubatch + 4` and `4` tokens before the end of the prompt, plus user-message
  starts) that never exists for Inkling (`n_swa = 512`, `-ub 512`: the checkpoint 516 tokens back is accepted, see the
  measurements; with the field's `-ub 256` nothing is), so `do_reset` fires: "forcing full prompt re-processing due to
  lack of cache data (likely due to SWA or hybrid/recurrent memory, see #13194)" and `n_past = 0`.
- Inkling is `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`: it is not in `llm_arch_supports_rs_rollback` (`src/llama-arch.cpp:1321`),
  and `common_context_can_seq_rm` (`common/common.cpp:1596`) finds that a partial `seq_rm` fails
  (`llama-memory-hybrid-iswa.cpp:149` -> `llama-memory-recurrent.cpp:186`). The server already creates context
  checkpoints for FULL/RS memories (`server-context.cpp:4306`), so rewinds work once the gate is right.

The fix: apply the `n_swa` back-off only when `pos_min` is a genuine window start, i.e. not for seq_rm types
FULL and RS (`pos_min_is_tail`). Then

- a strict extension of the cached sequence passes the gate (`tail = pos_next - 1 < pos_next`) and reuses the
  state as-is: only the new tokens are evaluated;
- an identical prompt or a rewind (the slot holds the previous answer, the user edited the prompt, ...) fails the gate
  as before and restores the newest checkpoint before the first changed token, exactly like a hybrid model without
  SWA does today;
- models with `n_swa == 0` are unaffected (the back-off was already a no-op), PART models (plain SWA attention)
  keep the back-off.

This is F3zz1k's fix from the fork commit `371c74a0553663d9d6e28b4ebecac9542f076c19` (https://github.com/F3zz1k/llama.cpp/commit/371c74a0553663d9d6e28b4ebecac9542f076c19), ported to the current gate (`pos_eval` / `pos_min_thold` have moved since)
and credited as `Co-authored-by`. No upstream PR carries it: `gh search prs --author F3zz1k` finds only his closed
disk-restore PRs (#24003, #24004); the open #24035 (Regrad, "preserve reusable checkpoints for recurrent / hybrid
prompts", +5 files) reworks checkpoint selection for exact-state memories and would subsume the gate, but it has
been open since June without review; #24785 (recurrent shrink/expand) is a different mechanism; #22932 (checkpoint
matching by `n_tokens` for FULL memories) was closed. The PR text (`PR.md`) references them.

## The test

`test_completion_cache_hybrid_memory[n_swa]` runs `ggml-org/LFM2-test-ci-80M` (50 MB, 4 layers: 3 shortconv +
1 attention, hybrid memory, FULL seq_rm) with `n_ubatch = 32`, twice: as-is (`n_swa = 0`, the regression guard) and
with a sliding window forced through `--override-kv lfm2.attention.sliding_window=int:64` (the model then takes the
`llama_memory_hybrid_iswa` path, `n_swa = 64 > n_ubatch + 4`, so the unfixed server cannot fall back to an older
checkpoint). The cached requests use slot 0, the uncached references slot 1. It checks, with `cache_n` / `prompt_n`
from `timings` and the top-5 logprobs of the first predicted token against the uncached reference:

1. prime: `(cache_n, prompt_n) == (0, n_base)`;
2. strict extension: `(n_base, n_ext)` - the whole cached prompt is reused;
3. identical prompt: `(n_full - 4, 4)` - restored from the checkpoint 4 tokens before the end;
4. rewind within the last 4 tokens: `(n_full - 4, 4)`;
5. rewind into the extension: `(n_base, 2 + n_alt)` - restored from the checkpoint taken before the extension;
6. rewind before every checkpoint: `(0, n)` - full re-processing, never a wrong partial reuse.

The output comparison is: same greedy (top-1) token, and the logprobs of the candidates both top-5 lists share
within 0.2 nats (at least 3 shared). An earlier version compared the full top-5 order within 1e-3, then 5e-2;
the cached request evaluates its suffix in fewer ubatches than the reference, and on this Q4_K_M model short
suffixes deviate by up to 0.19 nats on Metal, baseline included (A3's measurement), so those bounds were thin.

Results of the committed test version (`9b8b6bd93`, `bench/test-hybrid-*.log`, `bench/test-hybrid-fixed-repeats.log`,
`bench/test-subset-*.log`, 2026-10-29 06:14-06:21Z; fixed binary built from `9b8b6bd93`, baseline from `3582c5ee0`):
fixed `[0]` and `[64]` pass, 5/5 repeats pass; baseline `[0]` passes, `[64]` fails at the strict extension with
`(cache_n, prompt_n) == (0, 93)` instead of `(66, 27)`, the bug. The subset `test_completion.py test_slot_save.py
test_ctx_shift.py test_speculative.py test_basic.py test_chat_completion.py test_kv_keep_only_active.py` (`-m "not
slow"`, 4 workers): fixed 140 passed / 1 skipped / 0 failed; baseline 139 passed / 1 skipped / 1 failed (the new
`[64]`). The skip is upstream's `test_cache_vs_nocache_prompt`. The tests need `tools/server/tests/tmp` to exist
(CI's default `LLAMA_CACHE=tmp` creates it); without it `test_slot_save.py` servers die and `test_speculative.py`
errors on both binaries. A3 also ran the test on `origin/master` + patch: 2 passed.

How the tests were run (Python 3.12 venv with `tools/server/tests/requirements.txt`):

    cd tools/server/tests && mkdir -p tmp
    LLAMA_SERVER_BIN_PATH=../../../build-k3/bin/llama-server LLAMA_CACHE=<cache> \
      pytest -v unit/test_completion.py -k test_completion_cache_hybrid_memory

## Measurements on Inkling-Small (UD-IQ3_XXS, M5 Max 128 GB, Metal)

See `bench/results/` (`<tag>.jsonl` rows, `<tag>-server.log` with `-lv 4` so the `created/restored context
checkpoint` and `forcing full prompt re-processing` lines are there - grep with `-a`, the log carries the model's
bytes -, `<tag>-reuse-lines.txt` with those lines extracted, `<tag>-curl.sh` with the equivalent curl commands)
and `bench/RESULTS.md` for the tables. Server flags: `-np 1 --no-kv-unified -ngl 99 -fa 0 -c 70000 --jinja
--temp 0 -ub {512,256}`, `n_predict 64`, greedy.

Provenance and load. The binaries measured are `build-k3-baseline` = `3582c5ee0` (unpatched PR head) and
`build-k3` = `7251878de` (the fix's first commit; server code byte-identical to `9b8b6bd93`), although the `commit`
field of the `meta` rows says `5edd1f152` for all three runs (the driver recorded the worktree HEAD; it now records
`llama-server --version`). The runs were not equally loaded: `load1` at start 6.5 (before-ub512), 17.9
(after-ub512), 26.7 (before-ub256), with other jobs on the machine. The token counts are deterministic
(A3 reproduced every tuple) and are the result; the times are indicative, with about +-15% noise (the uncached
p_8k reference took 29.9 s in the before run and 34.5 s in the after run) and the after column the handicapped
one.

Three server runs under one `GPU_LOCK` hold (18 min): `before-ub512` and `after-ub512` with every request, and
`before-ub256` (the field's ubatch) without the uncached references. `p_8k` = 8218 tokens, `p_2k` = 2072 tokens,
the extension 200 tokens, model load 36 s cold / 6-12 s warm, 92-94 GiB resident. The model reports `n_swa = 512`
and `the context does not support partial sequence removal` (seq_rm type FULL) at start-up.

| request | before `-ub 512` | **after** `-ub 512` | before `-ub 256` |
|---|---|---|---|
| p_8k, no cache (reference) | 8218 tok / 29.9 s | 8218 tok / 34.5 s | - |
| (a) p_8k + 200 tokens, cache on, after a p_8k request | 716 tok / 3.60 s | **204 tok / 1.43 s** | 8418 tok / 42.3 s |
| (a2) p_8k + its 63 generated tokens + 200, cache on (strict extension) | 779 tok / 4.27 s | **200 tok / 1.42 s** | 8481 tok / 35.4 s |
| (c) p_8k with the last 100 tokens replaced, cache on (rewind) | 8218 tok / 30.3 s | **253 tok / 1.90 s** | 8218 tok / 35.1 s |
| p_8k again, cache on, slot held p_8k + 200 + generated | 8218 tok / 29.5 s | 316 tok / 1.83 s | 8218 tok / 46.5 s |

(`prompt_n` / `prompt_ms`; TTFT measured at the first streamed token is `prompt_ms` + 3-10 ms in every row.)
Read the token counts first: +200 tokens 716 -> 204 (`-ub 512`) and 8418 -> 204 (`-ub 256`), strict extension
779 -> 200 and 8481 -> 200, rewind 8218 -> 253. At the measured 3.5-4 ms per prompt token at this depth (5-7 ms
per token for the short suffixes) the times follow from the counts.

| chat turn (`/v1/chat/completions`, turn 1 = p_2k) | before `-ub 512` | **after** `-ub 512` | before `-ub 256` |
|---|---|---|---|
| turn 1 | 2085 tok / 9.05 s | 2085 tok / 9.24 s | 2085 tok / 8.28 s |
| turn 2 | 550 tok / 3.10 s | **18 tok / 0.83 s** | 2110 tok / 8.38 s |
| turn 3 | 592 tok / 3.49 s | **23 tok / 0.86 s** | 2163 tok / 8.60 s |
| turn 4 | 574 tok / 3.40 s | **16 tok / 0.76 s** | 2217 tok / 8.71 s |

What the server log says (`bench/results/*-reuse-lines.txt`):

- before, `-ub 512`: the strict extension and the +200 request fall back to the checkpoint 516 tokens before the
  end of the previous prompt (`restored context checkpoint (pos_min = 7701, ..., n_tokens = 7702)`), because
  516 > n_swa - 2 = 510; the identical prompt and the rewind hit `forcing full prompt re-processing due to lack of
  cache data`; chat turns 2-4 restore the checkpoint 516 tokens back (550-592 tokens re-evaluated).
- before, `-ub 256` (the field's flags): every cached request is `forcing full prompt re-processing`; chat turns
  2-4 restore the 9-token checkpoint at the first user-message start (2110-2217 of ~2120-2230 tokens re-evaluated),
  which is the thread's "turn 2 re-processes 3925 tokens".
- after: the strict extension reuses the live state (no checkpoint line, `cache_n = 8281 = 8218 + 63`); the +200
  request and the identical prompt restore the checkpoint 4 tokens before the end of the previous prompt
  (`n_tokens = 8214`); the rewind restores the nearest checkpoint before the edit (`n_tokens = 7965`); chat turns
  2-4 are strict extensions (18/23/16 tokens).

(d) greedy outputs, 64 tokens, cached vs the same prompt without the cache: the rewind case is identical on both
binaries (64/64). The two extension cases diverge on both binaries (before: at token 19 and 17 through its 516-back
checkpoint; after: at token 5 and 9), and the recorded top-3 logprobs show a near-tie at the divergence with the
same candidates on both sides (B1, token 5: cached -1.54 / -1.56 / -1.60 vs uncached -1.36 / -1.64 / -1.81 for the
same three tokens; B2, token 9: -1.48 / -1.59 vs -1.56 / -1.91 for the same two), and both continuations are
coherent text. That is the batched-vs-incremental drift of evaluating the suffix in a different ubatch split
(expert routing on an MoE model flips at near-ties), which upstream's own `test_cache_vs_nocache_prompt` is
skipped for; it is not a wrong state: the fixed server's checkpoint restore in (c) is bit-identical over 64
tokens, A3 measured the live state against its own checkpoint taken 4 tokens earlier at 0.0004 nats with no
divergence in 64 tokens, the backend is deterministic run-to-run (0.000 over 64 tokens), and the checkpoint-only
paths the baseline relies on diverge at least as early (tokens 9-32, `REVIEW.md` section 4). A3's `-ub 256`
no-cache control and a loop-rate sample were pending at review time.

## Follow-ups (not in this change)

1. **Checkpoint at the end of generation.** After a turn the slot holds prompt + answer, and the recurrent tail is
   the last decoded token. The next turn's prompt re-renders the answer and usually diverges a token or two before
   its end (end token, template whitespace), so the reuse goes through the checkpoint 4 tokens before the end of the
   *previous prompt* and the previous answer is re-evaluated (the per-turn `prompt_n` is ~ answer + new turn + 4,
   not just the new turn). A checkpoint taken when generation stops would cut that to the new turn. PR #24035
   covers part of this ground.
2. **Partial recurrent rollback for Inkling (`llm_arch_supports_rs_rollback`).** Evaluated and left out: the
   server only asks for `n_rs_seq` snapshots for draft-model speculation of the MTP/EAGLE3/DFLASH/DSPARK types
   (`common_params_speculative::need_n_rs_seq`, `common/common.h:403`), so it does not affect the prompt cache at all,
   and Inkling has no converted MTP head (R2 section 8d). Enabling it is small but lives in the Inkling PR's own
   graph code, not upstream: `build_sconv` in `src/models/inkling.cpp:308-345` writes only snapshot slot 0 of the
   conv state (`state_dst` at row `kv_head`); it would have to write slot `s` = the window `s` tokens back for
   `s < min(n_seq_tokens, n_rs_seq + 1)` at row `s * mem_size + kv_head`, exactly as `src/models/lfm-base.cpp:59-73`
   does for LFM2's identical shortconv primitive, for all four packed streams (`off_k`, `off_v`, `off_attn`,
   `off_mlp`), then add `LLM_ARCH_INKLING` to `llm_arch_supports_rs_rollback`. Verify with
   `tests/test-llama-context -m <inkling gguf>` (group RS, `cases-rs.cpp`: rollback and replay against a reference
   context). Suggest it on #25731 when a drafter exists.
3. **Chunked reuse (`--cache-reuse`)** is not gated on the seq_rm type: the server disables it only when the
   memory cannot shift (`llama_kv_cache_iswa::get_can_shift` is false unless the base and SWA caches have the same
   size, hence "cache_reuse is not supported by this context" for Inkling and every hybrid-iswa model), and accepts
   it for a hybrid without SWA, where the attention cache is shifted while the recurrent state still holds the
   removed chunk (an approximation, pre-existing on every binary; A3's case 8: top-1 equal, logprobs off by 0.15).
   Worth a separate upstream note that `can_cache_reuse` should require `COMMON_CONTEXT_SEQ_RM_TYPE_PART`.
4. **CUDA check**: the Modal H100 reproduction of the strict-extension case with the unfixed and fixed
   builds (`package/k1-metal-banded-fa/bench/cuda/CUDA-VERIFY.md`, K3 section) - that run (2026-10-29,
   step 3 of that file): `llama-server` built with `GGML_CUDA` (sm_90) from the same PR head with and without this
   patch, one H200, the same GGUF, `-fa 1 -ngl 99 -np 1 --no-kv-unified -c 16384 -ub 512 --temp 0`, the same prompts
   (p_8k = 8218 tokens, the 200-token continuation), `n_predict 64`. Same `prompt_n` / `cache_n` as the Mac in every
   row: p_8k uncached 8218 / 0 (7.2 s) on both; p_8k cached after the uncached requests: before 8218 / 0 (7.02 s,
   "forcing full prompt re-processing"), after 316 / 7902 (0.41 s); (a) p_8k + 200: before 716 / 7702 (0.77 s, the
   checkpoint 516 back), after 204 / 8214 (0.35 s); (a2) strict extension: before 779 / 7702 (0.81 s), after
   200 / 8281 (0.33 s, no restore line, live state); the uncached twin 8481 / 0 (7.2 s). Greedy tokens of the cached
   requests vs the uncached reference differ on both binaries there too (before: tokens 5 and 10; after: 9 and 5),
   the ubatch-split numerics of the CUDA path. The fix is in the server's reuse gate and is backend-independent.

## Files

- `patches/0001-server-do-not-apply-the-SWA-back-off-to-a-recurrent-.patch` - the commit (`git am`-able)
- `PR.md` - pull-request title and body for ggml-org/llama.cpp
- `bench/bench_prompt_cache.py` - the measurement (starts one server, runs (a)/(a2)/(b)/(c)/(d), kills it)
- `bench/run_all.sh`, `bench/run_four.sh` - the four runs under one `bench/GPU_LOCK` hold
  (`bench/baseline/with_gpu_lock.sh`)
- `bench/summarize.py` - the tables in `bench/RESULTS.md`; `bench/extract_reuse_lines.sh` - the decision lines
- `REVIEW.md` - A3's adversarial review (of `5edd1f152`)
- `bench/results/` - raw rows, server logs, curl commands, run logs
- `bench/build-*.log`, `bench/test-*.log` - build and test logs
