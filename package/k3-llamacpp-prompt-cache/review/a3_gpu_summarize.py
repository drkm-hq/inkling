#!/usr/bin/env python3
"""Print the tables of an a3_gpu_run.py jsonl (usage: a3_gpu_summarize.py <tag>.jsonl)."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    k = r["kind"]
    if k == "meta":
        print(f"meta: ub={r['ub']} n_predict={r['n_predict']} load1={r['load1']:.1f} cmd={' '.join(r['cmd'][1:])}")
    elif k == "load":
        print(f"load: {r['load_s']} s, rss {r['rss_gib']} GiB")
    elif k == "prompts":
        print(f"prompts: {r}")
    elif k == "completion":
        print(f"  {r['name']:16s} cache={str(r['cache_prompt']):5s} n_prompt={r['n_prompt']:5d} prompt_n={r['prompt_n']:5d} "
              f"cache_n={r['cache_n']:5d} prompt_ms={r['prompt_ms']:9.1f} ttft={r['ttft_s']} pred_n={r['predicted_n']} "
              f"pred_ms={r['predicted_ms']:.0f} rss={r['rss_gib']}  # {r['note']}")
    elif k == "compare":
        print(f"  compare {r['name']}: first_diff={r['first_diff']} n={r['n_cached']}/{r['n_ref']} "
              f"step0_dev={r.get('step0_max_dev')} max_dev_before_diff={r['max_dev_before_diff']} idset_diffs={r['idset_diffs_before_diff']}")
        if "margin_in_cached" in r:
            print(f"     at diff: tok cached={r['tok_cached']} ref={r['tok_ref']} margin_in_cached={r['margin_in_cached']} "
                  f"margin_in_ref={r['margin_in_ref']} defect_gt_0p7={r['defect_gt_0p7']}")
            print(f"     cached top5={r['at_diff_cached_top5']}")
            print(f"     ref    top5={r['at_diff_ref_top5']}")
        print(f"     step0 cached={r.get('step0_top5_cached')}")
        print(f"     step0 ref   ={r.get('step0_top5_ref')}")
    elif k == "chat":
        print(f"  chat turn {r['turn']} prompt_n={r['prompt_n']} cache_n={r['cache_n']} prompt_ms={r['prompt_ms']} "
              f"ttft={r['ttft_s']} pred_n={r['predicted_n']} pred_ms={r['predicted_ms']} rss={r['rss_gib']}  # {r['note']}")
    elif k == "chat_rss":
        print(f"  chat rss: before {r['rss_before_gib']} after {r['rss_after_gib']} GiB")
    elif k == "done":
        print(f"done: rss {r['rss_gib']} GiB load1 {r['load1']:.1f}")
