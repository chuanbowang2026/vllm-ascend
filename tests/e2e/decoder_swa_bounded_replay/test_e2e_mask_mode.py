#!/usr/bin/env python3
"""End-to-end check for Decoder SWA Bounded Replay, mask mode.

Compares a baseline service (feature off) against a mask-mode service on the
same prompts, greedy decoding.

WHAT THIS DOES AND DOES NOT ASSERT
----------------------------------
Mask mode narrows the decoder half's sliding-window visibility, so its output
is NOT expected to match the baseline bit-for-bit -- that divergence IS the
approximation the paper describes, and DeepSeek compensates for it with
train-aware adaptation during post-training. Asserting equality here would be
wrong.

What we do assert:
  E1  the service answers at all, with non-empty, finite output
  E2  divergence from baseline stays bounded (reported, with a soft gate)
  E3  throughput is NOT materially worse than baseline. Mask mode saves no
      compute, so a large regression means the patch added overhead rather
      than that the optimization "did not work".

The bit-exact test is mask-vs-compact, and belongs to stage 3.

Usage:
  # 1) baseline service on :8900, mask-mode service on :8901
  python3 -I test_e2e_mask_mode.py --baseline http://127.0.0.1:8900 \
                                   --candidate http://127.0.0.1:8901
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                "stage0-probe", "lib"))
from verdict import Report  # noqa: E402

PROMPTS = [
    "Explain in three sentences why a sliding-window attention cache cannot be shared across layers.",
    "List the first 12 prime numbers, separated by commas, and nothing else.",
    "Summarise the trade-off between prefill compute and KV cache size for long-context inference.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is 17 * 23? Answer with the number only.",
]
LONG_PROMPT_REPEATS = 400  # ~8-12k tokens, forces chunked prefill


def post(url: str, payload: dict, timeout: float = 1800.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), time.perf_counter() - t0


def generate(base_url: str, model: str, prompt: str, max_tokens: int):
    body, dt = post(base_url.rstrip("/") + "/v1/completions",
                    {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                     "temperature": 0, "seed": 1234, "stream": False})
    text = body["choices"][0]["text"]
    usage = body.get("usage", {})
    return text, int(usage.get("prompt_tokens", 0)), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--min-similarity", type=float, default=0.60,
                    help="soft gate on mean token-sequence similarity vs baseline")
    ap.add_argument("--max-throughput-regression", type=float, default=0.10,
                    help="fail if candidate prefill throughput is worse by more than this fraction")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("test_e2e_mask_mode", "baseline vs decoder SWA bounded replay (mask)")

    for name, url in (("baseline", args.baseline), ("candidate", args.candidate)):
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as r:
                json.loads(r.read().decode())
        except Exception as exc:
            rep.check(f"{name}_reachable", None, detail=f"{url}: {type(exc).__name__}: {exc}")
            return rep.emit(args.results_dir)
    rep.check("services_reachable", True)

    long_prompt = ("The following is a long technical document about inference systems. " *
                   LONG_PROMPT_REPEATS)
    cases = [(f"short[{i}]", p) for i, p in enumerate(PROMPTS)]
    cases.append(("long[chunked]", long_prompt + "\n\nIn one sentence, what was this document about?"))

    rows, sims = [], []
    for label, prompt in cases:
        try:
            b_text, b_tok, b_dt = generate(args.baseline, args.model, prompt, args.max_tokens)
            c_text, c_tok, c_dt = generate(args.candidate, args.model, prompt, args.max_tokens)
        except Exception as exc:
            rep.check(f"E1.{label}", False, detail=f"{type(exc).__name__}: {exc}")
            continue

        ok = bool(c_text.strip())
        rep.check(f"E1.{label}.non_empty", ok, "non-empty output", repr(c_text[:60]))
        sim = difflib.SequenceMatcher(None, b_text, c_text).ratio()
        sims.append(sim)
        rows.append({"case": label, "prompt_tokens": c_tok, "similarity": round(sim, 4),
                     "baseline_s": round(b_dt, 3), "candidate_s": round(c_dt, 3),
                     "baseline_text": b_text[:160], "candidate_text": c_text[:160]})

    rep.fact("cases", rows)
    if sims:
        mean_sim = statistics.mean(sims)
        rep.fact("mean_similarity", round(mean_sim, 4))
        rep.check("E2.divergence_bounded", mean_sim >= args.min_similarity,
                  f">= {args.min_similarity}", round(mean_sim, 4),
                  detail=("Mask mode is an APPROXIMATION -- divergence is expected and is not a "
                          "bug by itself. A very low score means the floor is wrong (bad index "
                          "frame) rather than merely approximate. Inspect the per-case texts."))
    else:
        rep.check("E2.divergence_bounded", None, detail="no case completed")

    # --- E3: throughput must not regress ----------------------------------
    long_rows = [r for r in rows if r["case"].startswith("long")]
    if long_rows:
        r = long_rows[0]
        b_tps = r["prompt_tokens"] / r["baseline_s"]
        c_tps = r["prompt_tokens"] / r["candidate_s"]
        regression = (b_tps - c_tps) / b_tps if b_tps > 0 else 0.0
        rep.fact("baseline_prefill_tok_per_s", round(b_tps, 1))
        rep.fact("candidate_prefill_tok_per_s", round(c_tps, 1))
        rep.fact("throughput_regression", round(regression, 4))
        rep.check("E3.no_throughput_regression", regression <= args.max_throughput_regression,
                  f"<= {args.max_throughput_regression}", round(regression, 4),
                  detail=("Mask mode saves NO compute, so throughput should be ~flat. A large "
                          "regression means the patch added overhead (e.g. a per-step H2D copy "
                          "that is not being overlapped), not that the feature underperformed."))
    else:
        rep.check("E3.no_throughput_regression", None, detail="long case did not complete")

    rep.note("Expected next step: stage 3 (compact mode) is where throughput should actually "
             "improve, by roughly 0.5 * (1 - keep_tokens/N) of prefill compute.")
    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
