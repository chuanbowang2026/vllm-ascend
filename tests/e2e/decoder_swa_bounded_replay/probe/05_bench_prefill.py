#!/usr/bin/env python3
"""Stage-0 probe: baseline prefill throughput and MFU against a running service.

Answers the single most important open number: the ACTUAL MFU. Every absolute
savings estimate for Decoder SWA Bounded Replay scales with it.

Method: send prompts of controlled length with max_tokens=1 so the step is
prefill-dominated, and measure wall time. The server's reported
`usage.prompt_tokens` is used as the ground-truth token count, so no tokenizer
is needed here.

Needs: a running vLLM service with the model loaded (see 04_baseline_serve.sh).
Only uses the stdlib.

Usage:
  python3 -I 05_bench_prefill.py --url http://127.0.0.1:8900 --model dsv41
  python3 -I 05_bench_prefill.py --url ... --lengths 4096,16384,65536 --concurrency 1,4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

# ---- model constants, verified by 03_probe_model.py ------------------------
D, H, dh, Lq, Lo, G = 5120, 64, 512, 1280, 1024, 8
E, K_TOP, NS, IM = 384, 6, 1, 2304
L, M, W, IDX_TOPK = 40, 4, 128, 512
HI, DI, CAND = 32, 128, 2048 * 8
N_IDX_SRC = 8

# A3 node peak, INT8 (W8A8). 16 logical devices x ~560 TOPS/die.
# NOTE: analyst-reconstructed, no official datasheet. Override with --peak-tops.
DEFAULT_PEAK_TOPS = 560e12 * 16


def dense_flops_per_token() -> float:
    attn_proj = D * Lq + Lq * H * dh + D * dh + H * dh * Lo + G * Lo * D
    moe = (K_TOP + NS) * 3 * D * IM + D * E
    hc = 2 * (M * D) * ((2 + M) * M)
    attn_kernel = 2 * H * dh * (W + IDX_TOPK)
    return 2.0 * L * (attn_proj + moe + hc + attn_kernel)


def indexer_flops_per_token(s_avg: float) -> float:
    scoring = 3 * HI * DI * (s_avg / 2) + 1 * HI * DI * s_avg
    qsli = 4 * HI * DI * CAND
    proj = N_IDX_SRC * (Lq * HI * DI + D * HI)
    return 2.0 * (scoring + qsli + proj)


def post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return body, time.perf_counter() - t0


def make_prompt(approx_tokens: int) -> str:
    # ~1 token per short word for this tokenizer family; the exact count is read
    # back from usage.prompt_tokens, so approximation here is fine.
    chunk = "the quick brown fox jumps over the lazy dog and then rests a while . "
    reps = max(1, approx_tokens // 16 + 1)
    return (chunk * reps)[: approx_tokens * 6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--lengths", default="2048,8192,32768,131072",
                    help="comma-separated approximate prompt token counts")
    ap.add_argument("--concurrency", default="1,4", help="comma-separated concurrency levels")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--peak-tops", type=float, default=DEFAULT_PEAK_TOPS,
                    help="node INT8 peak ops/s used for the MFU denominator")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("05_bench_prefill", "baseline prefill throughput and MFU")
    rep.fact("peak_tops_assumed", f"{args.peak_tops / 1e12:.0f} TOPS (INT8, whole node)")
    rep.note("peak_tops is analyst-reconstructed (no official 910C datasheet). "
             "MFU scales inversely with it; re-run with --peak-tops to rescale.")

    comp_url = args.url.rstrip("/") + "/v1/completions"

    # ---- liveness ---------------------------------------------------------
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=30) as r:
            served = [m.get("id") for m in json.loads(r.read().decode()).get("data", [])]
        rep.fact("served_models", served)
        if args.model not in served:
            rep.check("service_has_model", False, args.model, served)
            return rep.emit(args.results_dir)
        rep.check("service_reachable", True)
    except Exception as exc:
        rep.check("service_reachable", None, detail=f"{type(exc).__name__}: {exc}")
        return rep.emit(args.results_dir)

    lengths = [int(x) for x in args.lengths.split(",") if x]
    concs = [int(x) for x in args.concurrency.split(",") if x]

    def one(n_tokens: int) -> tuple[int, float]:
        body, dt = post(
            comp_url,
            {"model": args.model, "prompt": make_prompt(n_tokens),
             "max_tokens": 1, "temperature": 0, "stream": False},
            args.timeout,
        )
        return int(body.get("usage", {}).get("prompt_tokens", 0)), dt

    rows = []
    for n in lengths:
        for _ in range(args.warmup):
            try:
                one(n)
            except Exception:
                pass
        for c in concs:
            samples = []
            for _ in range(args.repeats):
                try:
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=c) as pool:
                        out = list(pool.map(lambda _i: one(n), range(c)))
                    wall = time.perf_counter() - t0
                    toks = sum(t for t, _ in out)
                    if toks == 0:
                        continue
                    samples.append((toks, wall))
                except Exception as exc:
                    rep.note(f"len={n} conc={c}: {type(exc).__name__}: {exc}")
            if not samples:
                rep.check(f"bench.len{n}.conc{c}", None, detail="all repeats failed")
                continue

            tps = [t / w for t, w in samples]
            best = max(tps)
            med = statistics.median(tps)
            real_tokens = samples[0][0] // c
            s_avg = real_tokens / 2.0  # mean context over a chunked prefill
            fl_per_tok = dense_flops_per_token() + indexer_flops_per_token(s_avg)
            mfu = best * fl_per_tok / args.peak_tops

            row = {
                "requested_tokens": n,
                "actual_prompt_tokens": real_tokens,
                "concurrency": c,
                "median_tok_per_s": round(med, 1),
                "best_tok_per_s": round(best, 1),
                "gflop_per_token": round(fl_per_tok / 1e9, 2),
                "mfu_at_best": round(mfu, 4),
            }
            rows.append(row)
            rep.check(f"bench.len{real_tokens}.conc{c}", True,
                      detail=f"{best:.0f} tok/s, MFU {100 * mfu:.1f}%")

    rep.fact("rows", rows)
    if rows:
        mfus = [r["mfu_at_best"] for r in rows]
        rep.fact("mfu_min", round(min(mfus), 4))
        rep.fact("mfu_max", round(max(mfus), 4))
        rep.note("Expected savings from Decoder SWA Bounded Replay = "
                 "~0.5 x (1 - keep_tokens/N) of PREFILL compute. Absolute gain in tok/s "
                 "scales with the MFU measured here.")
        rep.note("If MFU is far below ~25%, compute is not the binding constraint and the "
                 "EP all-to-all / overlap work in docs/04 should be reprioritised above the "
                 "layer skip.")
    else:
        rep.check("any_successful_row", None, detail="no benchmark row completed")

    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
