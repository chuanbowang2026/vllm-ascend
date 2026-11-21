#!/usr/bin/env python3
"""L1: mask vs compact visibility equivalence, pure logic.

No NPU, no weights, no vllm import. Runs anywhere in under a second.

This layer encodes the Decoder SWA Bounded Replay semantics and cross-checks
the two modes against each other AND against a closed-form reference. It is
where off-by-one errors live, and it is the only level that can enumerate
chunk boundaries exhaustively -- at L2 the scheduler decides them for you.

Modelled on DeepInfer's examples/yoco/yoco_equiv.py, including its central
property: comparisons are EXACT. There is no tolerance anywhere in this file.

What is compared, per (request, chunk, query row):
  * does the row exist at all
  * the visible sliding-window key interval [lo, hi]

Verdict on the first line; exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE.

    python3 -I l1_visibility_equiv.py
    python3 -I l1_visibility_equiv.py --window 128 --keep 134 --verbose
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

EMPTY = (None, None)


# --------------------------------------------------------------------------
# The semantics, written once. Both modes and the reference derive from here.
# --------------------------------------------------------------------------

def rect_lo(n_prompt: int, keep_tokens: int) -> int:
    """Absolute start of the retained rectangle for a request."""
    return max(0, n_prompt - keep_tokens)


def clip_prefix(n_prompt: int, keep_tokens: int, q_start: int, q_len: int) -> int:
    """Rows [clip_prefix, q_len) of this chunk survive compaction.

    The `q_len - 1` upper clamp keeps at least one row per request so every
    chunk still has an addressable output row for the sampler. DeepInfer:
    prepare_tensor_input_v4.cpp:225.
    """
    return min(max(rect_lo(n_prompt, keep_tokens) - q_start, 0), q_len - 1)


def visibility(pos: int, window: int, floor: int, *, clamp_like_kernel: bool) -> tuple:
    """Visible SWA key interval for a query at absolute position `pos`.

    `clamp_like_kernel` reproduces the arch22 kernel's Min(floor, mask_right):
    a row below the floor degenerates to a single key rather than producing an
    empty interval (which would underflow oriS2Size as uint32). Without the
    clamp the mathematically correct answer is "empty".
    """
    hi = pos
    lo = max(pos - window + 1, 0)
    if clamp_like_kernel:
        lo = max(lo, min(floor, hi))
    else:
        lo = max(lo, floor)
        if lo > hi:
            return EMPTY
    return (lo, hi)


def chunks(n_prompt: int, chunk_size: int):
    """(q_start, q_len) per prefill chunk for one request."""
    out, start = [], 0
    while start < n_prompt:
        length = min(chunk_size, n_prompt - start)
        out.append((start, length))
        start += length
    return out


# --------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------

def rows_mask(n_prompt, keep_tokens, q_start, q_len, window):
    """mask mode: every row survives; visibility narrowed, kernel-clamped."""
    floor = rect_lo(n_prompt, keep_tokens)
    return {
        q_start + j: visibility(q_start + j, window, floor, clamp_like_kernel=True)
        for j in range(q_len)
    }


def rows_compact(n_prompt, keep_tokens, q_start, q_len, window):
    """compact mode: only [clip_prefix, q_len) survive; same visibility."""
    floor = rect_lo(n_prompt, keep_tokens)
    cp = clip_prefix(n_prompt, keep_tokens, q_start, q_len)
    return {
        q_start + j: visibility(q_start + j, window, floor, clamp_like_kernel=True)
        for j in range(cp, q_len)
    }


def rows_reference(n_prompt, keep_tokens, q_start, q_len, window):
    """Closed-form reference, independent of both implementations."""
    floor = max(0, n_prompt - keep_tokens)
    out = {}
    for j in range(q_len):
        pos = q_start + j
        if pos < floor:
            continue  # not part of the retained rectangle
        lo = max(pos - window + 1, floor, 0)
        out[pos] = (lo, pos)
    return out


# --------------------------------------------------------------------------

def check_case(rep, label, n_prompt, keep_tokens, chunk_size, window, verbose):
    floor = rect_lo(n_prompt, keep_tokens)
    problems = []
    kept_total = 0

    for q_start, q_len in chunks(n_prompt, chunk_size):
        m = rows_mask(n_prompt, keep_tokens, q_start, q_len, window)
        c = rows_compact(n_prompt, keep_tokens, q_start, q_len, window)
        r = rows_reference(n_prompt, keep_tokens, q_start, q_len, window)
        kept_total += len(c)

        # (a) every row compact keeps must agree with mask, exactly.
        for pos, vis in c.items():
            if pos not in m:
                problems.append(f"{label}: compact kept {pos}, mask dropped it")
            elif m[pos] != vis:
                problems.append(f"{label}: pos {pos} mask={m[pos]} compact={vis}")

        # (b) rows mask keeps but compact drops must be below the rectangle.
        #     That is the ONE sanctioned difference between the modes.
        for pos in m.keys() - c.keys():
            if pos >= floor:
                problems.append(f"{label}: compact dropped in-rectangle row {pos} (floor={floor})")

        # (c) the surviving set must match the closed-form reference, except
        #     for the mandatory placeholder row when a whole chunk is clipped.
        extra = c.keys() - r.keys()
        missing = r.keys() - c.keys()
        if missing:
            problems.append(f"{label}: compact dropped reference rows {sorted(missing)[:5]}")
        if len(extra) > 1:
            problems.append(f"{label}: compact kept {len(extra)} rows below the rectangle, expected <=1")
        for pos in extra:
            if pos != q_start + q_len - 1:
                problems.append(f"{label}: below-rectangle survivor {pos} is not the placeholder row")
        for pos in c.keys() & r.keys():
            if c[pos] != r[pos]:
                problems.append(f"{label}: pos {pos} compact={c[pos]} reference={r[pos]}")

        # (d) THE SAFETY PROPERTY: a surviving row must never need a key that
        #     compact never wrote. Every key it reads must itself be at or
        #     above the rectangle. DeepInfer asserts the same thing in-engine
        #     (TORCH_CHECK_GE(vis.k_lo, clip_prefix)).
        for pos, (lo, hi) in c.items():
            if lo is None:
                continue
            if pos >= floor and lo < floor:
                problems.append(
                    f"{label}: row {pos} reads key {lo} below the rectangle (floor={floor}) "
                    "-- compact never wrote it")

        # (e) at least one row per chunk, for the sampler.
        if len(c) < 1:
            problems.append(f"{label}: chunk at {q_start} kept no rows")

    ok = not problems
    detail = "" if ok else "\n".join(problems[:6]) + (
        f"\n... and {len(problems) - 6} more" if len(problems) > 6 else "")
    rep.check(label, ok, detail=detail)
    if verbose and ok:
        rep.note(f"{label}: floor={floor}, kept {kept_total}/{n_prompt} rows "
                 f"({100 * kept_total / n_prompt:.2f}%)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--keep", type=int, default=0, help="0 => sweep a set of legal values")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("l1_visibility_equiv", "mask vs compact visibility, exact comparison")
    W = args.window

    keeps = [args.keep] if args.keep else [W + 1, W + 6, W + 12]
    for k in keeps:
        if k < W + 1:
            rep.check(f"keep={k}", False,
                      detail=f"keep_tokens must be >= window+1 ({W + 1}); a shorter tail "
                             "truncates the window after a speculative rollback")
            return rep.emit(args.results_dir)
    rep.fact("window", W)
    rep.fact("keep_tokens_swept", keeps)

    # Boundary-heavy case matrix: prompts shorter than / equal to / longer
    # than the tail, chunk sizes that land before, on and inside the
    # rectangle, and the off-by-one neighbourhood of every boundary.
    prompts = [1, 2, W - 1, W, W + 1, 255, 256, 257, 8191, 8192, 8193, 150_000]
    chunk_sizes = [1, 2, W, 400, 8192]

    for keep in keeps:
        for n, cs in itertools.product(prompts, chunk_sizes):
            # keep the matrix cheap: skip tiny chunks on the huge prompt
            if n > 20_000 and cs < 400:
                continue
            check_case(rep, f"keep{keep}.N{n}.chunk{cs}", n, keep, cs, W, args.verbose)

    # Near-boundary sweep around the rectangle edge for one representative N.
    n = 8192
    for keep in keeps:
        f = rect_lo(n, keep)
        for delta in (-2, -1, 0, 1, 2):
            cs = max(1, f + delta)
            check_case(rep, f"edge.keep{keep}.chunkends@floor{delta:+d}", n, keep, cs, W, args.verbose)

    rep.note("No tolerances are used anywhere in this file: every comparison is ==. "
             "An approximate optimization still has to be exactly the approximation it claims.")
    rep.note("What this level CANNOT catch: wrong gather/scatter indices, metadata that is "
             "compacted inconsistently across KV-cache groups, and anything numeric. "
             "That is L2's job.")
    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
