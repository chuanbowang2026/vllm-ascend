#!/usr/bin/env python3
"""L2: mask vs compact engine equivalence, exact.

Two services differing ONLY in decoder_swa_replay_mode, driven over the same
prompts with greedy decoding, compared with ZERO tolerance.

Why this direction, and why exact
---------------------------------
`mask` keeps every query row and only narrows sliding-window visibility.
`compact` drops the rows that mask made invisible. If compact is correct, the
rows it keeps see exactly what mask's corresponding rows saw, so the two must
agree BIT FOR BIT. They are the same computation on the same inputs; only the
rows that provably cannot influence the output are absent.

That is why there is no rtol/atol here, and why `off` is never the reference:
comparing against `off` would measure the approximation itself, which is a
model-quality question that DeepSeek answered in post-training, not an
implementation question. DeepInfer's examples/yoco/yoco_equiv.py enforces the
same direction (`assert ref_impls == {IMPL_MASK} and cand_impls == {IMPL_COMPACTION}`).

So: a mismatch here is a BUG in compact -- a wrong gather index, inconsistently
compacted metadata, or a stale shared buffer. It is not "acceptable drift".

Verdict on the first line; exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE.

    python3 -I l2_engine_equiv.py --mask http://127.0.0.1:8900 \
                                  --compact http://127.0.0.1:8901
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

TOP_LOGPROBS = 5
MAX_TOKENS = 32


def _post(url: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def sample(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Greedy, with top-k logprobs, so the comparison sees numbers not just text."""
    body = _post(base.rstrip("/") + "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "top_p": 1.0, "seed": 1234,
        "logprobs": TOP_LOGPROBS, "stream": False,
    })
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    return {
        "text": choice.get("text", ""),
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "top_logprobs": lp.get("top_logprobs"),
        "prompt_tokens": int(body.get("usage", {}).get("prompt_tokens", 0)),
        "finish_reason": choice.get("finish_reason"),
    }


def first_divergence(a: dict, b: dict) -> str | None:
    """Exact comparison. Returns a human-readable first difference, or None."""
    if a["tokens"] != b["tokens"]:
        ta, tb = a["tokens"] or [], b["tokens"] or []
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                return f"token[{i}] {x!r} != {y!r}"
        return f"token count {len(ta)} != {len(tb)}"
    if a["token_logprobs"] != b["token_logprobs"]:
        la, lb = a["token_logprobs"] or [], b["token_logprobs"] or []
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                return f"token_logprob[{i}] {x!r} != {y!r} (delta {abs((x or 0) - (y or 0)):.3e})"
        return "token_logprobs length differs"
    if a["top_logprobs"] != b["top_logprobs"]:
        for i, (x, y) in enumerate(zip(a["top_logprobs"] or [], b["top_logprobs"] or [])):
            if x != y:
                return f"top_logprobs[{i}] {x} != {y}"
        return "top_logprobs length differs"
    if a["finish_reason"] != b["finish_reason"]:
        return f"finish_reason {a['finish_reason']!r} != {b['finish_reason']!r}"
    return None


def build_prompts(keep_tokens: int) -> list[tuple[str, str]]:
    """Prompts chosen to straddle the rectangle boundary.

    The interesting lengths are around keep_tokens itself (whole prompt inside
    the rectangle), and well beyond it with chunked prefill so that early
    chunks fall entirely below the floor and keep only a placeholder row.
    """
    unit = "The quick brown fox jumps over the lazy dog near the riverbank. "
    cases = []
    cases.append(("tiny", "Name three primes."))
    cases.append(("inside_rect", unit * 8 + "\nSummarise the above in one sentence."))
    cases.append(("just_over_rect", unit * 40 + "\nSummarise the above in one sentence."))
    cases.append(("multi_chunk", unit * 900 + "\nSummarise the above in one sentence."))
    cases.append(("long_chunked", unit * 2500 + "\nWhat animal was mentioned?"))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="reference service, decoder_swa_replay_mode=mask")
    ap.add_argument("--compact", required=True, help="candidate service, mode=compact")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--keep-tokens", type=int, default=129)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("l2_engine_equiv", "mask (reference) vs compact (candidate), zero tolerance")
    rep.fact("keep_tokens_assumed", args.keep_tokens)

    # ---- comparability guards -------------------------------------------
    for name, url in (("mask", args.mask), ("compact", args.compact)):
        try:
            served = [m.get("id") for m in _get(url.rstrip("/") + "/v1/models").get("data", [])]
        except Exception as exc:
            rep.check(f"{name}_reachable", None, detail=f"{url}: {type(exc).__name__}: {exc}")
            return rep.emit(args.results_dir)
        if args.model not in served:
            rep.check(f"{name}_serves_model", False, args.model, served)
            return rep.emit(args.results_dir)
    rep.check("both_services_reachable", True)

    if args.mask.rstrip("/") == args.compact.rstrip("/"):
        rep.check("distinct_endpoints", False, detail="--mask and --compact point at the same service")
        return rep.emit(args.results_dir)
    rep.check("distinct_endpoints", True)
    rep.note("This harness CANNOT verify that the two services differ only in "
             "decoder_swa_replay_mode. Launch them from the same script with one flag "
             "changed, and keep --seed, --max-num-batched-tokens, --max-num-seqs, DP/TP and "
             "every VLLM_ASCEND_* variable identical. A config difference would show up here "
             "as a false failure.")

    # ---- the comparison --------------------------------------------------
    rows, mismatches = [], 0
    for label, prompt in build_prompts(args.keep_tokens):
        try:
            ref = sample(args.mask, args.model, prompt, args.max_tokens)
            cand = sample(args.compact, args.model, prompt, args.max_tokens)
        except Exception as exc:
            rep.check(f"case.{label}", None, detail=f"{type(exc).__name__}: {exc}")
            continue

        if ref["tokens"] is None or cand["tokens"] is None:
            rep.check(f"case.{label}", None,
                      detail="server returned no logprobs; rerun with a build that supports "
                             "`logprobs` on /v1/completions, otherwise only text can be compared")
            continue

        diff = first_divergence(ref, cand)
        ok = diff is None
        mismatches += 0 if ok else 1
        rows.append({"case": label, "prompt_tokens": cand["prompt_tokens"],
                     "equal": ok, "first_divergence": diff})
        rep.check(f"case.{label}", ok, "bit-identical to mask", diff or "identical",
                  critical=True,
                  detail=("" if ok else
                          "compact kept a row whose computation differs from mask's. Likely "
                          "causes, in order: a wrong gather index; metadata compacted "
                          "inconsistently across the 10 SWA cache groups; a shared index "
                          "buffer (topk/candidate) not gathered at the boundary layer."))

    rep.fact("cases", rows)
    rep.fact("mismatch_count", mismatches)
    if not rows:
        rep.check("any_case_completed", None, detail="no case produced a comparison")

    rep.note("A failure here is a bug in compact, not acceptable drift: mask and compact are "
             "the same computation over the same inputs, differing only in rows that cannot "
             "influence the output.")
    rep.note("If every case passes, the remaining risk is workload-shaped: run L1 for "
             "exhaustive chunk-boundary coverage, which the scheduler will not reproduce here.")
    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
