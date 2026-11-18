#!/usr/bin/env python3
"""Single-operator validation of the ori_win_floor kernel patch.

Replays one captured npu_sparse_flash_mla call (see capture_smla_args.py) three
ways and checks the properties that the patch must have. Needs an NPU and the
rebuilt operator package; needs NO model weights beyond the capture step.

Verdict on the first line; exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE.

    python3 -I test_ori_win_floor.py --dump /tmp/smla_consumer.pt

Checks, in order of importance:

  K1  absent floor is BIT-IDENTICAL to the captured (pre-patch) output.
      This is the no-regression guarantee for all 20 encoder layers and for
      every deployment that leaves the feature off.
  K2  a floor of 0 is also bit-identical (Max(x, 0, 0) == Max(x, 0)).
  K3  a floor beyond every query position does not crash and does not produce
      NaN/Inf. This is the Min(floor, mask_right) clamp: without it,
      oriMaskRight - oriMaskLeft + 1 underflows as uint32.
  K4  a floor inside the sequence actually CHANGES the output. A patch that
      silently ignores the tensor would pass K1-K3; only this catches it.
  K5  monotonicity: raising the floor never widens the attended set, so the
      output must keep changing (or stay equal) but never revert to the
      unfloored value once it has diverged.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

SCALAR_KEYS = {
    "softmax_scale", "cmp_ratio", "ori_mask_mode", "cmp_mask_mode",
    "ori_win_left", "ori_win_right", "layout_q", "layout_kv",
    "topk_value_mode", "return_softmax_lse",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="file written by capture_smla_args.py")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("test_ori_win_floor", "single-op validation of the ori_win_floor patch")

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        rep.check("import", None, detail=f"{type(exc).__name__}: {exc}")
        return rep.emit(args.results_dir)

    if not os.path.exists(args.dump):
        rep.check("dump_exists", None, actual=args.dump,
                  detail="run capture_smla_args.py against a live engine first")
        return rep.emit(args.results_dir)

    payload = torch.load(args.dump, map_location="cpu")
    reference_out = payload.pop("__output__", None)
    q = payload.pop("q")

    dev = "npu"
    kwargs = {}
    for k, v in payload.items():
        if k in SCALAR_KEYS or not hasattr(v, "to"):
            kwargs[k] = v
        else:
            kwargs[k] = v.to(dev)
    q = q.to(dev)
    kwargs.pop("ori_win_floor", None)

    cu = kwargs.get("cu_seqlens_q")
    seqused = kwargs.get("seqused_ori_kv")
    if cu is None or seqused is None:
        rep.check("capture_has_batch_info", None,
                  detail="dump lacks cu_seqlens_q / seqused_ori_kv; cannot build a floor")
        return rep.emit(args.results_dir)
    num_reqs = int(cu.numel() - 1)
    rep.fact("num_reqs", num_reqs)
    rep.fact("q_shape", list(q.shape))
    rep.fact("seqused_ori_kv", seqused[:num_reqs].tolist())
    rep.fact("ori_win_left", kwargs.get("ori_win_left"))
    rep.fact("ori_mask_mode", kwargs.get("ori_mask_mode"))

    op = torch.ops._C_ascend.npu_sparse_flash_mla

    def run(floor_values) -> "torch.Tensor":
        kw = dict(kwargs)
        if floor_values is not None:
            kw["ori_win_floor"] = torch.tensor(floor_values, dtype=torch.int32, device=dev)
        out, _ = op(q, **kw)
        return out.float().cpu()

    # --- K1: absent floor reproduces the capture bit-for-bit ---------------
    try:
        base = run(None)
    except Exception as exc:
        rep.check("K1.absent_floor_runs", False, detail=f"{type(exc).__name__}: {exc}",
                  critical=True)
        return rep.emit(args.results_dir)

    if reference_out is None:
        rep.check("K1.bitwise_vs_capture", None, detail="dump has no recorded output")
    else:
        same = torch.equal(base, reference_out.float())
        rep.check("K1.bitwise_vs_capture", same, "bit-identical",
                  f"max_abs_diff={float((base - reference_out.float()).abs().max()):.3e}",
                  critical=True,
                  detail="absent ori_win_floor must not change any existing path")

    # --- K2: floor 0 is a no-op -------------------------------------------
    zero = run([0] * num_reqs)
    rep.check("K2.floor_zero_is_noop", torch.equal(zero, base), "bit-identical",
              f"max_abs_diff={float((zero - base).abs().max()):.3e}", critical=True)

    # --- K3: floor above every position must not underflow -----------------
    huge = [int(seqused[i]) + 1024 for i in range(num_reqs)]
    try:
        out_huge = run(huge)
        finite = bool(torch.isfinite(out_huge).all())
        rep.check("K3.floor_above_sequence_is_safe", finite, "finite output",
                  f"finite={finite}", critical=True,
                  detail="guards the uint32 underflow of oriMaskRight - oriMaskLeft + 1")
    except Exception as exc:
        rep.check("K3.floor_above_sequence_is_safe", False,
                  detail=f"{type(exc).__name__}: {exc}", critical=True)

    # --- K4: an in-range floor must actually change the result -------------
    mid = [max(1, int(seqused[i]) // 2) for i in range(num_reqs)]
    out_mid = run(mid)
    changed = not torch.equal(out_mid, base)
    rep.check("K4.in_range_floor_changes_output", changed, "different from unfloored",
              f"max_abs_diff={float((out_mid - base).abs().max()):.3e}", critical=True,
              detail="a patch that silently ignores the tensor would still pass K1-K3")

    # --- K5: raising the floor keeps narrowing -----------------------------
    rows = []
    prev = base
    for frac in (0.25, 0.5, 0.75):
        f = [max(1, int(int(seqused[i]) * frac)) for i in range(num_reqs)]
        cur = run(f)
        rows.append({"frac": frac,
                     "diff_vs_base": float((cur - base).abs().max()),
                     "diff_vs_prev": float((cur - prev).abs().max())})
        prev = cur
    rep.fact("floor_sweep", rows)
    never_reverts = all(r["diff_vs_base"] > 0 for r in rows)
    rep.check("K5.monotone_narrowing", never_reverts, "every floor differs from unfloored",
              rows, detail="if a mid floor reverts to the base output the index frame is wrong")

    rep.note("Frame of reference: ori_win_floor[b] is an ABSOLUTE index in the same logical "
             "KV frame as the kernel's oriMaskLeft/oriMaskRight, i.e. [0, seqused_ori_kv[b]). "
             "A frame mismatch is silent -- wrong attention, no error -- so K4/K5 are the only "
             "signals that it is right.")
    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
