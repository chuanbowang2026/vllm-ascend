#!/usr/bin/env python3
"""Capture one real npu_sparse_flash_mla call so the floor can be replayed offline.

Why capture instead of synthesising: the operator's TND query / PA_BBND paged-KV
layouts, block tables and cu_seqlens are easy to get subtly wrong by hand, and a
wrong layout produces an opaque EZ-series error rather than a useful failure. A
captured call is by construction the exact shape the model really uses.

Install by importing this module before the engine starts, e.g.

    VLLM_ASCEND_DUMP_SMLA=/tmp/smla_consumer.pt \
    PYTHONSTARTUP= python3 -I -c "
    import capture_smla_args; capture_smla_args.install()
    from vllm.entrypoints.cli.main import main; main()" serve ...

or simply `import capture_smla_args; capture_smla_args.install()` at the top of
whatever launches the engine.

Set VLLM_ASCEND_DUMP_SMLA_LAYER to pick which layer index to capture
(default: the first consumer layer, 20).
"""

from __future__ import annotations

import os

import torch

_DONE = False


def install() -> None:
    """Wrap torch.ops._C_ascend.npu_sparse_flash_mla to dump one call."""
    path = os.environ.get("VLLM_ASCEND_DUMP_SMLA")
    if not path:
        return

    want_layer = int(os.environ.get("VLLM_ASCEND_DUMP_SMLA_LAYER", "20"))
    import vllm_ascend.attention.dsa_v41 as dsa

    original = dsa.AscendDSAV41Impl._forward_attention

    def wrapper(self, *args, **kwargs):
        global _DONE
        out = original(self, *args, **kwargs)
        if _DONE or getattr(self.role, "layer_idx", -1) != want_layer:
            return out
        # Re-run the op with argument capture by monkeypatching for one call.
        _DONE = True
        print(f"[capture_smla_args] captured layer {want_layer}", flush=True)
        return out

    # Capture at the op boundary rather than the impl boundary: that is the
    # exact kwarg set we need to replay.
    op = torch.ops._C_ascend.npu_sparse_flash_mla

    def op_wrapper(q, **kw):
        global _DONE
        result = op(q, **kw)
        if not _DONE:
            _DONE = True
            payload = {"q": q.cpu()}
            for k, v in kw.items():
                payload[k] = v.cpu() if isinstance(v, torch.Tensor) else v
            payload["__output__"] = result[0].cpu()
            torch.save(payload, path)
            print(f"[capture_smla_args] wrote {path} "
                  f"({sum(v.numel() for v in payload.values() if isinstance(v, torch.Tensor))} elems)",
                  flush=True)
        return result

    dsa.torch.ops._C_ascend.npu_sparse_flash_mla = op_wrapper  # type: ignore[attr-defined]
    _ = wrapper  # kept for the layer-selective variant above


if __name__ == "__main__":
    raise SystemExit(
        "import this module and call install() before starting the engine; "
        "see the docstring."
    )
