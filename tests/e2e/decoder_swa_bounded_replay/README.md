# Decoder SWA Bounded Replay for DeepSeek-V4.1-Flash (A3)

> ## ⚠ NOT HARDWARE-VERIFIED
>
> Nothing on this branch has been compiled or run. The operator has not been
> built, no test has been executed, and no accuracy or performance number has
> been measured. It is a design + implementation draft, written from source
> analysis of three reference implementations.
>
> **Do not treat any claim here as measured.** The checks in
> `tests/e2e/decoder_swa_bounded_replay/` exist precisely because the design
> rests on assumptions that only hardware can settle; the five open build and
> runtime questions are listed at the bottom of this file.

## What this is

DeepSeek-V4.1-Flash uses a **Causal Encoder-Decoder (CED)** architecture: the
decoder half's global KV is projected from the final encoder hidden state, not
from each decoder layer's own. Prefill therefore only has to run layers
`[0, boundary]` over every prompt token; the layers above the boundary need to
run only over a bounded tail, because the one thing they do own is a 128-wide
sliding window.

The paper (arXiv 2609.19969 §3.2.2) calls this **Decoder SWA Bounded Replay**
and reports that it "nearly halves prefill computation".

**This is not a new idea.** It is the model's documented design, and it is
already implemented by:

- **DeepInfer** (DeepSeek's own engine) — `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`
- **SGLang** — `enable_decoder_swa_bounded_replay`, **default on**

vLLM, and therefore vllm-ascend, implements the CED *caching* half (consumer
layers read the boundary layer's compressed KV) but not the *compute-saving*
half: `models/deepseek_v41/nvidia/model.py` and vllm-ascend's
`models/deepseek_v41/model.py` both iterate all 40 layers over all prompt
tokens.

## Scope of this branch: `mask` mode only

| mode | drops query rows | narrows visibility | saves compute |
|---|---|---|---|
| `off` | no | no | — |
| **`mask`** (this branch) | **no** | **yes** | **no** |
| `compact` (future) | yes | yes | yes |

`mask` is deliberately the first step. It pins the semantics and is the
bit-exact reference against which `compact` is later validated — DeepInfer's
own harness (`examples/yoco/yoco_equiv.py`) validates `compact` by comparing
it to `mask` with **zero tolerance**. Shipping `compact` first would leave no
way to tell a row-selection bug from a visibility bug.

So: **expect flat throughput from this branch.** A large regression would mean
the patch added overhead, not that the optimization underperformed.

## Design

Boundary layer = `max(kv_source_layer_ids)` = 20 on the released checkpoint,
cross-checked against the first `compress_ratio == 1` layer (DeepInfer's
`n_prefill_layer`). The two derivations coincide by constraint, not by
definition, so `build_layer_plan` raises when they disagree rather than
picking one.

For a request whose full prompt length is `N` and a retained tail of
`keep_tokens`, define `rect_lo = max(0, N - keep_tokens)`. In consumer layers a
query at absolute position `i` may attend sliding-window keys only in

```
[ max(i - W + 1, rect_lo), i ]        W = sliding_window = 128
```

`rect_lo` is anchored on the **full prompt length**, which is chunk-independent
— so chunked prefill needs no "is this the last chunk" flag, exactly as in
DeepInfer.

`keep_tokens` defaults to `sliding_window + num_speculative_tokens + 1`
(129 plain, 134 with DSpark-5). The margin is not cosmetic: a rejected
speculative token rolls the sequence back, and every rolled-back position still
needs a full window of keys.

### The kernel change

On A3 the sliding-window bound is a **scalar operator attribute**
(`ori_win_left`), and the kernel's only lower clamp is `0`. The explicit-index
route is unavailable: consumer layers always carry `cmp_kv`, which selects
`ORI_CMP_SPARSE_TEMPLATE`, and that template is not compiled for A2/A3.

So this branch adds an optional per-request INT32 input `ori_win_floor [B]` to
the in-tree `SparseFlashMla` operator:

```cpp
__aicore__ inline int32_t SparseFlashMlaCsa<SMLAT>::GetOriWinFloor(uint32_t bIdx, int32_t oriMaskRight)
{
    if (constInfo.oriWinFloorDims == 0) {
        return 0;                                        // absent -> bit-identical to before
    }
    return Min(oriWinFloorGm.GetValue(bIdx), oriMaskRight);
}
```

The `Min` is load-bearing. Without it a query below the floor gets
`oriMaskLeft > oriMaskRight`, and `uint32_t oriS2Size = oriMaskRight -
oriMaskLeft + 1` wraps around; `UpdateInnerLoopCond` does not catch that case.
Those rows instead get a degenerate single-key window, which is harmless: in
`compact` mode they will not exist, and nothing above the floor can read them
because attention is the only cross-token path.

`constInfo.mBaseSize = constInfo.gSize` on this path, so a CSA M-block is one
query row and the per-request floor is evaluated per query.

A3's `sliding_window = 128` gives a 128-key window against a fixed
`s2BaseSize = 512`, so `oriSplitNum = 1` with or without the floor and the
AICPU metadata operator needs no change. **This would not hold for
`sliding_window > 512`.**

## Usage

```
--additional-config '{"enable_decoder_swa_bounded_replay": true,
                      "decoder_swa_replay_mode": "mask"}'
```

Refused (raise, never silently degrade): prefix caching, the vision path,
`decoder_swa_replay_mode="compact"`. Models without a CED topology downgrade
with a warning.

## Tests

| level | needs | what |
|---|---|---|
| `tests/ut/models/test_bounded_replay_plan.py` | nothing | boundary derivation; the "no consumer owns compressed KV" premise; ambiguous topologies rejected |
| `tests/e2e/decoder_swa_bounded_replay/test_ori_win_floor.py` | NPU | capture-and-replay of one real operator call, K1–K5 |
| `tests/e2e/decoder_swa_bounded_replay/test_e2e_mask_mode.py` | NPU + weights | service health, bounded divergence, no throughput regression |

The operator test replays a **captured** call rather than synthesising tensors:
the TND / PA_BBND layouts, block tables and `cu_seqlens` are easy to get subtly
wrong by hand, and a wrong layout yields an opaque EZ error instead of a useful
failure.

**K4 is the important one.** K1–K3 (absent floor unchanged, floor 0 a no-op,
out-of-range floor safe) would all pass for a patch that silently ignores the
new tensor. Only K4 — an in-range floor must change the output — catches that.

## Open questions, for whoever compiles this first

1. **Tiling-data capacity.** `SparseFlashMlaSwaParams` grows 4 bytes;
   `SaveToBuffer(..., GetCapacity())` fails at runtime if the registered
   capacity is now too small. Appending at the end preserves every existing
   field offset, so arch35 should be unaffected.
2. **Generated aclnn header.** `aclnnSparseFlashMlaGetWorkspaceSize` is
   generated from the prototype; confirm the new input lands between
   `metadata` and `softmaxScale`, and reorder the `EXEC_NPU_CMD` call if not.
3. **`-Werror` on the A5 compile.** `(void)oriWinFloor;` was added in the 310
   branch and in the SWA `Init`; confirm no unused-parameter warning survives.
4. **Stale installed operator package.** Reinstall the `.run` after every
   rebuild — a new `.so` against an old package fails with an EZ-series
   argument-count error at runtime, not a link error.
5. **Index frame.** `ori_win_floor[b]` is absolute within
   `[0, seqused_ori_kv[b])`. A mismatch is silent — wrong attention, no error.
   K4 and K5 are the only signals that it is right.

## References

- DeepSeek-V4.1-Flash, arXiv 2609.19969, §2.2 (CED) and §3.2.2 (SWA Bounded Replay)
- DeepInfer — `deepseek-ai/DeepInfer`, `DEEPINFER_YOCO_PREFILL_*`
- SGLang — `sgl-project/sglang`, `python/sglang/srt/models/deepseek_v4.py`
- YOCO — Sun et al., 2024 (arXiv 2405.05254), which the paper cites as the inspiration
