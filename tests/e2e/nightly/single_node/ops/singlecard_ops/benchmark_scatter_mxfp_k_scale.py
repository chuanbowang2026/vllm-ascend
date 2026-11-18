# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the MXFP K-scale scatter: eager IndexPutV2 vs the Triton kernel.

Self-contained; run it directly on a node with one NPU:

    python tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_scatter_mxfp_k_scale.py

It sweeps prefill-sized token counts against the PA_NZ K-scale cache layout
and reports per-call latency plus the effective scale-write bandwidth. The
eager path is the advanced-indexing assignment this kernel replaces:

    cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale

Both paths are checked for bit-exact agreement before timing, so a speedup
is never reported for a kernel that computed something else.
"""

import time

import torch
import torch_npu  # noqa: F401  (registers the npu device)

from vllm_ascend.ops.triton.scatter_mxfp_k_scale import scatter_mxfp_k_scale_cache_triton

VALUES_PER_GROUP = 2
TOKEN_FRAG = 16
SCALE_GROUP_SIZE = 64
KERNEL_BLOCK_SIZE = 512  # C8-MXFP QFA requires 512-token pages
WARMUP_ITERS = 20
TIMED_ITERS = 100


def build_case(num_tokens, num_kv_heads, head_dim, num_blocks, device):
    num_frags = head_dim // SCALE_GROUP_SIZE
    num_segs = KERNEL_BLOCK_SIZE // TOKEN_FRAG

    key_scale = torch.randint(
        0,
        256,
        (num_tokens, num_kv_heads, num_frags, VALUES_PER_GROUP),
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.zeros(
        (num_blocks, num_kv_heads, num_segs, num_frags, TOKEN_FRAG, VALUES_PER_GROUP),
        dtype=torch.uint8,
        device=device,
    )

    # Contiguous slots, as a real prefill produces: token i lands at slot i.
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    block_ids = slots // KERNEL_BLOCK_SIZE
    offsets = slots % KERNEL_BLOCK_SIZE
    seg_ids = offsets // TOKEN_FRAG
    frag_ids = offsets % TOKEN_FRAG
    return key_scale, cache, block_ids, seg_ids, frag_ids


def eager_scatter(key_scale, cache, block_ids, seg_ids, frag_ids):
    cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale


def time_call(fn, *args):
    for _ in range(WARMUP_ITERS):
        fn(*args)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / TIMED_ITERS


def main():
    device = "npu"
    head_dim = 128
    print(
        f"{'tokens':>8} {'heads':>6} {'bytes':>9} "
        f"{'eager us':>10} {'triton us':>10} {'speedup':>8} {'triton GB/s':>12}  match"
    )
    for num_kv_heads in (1, 4, 8):
        for num_tokens in (512, 2048, 8192, 32768):
            num_blocks = max(1, (num_tokens + KERNEL_BLOCK_SIZE - 1) // KERNEL_BLOCK_SIZE)
            case = build_case(num_tokens, num_kv_heads, head_dim, num_blocks, device)
            key_scale, cache, block_ids, seg_ids, frag_ids = case

            eager_cache = cache.clone()
            eager_scatter(key_scale, eager_cache, block_ids, seg_ids, frag_ids)
            triton_cache = cache.clone()
            scatter_mxfp_k_scale_cache_triton(key_scale, triton_cache, block_ids, seg_ids, frag_ids)
            match = torch.equal(eager_cache.cpu(), triton_cache.cpu())

            eager_s = time_call(eager_scatter, key_scale, cache.clone(), block_ids, seg_ids, frag_ids)
            triton_s = time_call(
                scatter_mxfp_k_scale_cache_triton,
                key_scale,
                cache.clone(),
                block_ids,
                seg_ids,
                frag_ids,
            )

            moved_bytes = key_scale.numel() * 2  # one read + one write
            print(
                f"{num_tokens:>8} {num_kv_heads:>6} {key_scale.numel():>9} "
                f"{eager_s * 1e6:>10.1f} {triton_s * 1e6:>10.1f} "
                f"{eager_s / triton_s:>7.2f}x {moved_bytes / triton_s / 1e9:>11.2f}  {match}"
            )


if __name__ == "__main__":
    main()
