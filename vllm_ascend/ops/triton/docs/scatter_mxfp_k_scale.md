# scatter_mxfp_k_scale

## Description

- **Function**: Scatters the per-token MXFP K scales produced by `reshape_and_cache` into the paged PA_NZ K-scale cache of the C8-MXFP (QFA) attention backend. It replaces the eager advanced-indexing assignment `key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale` in `vllm_ascend/attention/attention_c8_mxfp.py`, which runs once per full-attention layer per step.

- **Formula**: Pure data movement (load/store only, no arithmetic). For every token `t`, KV head `n`, scale fragment `f` and byte `j`:

    ```text
    key_scale_cache[block_ids[t], n, seg_ids[t], f, frag_ids[t], j] = key_scale[t, n, f, j]
    ```

    where a token at in-block offset `o` lands at `[block, n, o // 16, f, o % 16, j]`, with `block = slot // block_size` and `o = slot % block_size`. The decomposition `(block_ids, seg_ids, frag_ids)` is derived once per step by `mxfp_k_scale_slot_index` and shared across the step's layers.

    Layouts:
    - Input `key_scale`: `[num_tokens, N, F, 2]`, dense
    - In-place output `key_scale_cache` (PA_NZ): `[num_blocks, N, S, F, 16, 2]`
    - `N = num_kv_heads`, `F = head_dim // 64`, `S = block_size // 16`, trailing axis = 2 E8M0 bytes per 64-element group

- **Algorithm flow** (processed row by row, independently):
  1. Bitcast both tensors from their 1-byte dtype to 16-bit and drop the trailing axis. Source and cache both end in a `size=2, stride=1` byte pair, so this is a pure reinterpretation that halves the element count and doubles the access granularity. The kernel then performs `dst[block, n, seg, f, frag] = src[t, n, f]`. A layout that cannot be bitcast raises `ValueError`, and the caller falls back to the eager assignment.
  2. Derive the launch geometry on the host: `payload = N * F`, `PAYLOAD_P2 = next_pow2(payload)`, `BLOCK_T = clamp(4096 // PAYLOAD_P2, 1, 256)` (a 4096-element tile is 8 KiB, inside the smallest shipping UB), `grid = cdiv(num_tokens, BLOCK_T)`.
  3. Each program handles a `[BLOCK_T, PAYLOAD_P2]` tile. Token indices are promoted to int64 before any address multiplication so a large cache cannot overflow a 32-bit intermediate. Out-of-range tokens read index 0 and are masked out of the store.
  4. Load the three index vectors once per token row (`block`, `seg`, `frag`), broadcast along the payload axis.
  5. Decompose the flat payload index into `(head, frag_dim) = (p // F, p % F)`; `F` is a `constexpr`, so a power-of-two fragment count compiles to shifts. Lanes with `p >= payload` are masked.
  6. Compute source and destination offsets from the tensors' own strides (in elements of the 16-bit view) and perform one masked load and one masked store. Nothing is materialised.

- **Supported modes**: The kernel is generic Triton data movement and carries no device-specific intrinsics. The C8-MXFP path that calls it is A5-only in practice: it is gated on `cache_config.cache_dtype == "mxfp8"`, requires 512-token kernel blocks (`A5_C8_MXFP_KV_CACHE_BLOCK_SIZE`), and its QFA operator accepts `FLOAT8_E4M3FN` only on 950PR/950DT. Works in both eager and graph-capture modes: the launch performs no host-device sync and has no data-dependent shapes.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `key_scale` | Input | Per-token K scales `[num_tokens, num_kv_heads, head_dim // 64, 2]`; callers pass a uint8 view of the E8M0 scales | uint8 (any 1-byte dtype) | ND |
| `key_scale_cache` | Input/Output (in-place) | Paged PA_NZ K-scale cache `[num_blocks, num_kv_heads, block_size // 16, head_dim // 64, 16, 2]` | uint8 (any 1-byte dtype) | ND (PA_NZ, may be a strided view with a page gap) |
| `block_ids` | Input | Destination kernel-block index per token, `slot // block_size` | int64 | ND, `[num_tokens]` |
| `seg_ids` | Input | Destination token-group index per token, `(slot % block_size) // 16` | int64 | ND, `[num_tokens]` |
| `frag_ids` | Input | Destination in-group token index per token, `(slot % block_size) % 16` | int64 | ND, `[num_tokens]` |

## Constraints

- `key_scale.shape[0] == block_ids.numel() == seg_ids.numel() == frag_ids.numel()`; an empty input returns without launching.
- Both tensors must have a 1-byte dtype and a trailing axis of exactly `size=2, stride=1`. All remaining strides and the storage offset must be even, so every axis lands on a 16-bit boundary. Any violation raises `ValueError` and the caller falls back to the eager assignment; every other failure propagates.
- `key_scale_cache.shape[1]` and `.shape[3]` must equal `key_scale.shape[1]` and `.shape[2]` (head and fragment axes).
- `head_dim % 64 == 0` and `block_size % 16 == 0`, inherited from the C8-MXFP cache layout. `F` need not be a power of two (`head_dim = 192` gives `F = 3`); the masked payload tail handles it.
- **Destination slots must be unique.** This is a caller precondition, identical to the requirement of the advanced-indexing assignment it replaces, and is not checked because checking needs a device-to-host sync.
- Padded rows arrive already clamped to slot 0 by `mxfp_k_scale_slot_index` and are written to block 0, the null block `BlockPool` never hands to a request. They are deliberately not masked, which keeps every shape static and the launch capture-safe.
- Cache and source storage must not overlap. Concurrent writers are unsupported.
- `num_tokens` is dynamic; `F`, `PAYLOAD`, `PAYLOAD_P2` and `BLOCK_T` are compile-time `constexpr`, and strides are specialised per layout.

## Origin and Differences

- **Origin**: Developed from scratch. There is no upstream vLLM operator for this write; the code it replaces is a plain PyTorch advanced-indexing assignment introduced with the C8-MXFP backend.
- **Differences**:
    - NPU adaptation for performance: the eager form places three advanced indices on dims 0/2/4, separated by basic slices on dims 1/3/5. Torch must broadcast them into a leading result axis and materialise a permuted intermediate before handing the write to `IndexPutV2`; on float8 inputs that operator falls back to AICPU entirely, which is why the existing call site already passes a uint8 view. Each token also carries only `N * F * 2` bytes (32 bytes at `head_dim = 128`, `N = 8`), so the write is pure fragmented scatter. The Triton kernel computes every destination address directly, materialises no intermediate, and raises the access granularity to 16 bits by bitcasting the trailing byte pair away.
    - Modified for a specific vllm-ascend logic: consumes the `(block_ids, seg_ids, frag_ids)` decomposition that `mxfp_k_scale_slot_index` derives once per step and shares across layers, and preserves the null-block destination for clamped padded rows so graph capture sees static shapes.

## Test Cases

As this is a pure data-movement operator (no arithmetic), the unified precision tolerance is bit-exact (`rtol=0, atol=0`); the NPU test compares against the eager advanced-indexing assignment it replaces.

The accuracy test uses `block_size = 512`, the kernel block size the C8-MXFP backend mandates (`get_supported_kernel_block_sizes() == [512]`), and sweeps `num_tokens ∈ {1, 15, 512, 4096}`, `num_kv_heads ∈ {1, 4, 8}` (per TP rank) and `head_dim ∈ {64, 128, 192}`, i.e. `F ∈ {1, 2, 3}` including a non-power-of-two fragment count. It also covers a strided cache whose pages carry a gap, which is what a real PA_NZ allocation looks like. When a concrete deployment model is fixed, add its exact `(num_kv_heads, head_dim)` per TP rank as a dedicated case.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_scatter_mxfp_k_scale.py
```

A self-contained benchmark comparing the kernel against the eager path is available alongside it; it verifies both paths agree bit-exact before reporting any timing.

```bash
python tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_scatter_mxfp_k_scale.py
```

Host-side launch geometry, the bitcast contract and the address arithmetic are additionally covered without hardware:

```bash
pytest -sv tests/ut/ops/test_scatter_mxfp_k_scale.py
```
