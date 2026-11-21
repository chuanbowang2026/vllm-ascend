#!/usr/bin/env python3
"""Stage-0 probe: verify the structural claims the Bounded Replay design rests on.

Pure arithmetic over the checkpoint's config.json. Needs NO NPU and NO weights
(only config.json, ~3 KB). Runs anywhere, including a laptop.

Verifies claims A3-A6 from the deliverable README:
  A3  KV pool page sizes == [131072]*3 + [147712], pool == 540928 B/block
  A4  useful 2885 B/token vs allocated 4226 B/token
  A5  backbone ~551.6B params, ~377.3M active per token per layer
  A6  boundary layer == max(kv_source_layer_ids) == 20, and layers > 20 own
      NO compressed KV / index cache / compressor ring  <-- the whole premise

If A6 fails, the Decoder SWA Bounded Replay design is invalid and must be redone.

Usage:
  python3 -I 03_probe_model.py --config /path/to/DeepSeek-V4.1-Flash/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

# ---- expectations, from the no-hardware analysis (docs/01, docs/05) --------
EXPECT = {
    "num_hidden_layers": 40,
    "hidden_size": 5120,
    "num_attention_heads": 64,
    "head_dim": 512,
    "sliding_window": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2304,
    "index_head_dim": 128,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "boundary_layer": 20,
    "page_sizes": [131072, 131072, 131072, 147712],
    "pool_bytes_per_block": 540928,
    "bytes_per_token_allocated": 4226.0,
    "bytes_per_token_useful": 2885.0,
    "backbone_params_B": 551.6,
    "active_per_layer_M": 377.3,
}

STATE_RING_ROWS = 32  # vllm_ascend/models/deepseek_v41/cache_config.py:27
BF16 = 2
FP32 = 4
INT8 = 1
FP16 = 2


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    # DeepseekV41Config flattens text_config onto the top level.
    cfg = dict(raw.get("text_config") or {})
    for k, v in raw.items():
        if k not in ("text_config", "vision_config"):
            cfg.setdefault(k, v)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to the checkpoint's config.json")
    ap.add_argument("--block-size", type=int, default=128, help="logical KV block size (--block-size)")
    ap.add_argument("--max-model-len", type=int, default=150000)
    ap.add_argument("--hbm-per-die-gb", type=float, default=64.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--num-logical-devices", type=int, default=16)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    rep = Report("03_probe_model", "layer plan / KV pool / parameter count from config.json")

    if not os.path.exists(args.config):
        rep.check("config_exists", None, actual=args.config, detail="config.json not found")
        return rep.emit(args.results_dir)

    cfg = load_cfg(args.config)

    # ---- basic dims --------------------------------------------------------
    D = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    dh = cfg["head_dim"]
    Lq = cfg["q_lora_rank"]
    Lo = cfg["o_lora_rank"]
    G = cfg["o_groups"]
    E = cfg["n_routed_experts"]
    k = cfg["num_experts_per_tok"]
    Ns = cfg["n_shared_experts"]
    Im = cfg["moe_intermediate_size"]
    V = cfg["vocab_size"]
    M = cfg["hc_mult"]
    W = cfg["sliding_window"]
    Hi = cfg["index_n_heads"]
    di = cfg["index_head_dim"]
    ratios = cfg["compress_ratios"]
    kv_src = list(cfg["kv_source_layer_ids"])
    idx_src = list(cfg["index_source_layer_ids"])

    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "head_dim",
                "sliding_window", "n_routed_experts", "num_experts_per_tok",
                "n_shared_experts", "moe_intermediate_size", "index_head_dim"):
        rep.check(f"cfg.{key}", cfg.get(key) == EXPECT[key], EXPECT[key], cfg.get(key))
    rep.check("cfg.kv_source_layer_ids", kv_src == EXPECT["kv_source_layer_ids"],
              EXPECT["kv_source_layer_ids"], kv_src)

    # ---- A6: the premise of the whole optimization -------------------------
    boundary = max(kv_src)
    rep.check("A6.boundary_layer", boundary == EXPECT["boundary_layer"],
              EXPECT["boundary_layer"], boundary, critical=True,
              detail="boundary = max(kv_source_layer_ids); layers >= boundary are the 'late'/consumer half")

    # compress_ratios may carry trailing draft layers (num_nextn_predict_layers)
    backbone_ratios = ratios[:L]
    enc = [i for i in range(L) if i < boundary]
    dec = [i for i in range(L) if i >= boundary]
    rep.fact("encoder_layers", f"{len(enc)} (0..{boundary - 1})")
    rep.fact("decoder_layers", f"{len(dec)} ({boundary}..{L - 1})")
    rep.check("A6.half_split", len(enc) == len(dec) == L // 2, f"{L//2}/{L//2}",
              f"{len(enc)}/{len(dec)}",
              detail="paper says 20 causal-encoder + 20 decoder layers")

    # ratio transition must coincide with the boundary
    r_at = {i: backbone_ratios[i] for i in range(L)}
    first_r1 = next((i for i in range(L) if r_at[i] == 1), None)
    rep.check("A6.ratio_transition_at_boundary", first_r1 == boundary, boundary, first_r1,
              detail="DeepInfer derives n_prefill_layer as the first layer with compress_ratio == 1")

    # layers above the boundary must own NO compressed KV / index / ring
    owners_kv = [i for i in dec if i in kv_src]
    owners_idx_cache = [i for i in dec if i in kv_src]  # index k_cache gated on is_kv_source
    rings = [i for i in range(L) if i in kv_src and r_at[i] == 2]
    rep.check("A6.no_kv_owner_above_boundary", owners_kv == [boundary], [boundary], owners_kv,
              critical=True,
              detail="only the boundary layer itself may own compressed KV in the decoder half")
    rep.check("A6.no_ring_above_boundary", all(i < boundary for i in rings),
              f"all < {boundary}", rings,
              detail="compressor ring exists only for ratio==2 kv-source layers")
    idx_consumers = [i for i in idx_src if i > boundary]
    rep.fact("index_consumers_above_boundary", idx_consumers)
    rep.note("index-source layers above the boundary (24/28/32/36) are consumers only: "
             "owns_k = is_kv_source is False for them, so they hold no persistent index cache.")

    # ---- A3/A4: KV pool page sizes (non-A5 / BF16 path) --------------------
    BS = args.block_size

    def mla_page(r: int) -> int:
        return (BS // r) * dh * BF16

    def idx_page(r: int) -> int:
        return (BS // r) * (di * INT8 + 1 * FP16)

    swa_page = BS * dh * BF16
    state_page = STATE_RING_ROWS * (2 * dh) * FP32

    page_sizes, needs = [], []
    for slot, layer in enumerate(sorted(kv_src)):
        r = r_at[layer]
        need = mla_page(r) + idx_page(r)
        aliases = [swa_page] + ([state_page] if r == 2 else [])
        page_sizes.append(max(need, *aliases))
        needs.append(need)

    pool = sum(page_sizes)
    useful = sum(needs)
    rep.check("A3.page_sizes", page_sizes == EXPECT["page_sizes"],
              EXPECT["page_sizes"], page_sizes,
              detail=f"swa_page={swa_page}, state_page={state_page}")
    rep.check("A3.pool_bytes_per_block", pool == EXPECT["pool_bytes_per_block"],
              EXPECT["pool_bytes_per_block"], pool)

    bpt_alloc = pool / BS
    bpt_useful = useful / BS
    rep.check("A4.bytes_per_token_allocated", abs(bpt_alloc - EXPECT["bytes_per_token_allocated"]) < 1,
              EXPECT["bytes_per_token_allocated"], round(bpt_alloc, 1))
    rep.check("A4.bytes_per_token_useful", abs(bpt_useful - EXPECT["bytes_per_token_useful"]) < 1,
              EXPECT["bytes_per_token_useful"], round(bpt_useful, 1))
    rep.fact("kv_padding_waste_pct", round(100 * (pool - useful) / pool, 2))

    # ---- A5: parameter counts ---------------------------------------------
    attn = D * Lq + Lq * H * dh + D * dh + (H * dh // G) * (G * Lo) + G * Lo * D
    expert = 3 * D * Im
    moe_all = E * expert + Ns * expert + D * E
    hc = 2 * (M * D) * ((2 + M) * M)
    per_layer = attn + moe_all + hc
    idx_proj = len(idx_src) * (Lq * Hi * di + D * Hi)
    comp = sum(2 if r_at[i] == 2 else 1 for i in kv_src) * D * dh
    backbone = L * per_layer + 2 * V * D + idx_proj + comp

    active_layer = attn + (k + Ns) * expert + D * E + hc
    rep.check("A5.backbone_params_B", abs(backbone / 1e9 - EXPECT["backbone_params_B"]) < 2.0,
              EXPECT["backbone_params_B"], round(backbone / 1e9, 1),
              detail="paper abstract says 552B backbone parameters")
    rep.check("A5.active_per_layer_M", abs(active_layer / 1e6 - EXPECT["active_per_layer_M"]) < 5.0,
              EXPECT["active_per_layer_M"], round(active_layer / 1e6, 1))

    act_enc = (boundary + 1) * active_layer / 1e9   # layers 0..boundary inclusive
    act_all = L * active_layer / 1e9
    rep.fact("active_params_prefill_B", f"{act_enc:.2f}  (paper: ~8B)")
    rep.fact("active_params_decode_B", f"{act_all:.2f}  (paper: ~16B)")
    rep.check("A5.matches_8B_claim", abs(act_enc - 8.0) < 1.0, "~8B", round(act_enc, 2),
              detail="layers 0..20 inclusive is what prefill must still compute for every token")
    rep.check("A5.matches_16B_claim", abs(act_all - 16.0) < 1.5, "~16B", round(act_all, 2))

    # ---- expected speedup (DeepInfer FlopsEstimator form) ------------------
    # saving = f * (1 - keep_tokens / N), f = consumer half's share of prefill FLOPs -> ~1/2
    f = (L - boundary) / L
    rep.fact("consumer_share_f", round(f, 4))
    for keep in (129, 134, 140):
        for N in (4096, 32768, 150000):
            rep.fact(f"expected_saving_keep{keep}_N{N}", f"{100 * f * (1 - keep / N):.1f}%")

    # ---- capacity ----------------------------------------------------------
    free = args.hbm_per_die_gb * 1e9 * args.gpu_memory_utilization - backbone / args.num_logical_devices
    blocks_per_seq = -(-args.max_model_len // BS) + 10 * 2 + 1  # MLA + 10 SWA groups + 1 state
    bytes_per_seq = blocks_per_seq * pool
    rep.fact("free_bytes_per_die_GB", round(free / 1e9, 1))
    rep.fact("blocks_per_seq", blocks_per_seq)
    rep.fact("MiB_per_seq", round(bytes_per_seq / 2**20, 1))
    rep.fact("max_seqs_per_dp_engine", round(free / bytes_per_seq, 1))
    rep.note("compare max_seqs_per_dp_engine against --max-num-seqs 32: if the ratio is near 1, "
             "the recipe sits on a recompute-preemption cliff.")

    return rep.emit(args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
