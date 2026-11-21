#!/usr/bin/env bash
# Launch the mask (reference) and compact (candidate) services from ONE script
# with exactly one flag different, so L2's comparison is meaningful.
#
# Usage: bash run_both_modes.sh /path/to/weights
#
# Everything below is the official A3 single-node recipe, unchanged, except the
# two lines marked BOUNDED REPLAY. Keep it that way: any other difference
# between the two services shows up in L2 as a false failure.
set -euo pipefail
MODEL_PATH="${1:?usage: bash run_both_modes.sh <MODEL_PATH>}"

serve() {   # serve <mode> <port> <devices>
  local mode="$1" port="$2" devices="$3"
  ASCEND_RT_VISIBLE_DEVICES="$devices" \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  VLLM_ENGINE_READY_TIMEOUT_S=36000 \
  HCCL_BUFFSIZE=1024 \
  HCCL_OP_EXPANSION_MODE=AIV \
  vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$port" \
    --max-model-len 150000 --max-num-batched-tokens 8192 --max-num-seqs 32 \
    --served-model-name dsv41 --gpu-memory-utilization 0.9 \
    --data-parallel-size 2 --tensor-parallel-size 4 --enable-expert-parallel \
    --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --safetensors-load-strategy lazy --quantization ascend --block-size 128 \
    --no-enable-prefix-caching \
    --seed 1234 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config "{
        \"enable_cpu_binding\":true,
        \"enable_fused_mc2\":1,
        \"enable_dsa_cp\":true,
        \"enable_flashcomm1\":true,
        \"enable_shared_expert_dp\":true,
        \"enable_decoder_swa_bounded_replay\":true,
        \"decoder_swa_replay_mode\":\"$mode\"
    }" > "serve_${mode}.log" 2>&1 &
  echo "  $mode -> port $port, devices $devices, log serve_${mode}.log (pid $!)"
}

# Two DP2/TP4 services side by side on one 16-device node. NOTE this is NOT the
# DP4/TP4 production topology -- it is halved so both can be resident at once.
# L2 compares the two modes against each other, so the topology only has to be
# IDENTICAL between them, not identical to production.
echo "launching both modes (DP2/TP4 each):"
serve mask    8900 "0,1,2,3,4,5,6,7"
serve compact 8901 "8,9,10,11,12,13,14,15"
echo
echo "wait for 'Application startup complete' in both logs, then:"
echo "  python3 -I l2_engine_equiv.py --mask http://127.0.0.1:8900 --compact http://127.0.0.1:8901"
wait
