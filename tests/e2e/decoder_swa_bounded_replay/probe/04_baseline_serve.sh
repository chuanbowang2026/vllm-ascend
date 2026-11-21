#!/usr/bin/env bash
# Stage-0: launch the OFFICIAL A3 single-node colocated recipe, unmodified.
#
# This is the baseline every later measurement is compared against, so do not
# edit the flags. It is a verbatim transcription of
#   docs/source/tutorials/models/DeepSeek-V4.1-Flash.md  section 5.1 (A3 series)
# from vllm-ascend releases/v0.30.0rc2.
#
# Runs in the FOREGROUND. Wait for "Application startup complete", then run
# 05_bench_prefill.py from another shell.
#
# Usage: bash 04_baseline_serve.sh /path/to/DeepSeek-V4.1-Flash-w8a8 [port]

set -euo pipefail

MODEL_PATH="${1:?usage: bash 04_baseline_serve.sh <MODEL_PATH> [PORT]}"
PORT="${2:-8900}"
RESULTS_DIR="${PROBE_RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/04_baseline_serve.log"

if [ ! -d "$MODEL_PATH" ]; then
  echo "INCONCLUSIVE"
  echo "# model path does not exist: $MODEL_PATH"
  exit 2
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "# baseline serve: $MODEL_PATH  port=$PORT"
echo "# log -> $LOG"
echo "# after startup, in another shell:"
echo "#   python3 -I 05_bench_prefill.py --url http://127.0.0.1:$PORT --model dsv41"
echo "#"
echo "# while it runs, these log greps answer stage-0 questions:"
echo "#   grep -i 'cpu_bind_mode'                  -> did CPU binding apply? (claim A7)"
echo "#   grep -i 'enable_npugraph_ex is enabled'  -> is torch.compile reached? (claim A2)"
echo "#   grep -i 'Disabling DSA-CP'               -> was DSA-CP silently force-disabled?"
echo "#   grep -i 'MegaMoe is not supported'       -> did fused_mc2 fall back?"
echo ""

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 150000 \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv41 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --quantization ascend \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "ascend_compilation_config":{
            "enable_npugraph_ex":true,
            "enable_static_kernel":false
        },
        "enable_cpu_binding":true,
        "enable_fused_mc2":1,
        "enable_dsa_cp":true,
        "enable_flashcomm1":true,
        "enable_shared_expert_dp":true
    }' 2>&1 | tee "$LOG"
