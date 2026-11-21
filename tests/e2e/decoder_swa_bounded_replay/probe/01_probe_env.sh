#!/usr/bin/env bash
# Stage-0 probe: environment facts. Read-only, changes nothing.
# Verdict on the first line; exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE.
#
# Usage: bash 01_probe_env.sh   (inside the official container, NPU visible)

set -uo pipefail
RESULTS_DIR="${PROBE_RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/01_probe_env.txt"
: > "$OUT"

FAILED=0
UNKNOWN=0

say() { printf '%s\n' "$*" >> "$OUT"; }
sec() { say ""; say "## $*"; }

# run <label> <cmd...> — capture output, mark UNKNOWN if the command fails
run() {
  local label="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    say "### $label"
    printf '%s\n' "$out" | sed 's/^/    /' >> "$OUT"
  else
    say "### $label  [UNAVAILABLE]"
    UNKNOWN=$((UNKNOWN + 1))
  fi
}

sec "identity"
say "    date         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "    hostname     : $(hostname)"
say "    uname        : $(uname -srm)"
say "    container    : $( [ -f /.dockerenv ] && echo yes || echo 'no/unknown' )"

sec "npu devices"
run "npu-smi info" npu-smi info
say "    /dev/davinci*: $(ls -1 /dev/davinci* 2>/dev/null | tr '\n' ' ')"
DAVINCI_COUNT=$(ls -1d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
say "    davinci count: $DAVINCI_COUNT  (expect 16 for a full A3 node)"
if [ "${DAVINCI_COUNT:-0}" -ne 16 ]; then
  say "    [WARN] not 16 logical devices — the official DP4/TP4 recipe assumes 16"
  UNKNOWN=$((UNKNOWN + 1))
fi

sec "driver / cann"
run "driver version.info" cat /usr/local/Ascend/driver/version.info
run "ascend_install.info" cat /etc/ascend_install.info
say "    ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-<unset>}"
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "$ASCEND_HOME_PATH/version.cfg" ]; then
  run "CANN version.cfg" cat "$ASCEND_HOME_PATH/version.cfg"
fi

sec "cpu / numa"
NPROC=$(nproc 2>/dev/null || echo 0)
say "    nproc            : $NPROC"
for f in /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/cpuset/cpuset.cpus; do
  [ -f "$f" ] && say "    cpuset ($f): $(cat "$f")"
done
say "    note: vllm_ascend cpu_binding needs >= 5 CPUs per logical device (80 for 16)"
run "lscpu (head)" sh -c 'lscpu | head -20'
run "numactl -H" numactl -H

sec "memory"
say "    MemTotal     : $(awk '/MemTotal/{print $2, $3}' /proc/meminfo 2>/dev/null)"
say "    HugePages_Total: $(awk '/HugePages_Total/{print $2}' /proc/meminfo 2>/dev/null)"
say "    /dev/shm     : $(df -h /dev/shm 2>/dev/null | tail -1)"
say "    note: --engram-config dp_shared_memory needs a shared IPC namespace (--ipc=host)"

sec "network (hccl)"
run "hccn.conf" cat /etc/hccn.conf
say "    HCCL_IF_IP            : ${HCCL_IF_IP:-<unset>}"
say "    HCCL_BUFFSIZE         : ${HCCL_BUFFSIZE:-<unset>}"
say "    HCCL_OP_EXPANSION_MODE: ${HCCL_OP_EXPANSION_MODE:-<unset>}"

sec "python stack"
run "python version" python3 -c 'import sys; print(sys.version)'
run "torch / torch_npu" python3 -I -c 'import torch,torch_npu;print("torch",torch.__version__);print("torch_npu",torch_npu.__version__)'
run "vllm / vllm_ascend" python3 -I -c 'import vllm,vllm_ascend;print("vllm",getattr(vllm,"__version__","?"));print("vllm_ascend",getattr(vllm_ascend,"__version__","?"))'

sec "relevant env"
for v in ASCEND_RT_VISIBLE_DEVICES VLLM_USE_V2_MODEL_RUNNER VLLM_ENGINE_READY_TIMEOUT_S \
         TASK_QUEUE_ENABLE PYTORCH_NPU_ALLOC_CONF LD_PRELOAD VLLM_ASCEND_ENABLE_FLASHCOMM1; do
  say "    $v=${!v:-<unset>}"
done

sec "jemalloc"
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
  say "    present (the official recipe LD_PRELOADs it)"
else
  say "    absent  (the recipe's LD_PRELOAD branch will be skipped)"
fi

if [ "$FAILED" -gt 0 ]; then VERDICT=FAIL; CODE=1
elif [ "$UNKNOWN" -gt 0 ]; then VERDICT=INCONCLUSIVE; CODE=2
else VERDICT=PASS; CODE=0; fi

printf '%s\n' "$VERDICT"
echo "# 01_probe_env — environment facts"
cat "$OUT"
echo ""
echo "[written] $OUT"
exit "$CODE"
