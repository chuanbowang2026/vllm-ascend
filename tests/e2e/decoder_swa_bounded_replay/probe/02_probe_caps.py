#!/usr/bin/env python3
"""Stage-0 probe: hardware capabilities + config resolution on the real A3 box.

Needs: NPU visible, vllm_ascend importable. Does NOT need model weights.
Run inside the official container.

Verifies claims A1, A2, A7 from the deliverable README:
  A1  device type is A3 and DSV41_PACKED_CACHE is ABSENT (so the BF16 cache
      path is used, and KV is ~2885 B/token, not the paper's 890)
  A2  the DeepSeek-V4.1 model chain carries NO @support_torch_compile, so the
      whole `ascend_compilation_config` block in the official recipe is inert
  A7  CPU binding can actually apply (container cpuset >= 5 CPUs per device)

Usage:
  python3 -I 02_probe_caps.py
"""

from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from verdict import Report  # noqa: E402

# Capabilities we expect A3 to HAVE / LACK (vllm_ascend/device/hardware_profile.py)
EXPECT_PRESENT = [
    "NPUGRAPH_EX", "CANN_MEGAMOE", "MC2_FULLMESH_V2_COMM", "GMM_DEQUANT_SITU_QUANT",
    "RMS_NORM_CAST", "MOE_DISPATCH_EXTRA_ARGS", "IRQ_CPU_RESERVATION",
]
EXPECT_ABSENT = [
    "DSV41_PACKED_CACHE",            # A5 only -> A3 runs the generic BF16 layout
    "DSV4_COMPRESSED_CACHE",
    "CHUNKED_PREFILL_PHASE_SPLIT",   # A5 only
    "FP8_ATTENTION",                 # A5 only
    "UNRESTRICTED_MLAPO",            # A5 only
    "ENGRAM_MXFP8",                  # A5 only
]
MIN_CPUS_PER_NPU = 5  # vllm_ascend/cpu_binding.py:21


def parse_cpuset(text: str) -> int:
    n = 0
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n


def main() -> int:
    rep = Report("02_probe_caps", "A3 hardware capabilities, compile decorator, cpu binding")

    # ---- import -----------------------------------------------------------
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except Exception as exc:
        rep.check("import_torch_npu", None, detail=f"{type(exc).__name__}: {exc}")
        return rep.emit()

    try:
        from vllm_ascend.device.device_config import get_ascend_device_type
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            get_current_hardware_profile,
        )
    except Exception as exc:
        rep.check("import_vllm_ascend", None, detail=f"{type(exc).__name__}: {exc}")
        return rep.emit()

    # ---- A1: device type + capabilities -----------------------------------
    try:
        soc = torch_npu.npu.get_soc_version()
        rep.fact("runtime_soc_version", soc)
    except Exception as exc:
        rep.fact("runtime_soc_version", f"<unavailable: {exc}>")

    dev = get_ascend_device_type()
    rep.fact("device_type", str(dev))
    rep.check("A1.device_is_A3", "A3" in str(dev), "A3", str(dev), critical=True)

    profile = get_current_hardware_profile()
    present, absent, unknown = [], [], []
    for name in EXPECT_PRESENT + EXPECT_ABSENT:
        cap = getattr(HardwareCapability, name, None)
        if cap is None:
            unknown.append(name)
            continue
        (present if profile.supports(cap) else absent).append(name)
    if unknown:
        rep.note(f"capability names not found in this build (API drift?): {unknown}")

    for name in EXPECT_PRESENT:
        if name in unknown:
            rep.check(f"cap.{name}", None, "present", "<name not in enum>")
        else:
            rep.check(f"cap.{name}", name in present, "present", "present" if name in present else "ABSENT")
    for name in EXPECT_ABSENT:
        if name in unknown:
            rep.check(f"cap.{name}", None, "absent", "<name not in enum>")
        else:
            crit = name == "DSV41_PACKED_CACHE"
            rep.check(f"cap.{name}", name in absent, "absent",
                      "absent" if name in absent else "PRESENT", critical=crit,
                      detail=("if PRESENT, A3 can use the packed FP4/MXFP8 cache and the whole "
                              "KV memory account in docs/01 §4 must be redone")
                      if crit else "")

    rep.fact("moe_comm_policy", str(getattr(profile, "moe_comm_policy", "?")))
    rep.fact("cpu_binding_mode", str(getattr(profile, "cpu_binding_mode", "?")))
    rep.fact("device_addressing_mode", str(getattr(profile, "device_addressing_mode", "?")))

    # full capability dump, for the record
    try:
        all_caps = sorted(c.name for c in HardwareCapability if profile.supports(c))
        rep.fact("capabilities_present_count", len(all_caps))
        rep.fact("capabilities_present", all_caps)
    except Exception:
        pass

    # ---- A2: is @support_torch_compile on the V4.1 chain? ------------------
    decorated, undecorated, failed = [], [], []
    for mod_name in (
        "vllm_ascend.models.deepseek_v41.model",
        "vllm_ascend.models.deepseek_v41.vl_model",
        "vllm_ascend.models.deepseek_v41.dspark",
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
            src = inspect.getsource(mod)
        except Exception as exc:
            failed.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue
        hits = re.findall(r"^\s*@support_torch_compile.*$", src, re.M)
        (decorated if hits else undecorated).append(f"{mod_name} ({len(hits)} hits)")
    if failed:
        rep.note("modules that could not be read: " + "; ".join(failed))

    target_chain_clean = not any("deepseek_v41.model" in d for d in decorated) and not any(
        "deepseek_v41.vl_model" in d for d in decorated
    )
    rep.check(
        "A2.no_support_torch_compile_on_v41_target",
        target_chain_clean if (decorated or undecorated) else None,
        "model.py and vl_model.py undecorated",
        f"decorated={decorated or 'none'}",
        detail=("If undecorated, torch.compile never runs for the V4.1 target model, so "
                "enable_npugraph_ex / enable_static_kernel / the three fusion flags in the "
                "official recipe do nothing. Confirm at runtime by grepping the worker log for "
                "'enable_npugraph_ex is enabled'."),
    )
    rep.fact("support_torch_compile_decorated_modules", decorated or "none")

    # ---- A7: cpu binding feasibility --------------------------------------
    ncpu_cpuset = None
    for path in ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset/cpuset.cpus"):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    ncpu_cpuset = parse_cpuset(fh.read())
                rep.fact("cpuset_source", path)
                break
            except OSError:
                pass
    if ncpu_cpuset is None:
        try:
            ncpu_cpuset = len(os.sched_getaffinity(0))
            rep.fact("cpuset_source", "sched_getaffinity")
        except Exception:
            pass

    ndev = int(os.environ.get("PROBE_NUM_DEVICES", "16"))
    rep.fact("assumed_logical_devices", ndev)
    rep.fact("cpuset_cpu_count", ncpu_cpuset)
    if ncpu_cpuset is None:
        rep.check("A7.cpu_binding_feasible", None, detail="could not determine cpuset size")
    else:
        need = MIN_CPUS_PER_NPU * ndev
        rep.check("A7.cpu_binding_feasible", ncpu_cpuset >= need, f">= {need}", ncpu_cpuset,
                  detail=("vllm_ascend/cpu_binding.py MIN_CPUS_PER_NPU=5; below this the binding "
                          "raises RuntimeError which worker.py:952-955 swallows as a warning, so "
                          "it silently never applies. Grep the worker log for '[cpu_bind_mode]'."))

    rep.check("A7.proc_irq_writable", os.access("/proc/irq", os.W_OK), "writable",
              os.access("/proc/irq", os.W_OK),
              detail="unprivileged containers skip IRQ affinity (cpu_binding.py:697-701)")
    rep.check("A7.is_aarch64", os.uname().machine == "aarch64", "aarch64", os.uname().machine,
              detail="cpu_binding.py:798-800 returns early on non-ARM")

    # ---- environment, for the record --------------------------------------
    for var in ("ASCEND_RT_VISIBLE_DEVICES", "VLLM_USE_V2_MODEL_RUNNER", "HCCL_BUFFSIZE",
                "HCCL_OP_EXPANSION_MODE", "TASK_QUEUE_ENABLE", "PYTORCH_NPU_ALLOC_CONF",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1", "LD_PRELOAD"):
        rep.fact(f"env.{var}", os.environ.get(var, "<unset>"))

    try:
        import vllm
        rep.fact("vllm_version", getattr(vllm, "__version__", "?"))
    except Exception:
        rep.fact("vllm_version", "<not importable>")
    try:
        import vllm_ascend
        rep.fact("vllm_ascend_version", getattr(vllm_ascend, "__version__", "?"))
    except Exception:
        pass

    return rep.emit()


if __name__ == "__main__":
    raise SystemExit(main())
