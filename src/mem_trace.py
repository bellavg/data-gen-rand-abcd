"""mem_trace.py - Linux process-memory diagnostics for long training runs.

Primary capabilities:
  - Snapshot /proc/self/smaps and diff region RSS over time
  - Group memory by category (heap/anon/cuda/file/stack)
  - Report cgroup memory.current/memory.max and OOM counters
  - Report smaps_rollup metrics (Anonymous, Private_*, Shared_*, Swap)
  - Report process fault counters (minor/major page faults) and FD/thread counts
  - Report Python object census (torch tensors, numpy arrays, PyG Data/Batch)
  - Optional tracemalloc stack-based allocation summary

All methods are best-effort and silently degrade when /proc or a specific file
is unavailable (for portability and low operational risk).
"""

from __future__ import annotations

import collections
import ctypes
import gc
import os
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Optional
import warnings

_SMAPS_PATH = "/proc/self/smaps"
_SMAPS_ROLLUP_PATH = "/proc/self/smaps_rollup"
_STATUS_PATH = "/proc/self/status"
_STAT_PATH = "/proc/self/stat"
_FD_PATH = "/proc/self/fd"
_CGROUP_PATH = "/proc/self/cgroup"
_CGROUP_V2_ROOT = "/sys/fs/cgroup"
_CGROUP_V1_MEM_ROOT = "/sys/fs/cgroup/memory"


@dataclass
class _Region:
    start: int
    end: int
    perms: str
    pathname: str  # "" = anonymous, "[heap]", "[stack]", or an fs path
    rss_kb: int
    private_dirty_kb: int
    private_clean_kb: int
    shared_dirty_kb: int
    size_kb: int  # virtual size - not necessarily resident


@dataclass
class _ProcStat:
    minflt: int
    majflt: int
    num_threads: int
    vsize_bytes: int
    rss_pages: int


@dataclass
class Snapshot:
    label: str
    regions: list[_Region] = field(default_factory=list)

    @staticmethod
    def _category(r: _Region) -> str:
        p = r.pathname
        if p == "[heap]":
            return "heap"
        if p == "[stack]" or p.startswith("[stack:"):
            return "stack"
        if p in ("[vvar]", "[vdso]", "[vsyscall]"):
            return "kernel_virt"
        if p and not p.startswith("["):
            lp = p.lower()
            if "cuda" in lp or "nvidia" in lp or "/dev/nvidia" in lp:
                return "cuda_driver"
            if "tcmalloc" in lp or "gperftools" in lp:
                return "tcmalloc_lib"
            return "file_mapped"
        return "anon"

    def summary(self) -> dict[str, int]:
        cats: dict[str, int] = {
            "heap": 0,
            "stack": 0,
            "file_mapped": 0,
            "cuda_driver": 0,
            "tcmalloc_lib": 0,
            "kernel_virt": 0,
            "anon": 0,
        }
        for r in self.regions:
            cats[self._category(r)] += r.rss_kb
        cats["total"] = sum(v for k, v in cats.items() if k != "total")
        return cats

    def private_dirty_summary(self) -> dict[str, int]:
        cats: dict[str, int] = {
            "heap": 0,
            "stack": 0,
            "file_mapped": 0,
            "cuda_driver": 0,
            "tcmalloc_lib": 0,
            "anon": 0,
        }
        for r in self.regions:
            cat = self._category(r)
            if cat in cats:
                cats[cat] += r.private_dirty_kb
        cats["total"] = sum(cats.values())
        return cats

    def category_counts(self) -> dict[str, int]:
        counts: collections.Counter[str] = collections.Counter()
        for r in self.regions:
            counts[self._category(r)] += 1
        return dict(counts)

    def region_map(self) -> dict[int, _Region]:
        return {r.start: r for r in self.regions}

    def top_regions(self, n: int = 20) -> list[_Region]:
        return sorted(self.regions, key=lambda r: r.rss_kb, reverse=True)[:n]


def _parse_smaps() -> list[_Region]:
    regions: list[_Region] = []
    try:
        with open(_SMAPS_PATH) as fh:
            current: Optional[dict[str, int | str]] = None
            for raw in fh:
                line = raw.rstrip()
                if not line:
                    continue
                if line[0] in "0123456789abcdef" and "-" in line[:20]:
                    if current is not None:
                        regions.append(_Region(**current))
                    parts = line.split(None, 5)
                    addr_range = parts[0]
                    perms = parts[1] if len(parts) > 1 else ""
                    pathname = parts[5].strip() if len(parts) > 5 else ""
                    start_s, end_s = addr_range.split("-", 1)
                    current = {
                        "start": int(start_s, 16),
                        "end": int(end_s, 16),
                        "perms": perms,
                        "pathname": pathname,
                        "rss_kb": 0,
                        "private_dirty_kb": 0,
                        "private_clean_kb": 0,
                        "shared_dirty_kb": 0,
                        "size_kb": 0,
                    }
                elif current is not None:
                    if line.startswith("Rss:"):
                        current["rss_kb"] = int(line.split()[1])
                    elif line.startswith("Private_Dirty:"):
                        current["private_dirty_kb"] = int(line.split()[1])
                    elif line.startswith("Private_Clean:"):
                        current["private_clean_kb"] = int(line.split()[1])
                    elif line.startswith("Shared_Dirty:"):
                        current["shared_dirty_kb"] = int(line.split()[1])
                    elif line.startswith("Size:"):
                        current["size_kb"] = int(line.split()[1])
            if current is not None:
                regions.append(_Region(**current))
    except (OSError, ValueError):
        pass
    return regions


def _read_proc_status() -> dict[str, int]:
    fields: dict[str, int] = {}
    keys = {
        "VmRSS",
        "VmHWM",
        "VmPeak",
        "VmData",
        "VmAnon",
        "VmPTE",
        "VmSwap",
        "RssAnon",
        "RssFile",
        "RssShmem",
        "Threads",
        "FDSize",
    }
    try:
        with open(_STATUS_PATH) as fh:
            for line in fh:
                key = line.split(":")[0]
                if key in keys:
                    try:
                        fields[key] = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    return fields


def _read_smaps_rollup() -> dict[str, int]:
    fields: dict[str, int] = {}
    keys = {
        "Rss",
        "Pss",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
        "Referenced",
        "Anonymous",
        "AnonHugePages",
        "Swap",
        "FilePmdMapped",
        "ShmemPmdMapped",
        "Locked",
    }
    try:
        with open(_SMAPS_ROLLUP_PATH) as fh:
            for raw in fh:
                if ":" not in raw:
                    continue
                key = raw.split(":", 1)[0]
                if key not in keys:
                    continue
                try:
                    fields[key] = int(raw.split()[1])
                except (IndexError, ValueError):
                    pass
    except OSError:
        pass
    return fields


def _read_proc_stat() -> Optional[_ProcStat]:
    try:
        with open(_STAT_PATH) as fh:
            raw = fh.read().strip()
        right_paren = raw.rfind(")")
        if right_paren < 0:
            return None
        rest = raw[right_paren + 2 :].split()
        if len(rest) < 22:
            return None
        return _ProcStat(
            minflt=int(rest[7]),
            majflt=int(rest[9]),
            num_threads=int(rest[17]),
            vsize_bytes=int(rest[20]),
            rss_pages=int(rest[21]),
        )
    except (OSError, ValueError, IndexError):
        return None


def _count_open_fds() -> Optional[int]:
    try:
        return len(os.listdir(_FD_PATH))
    except OSError:
        return None


def _read_numeric_file(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_kv_numeric_file(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open(path) as fh:
            for raw in fh:
                parts = raw.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    out[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _read_cgroup_paths() -> tuple[Optional[str], Optional[str]]:
    rel_v2: Optional[str] = None
    rel_v1_mem: Optional[str] = None
    try:
        with open(_CGROUP_PATH) as fh:
            for raw in fh:
                parts = raw.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _, controllers, rel = parts
                if controllers == "":
                    rel_v2 = rel
                else:
                    ctrls = controllers.split(",")
                    if "memory" in ctrls:
                        rel_v1_mem = rel
    except OSError:
        pass
    return rel_v2, rel_v1_mem


def _read_cgroup_memory() -> dict[str, int | str | None]:
    rel_v2, rel_v1_mem = _read_cgroup_paths()
    candidates: list[tuple[str, str]] = []

    if rel_v2 is not None:
        candidates.append(("v2", os.path.join(_CGROUP_V2_ROOT, rel_v2.lstrip("/"))))
        candidates.append(("v2", _CGROUP_V2_ROOT))
    if rel_v1_mem is not None:
        candidates.append(
            ("v1", os.path.join(_CGROUP_V1_MEM_ROOT, rel_v1_mem.lstrip("/")))
        )
        candidates.append(("v1", _CGROUP_V1_MEM_ROOT))

    if not candidates:
        candidates = [("v2", _CGROUP_V2_ROOT), ("v1", _CGROUP_V1_MEM_ROOT)]

    seen: set[str] = set()
    for controller, base in candidates:
        if not base or base in seen:
            continue
        seen.add(base)

        if controller == "v2":
            current = _read_numeric_file(os.path.join(base, "memory.current"))
            limit = _read_numeric_file(os.path.join(base, "memory.max"))
            high = _read_numeric_file(os.path.join(base, "memory.high"))
            events = _read_kv_numeric_file(os.path.join(base, "memory.events"))
        else:
            current = _read_numeric_file(os.path.join(base, "memory.usage_in_bytes"))
            limit = _read_numeric_file(os.path.join(base, "memory.limit_in_bytes"))
            high = None
            events = _read_kv_numeric_file(os.path.join(base, "memory.oom_control"))
            failcnt = _read_numeric_file(os.path.join(base, "memory.failcnt"))
            if failcnt is not None:
                events.setdefault("failcnt", failcnt)

        if current is None and limit is None and not events:
            continue

        if limit is not None and limit >= (1 << 60):
            limit = None
        if high is not None and high >= (1 << 60):
            high = None

        return {
            "controller": controller,
            "path": base,
            "current_bytes": current,
            "limit_bytes": limit,
            "high_bytes": high,
            "oom": events.get("oom"),
            "oom_kill": events.get("oom_kill"),
            "failcnt": events.get("failcnt"),
        }

    return {}


def _mib_from_kb(kb: Optional[int]) -> str:
    if kb is None:
        return "n/a"
    return f"{kb // 1024}MiB"


def _mib_from_bytes(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value // (1024**2)}MiB"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _rss_gib() -> float:
    status = _read_proc_status()
    vmrss = status.get("VmRSS")
    if vmrss is None:
        return float("nan")
    return vmrss / (1024**2)


_release_fn_checked: bool = False

# Suppress specific PyTorch and Lightning deprecation warnings that clutter logs
warnings.filterwarnings("ignore", category=FutureWarning, message=".*reduce_op is deprecated.*")
warnings.filterwarnings(r"ignore", message=r".*isinstance\(treespec, LeafSpec\) is deprecated.*")
warnings.filterwarnings(r"ignore", message=r".*'torch_geometric.contrib' contains experimental code.*")


def release_reclaimable_memory(label: str | None = None) -> None:
    """Best-effort host/GPU cache release using the active allocator hooks.

    Uses CDLL(None) so LD_PRELOAD'ed allocators (e.g. TCMalloc) are respected.
    """

    global _release_fn_checked

    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        release_fn = getattr(libc, "MallocExtension_ReleaseFreeMemory", None)
        if callable(release_fn):
            if not _release_fn_checked:
                print(
                    "[release] MallocExtension_ReleaseFreeMemory resolved - TCMalloc active",
                    flush=True,
                )
                _release_fn_checked = True
            release_fn()
        else:
            trim = getattr(libc, "malloc_trim", None)
            if not _release_fn_checked:
                print(
                    "[release] MallocExtension_ReleaseFreeMemory NOT found; "
                    f"malloc_trim callable={callable(trim)}",
                    flush=True,
                )
                _release_fn_checked = True
            if callable(trim):
                trim(0)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
    except Exception:
        pass

    if label is not None:
        print(f"[mem] {label} host_rss={_rss_gib():.2f} GiB", flush=True)


def drop_trainer_batch_refs(trainer: Any) -> None:
    """Defensively clear trainer-side batch references after a step."""

    try:
        results = getattr(trainer, "_results", None)
        if results is not None:
            results.batch = None
            results.batch_size = None
    except Exception:
        pass


@dataclass
class MemoryTraceConfig:
    step_interval: int = 1000
    max_step: int = 3000
    top_types: int = 25
    top_objects: int = 10
    top_regions: int = 5
    enable_tracemalloc: bool = False
    tracemalloc_frames: int = 25
    verbose: bool = False

    @staticmethod
    def _env_int(name: str, default: int, minimum: int = 0) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return max(minimum, default)
        try:
            return max(minimum, int(raw))
        except ValueError:
            return max(minimum, default)

    @classmethod
    def from_env(cls) -> "MemoryTraceConfig":
        step_interval = cls._env_int("MEM_TRACE_STEP_INTERVAL", 1000, minimum=1)
        max_step = cls._env_int("MEM_TRACE_MAX_STEP", 3000, minimum=step_interval)
        return cls(
            step_interval=step_interval,
            max_step=max_step,
            top_types=cls._env_int("MEM_TRACE_TOP_TYPES", 25, minimum=1),
            top_objects=cls._env_int("MEM_TRACE_TOP_OBJECTS", 10, minimum=1),
            top_regions=cls._env_int("MEM_TRACE_TOP_REGIONS", 5, minimum=1),
            enable_tracemalloc=_env_flag("MEM_TRACE_ENABLE_TRACEMALLOC", False),
            tracemalloc_frames=cls._env_int(
                "MEM_TRACE_TRACEMALLOC_FRAMES", 25, minimum=1
            ),
            verbose=_env_flag("MEM_TRACE_VERBOSE", False),
        )


class MemoryTracer:
    """Captures /proc snapshots and prints diagnostics between checkpoints.

    Intended for single-process, main-thread use.
    """

    def __init__(
        self,
        *,
        enable_tracemalloc: Optional[bool] = None,
        tracemalloc_frames: int = 25,
        top_types: int = 20,
        verbose: bool = False,
    ) -> None:
        self._snapshots: dict[str, Snapshot] = {}
        self._last_proc_stat: Optional[_ProcStat] = None
        self._top_types = max(1, int(top_types))
        self.verbose = bool(verbose)
        self._last_tensor_stats: dict[str, float] = {}

        if enable_tracemalloc is None:
            enable_tracemalloc = (
                os.environ.get("MEM_TRACE_ENABLE_TRACEMALLOC", "0").strip().lower()
                in {"1", "true", "yes"}
            )

        self._tracemalloc_enabled = False
        self._owns_tracemalloc = False
        if enable_tracemalloc:
            try:
                if not tracemalloc.is_tracing():
                    tracemalloc.start(max(1, int(tracemalloc_frames)))
                    self._owns_tracemalloc = True
                self._tracemalloc_enabled = True
                self._print(
                    f"[mem_diag] tracemalloc=on nframe={max(1, int(tracemalloc_frames))}"
                )
            except Exception:
                self._tracemalloc_enabled = False
                self._owns_tracemalloc = False

    def _print(self, msg: str) -> None:
        """Helper to hide spam unless MEM_TRACE_VERBOSE=1"""
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _region_archetype(r: _Region) -> tuple[str, str, str]:
        cat = Snapshot._category(r)
        if cat == "anon":
            name = "<anon>"
        elif r.pathname:
            name = os.path.basename(r.pathname) if r.pathname.startswith("/") else r.pathname
        else:
            name = "<none>"
        return (cat, r.perms, name)

    @staticmethod
    def _push_top(
        bucket: list[tuple[int, str]],
        size_bytes: int,
        desc: str,
        keep: int,
    ) -> None:
        if keep <= 0 or size_bytes <= 0:
            return
        bucket.append((size_bytes, desc))
        if len(bucket) > keep * 3:
            bucket.sort(key=lambda x: x[0], reverse=True)
            del bucket[keep:]

    @staticmethod
    def _format_top_bytes(entries: list[tuple[int, str]], n: int = 8) -> str:
        if not entries:
            return ""
        entries = sorted(entries, key=lambda x: x[0], reverse=True)[:n]
        return " | ".join(
            f"{size / (1024**2):.1f}MiB:{desc}" for size, desc in entries
        )

    def snapshot(self, label: str) -> Snapshot:
        snap = Snapshot(label=label, regions=_parse_smaps())
        self._snapshots[label] = snap

        status = _read_proc_status()
        summ = snap.summary()
        pd = snap.private_dirty_summary()
        counts = snap.category_counts()
        self._print(
            f"[mem_trace] snap={label!r}"
            f"  heap={summ['heap'] // 1024}MiB"
            f"  anon={summ['anon'] // 1024}MiB"
            f"  cuda_driver={summ['cuda_driver'] // 1024}MiB"
            f"  file_mapped={summ['file_mapped'] // 1024}MiB"
            f"  total_rss={summ['total'] // 1024}MiB"
            f"  VmRSS={status.get('VmRSS', 0) // 1024}MiB"
            f"  VmHWM={status.get('VmHWM', 0) // 1024}MiB"
            f"  private_dirty={pd['total'] // 1024}MiB"
            f"  regions={len(snap.regions)}"
            f"  anon_regions={counts.get('anon', 0)}"
        )
        return snap

    def report_process_snapshot(self, label: str, *, step: Optional[int] = None) -> None:
        step_tag = f" step={step}" if step is not None else ""

        status = _read_proc_status()
        rollup = _read_smaps_rollup()
        cgroup = _read_cgroup_memory()
        proc_stat = _read_proc_stat()
        fd_count = _count_open_fds()

        self._print(
            f"[mem_diag]{step_tag} label={label!r}"
            f" VmRSS={_mib_from_kb(status.get('VmRSS'))}"
            f" VmHWM={_mib_from_kb(status.get('VmHWM'))}"
            f" VmAnon={_mib_from_kb(status.get('VmAnon'))}"
            f" VmData={_mib_from_kb(status.get('VmData'))}"
            f" VmSwap={_mib_from_kb(status.get('VmSwap'))}"
            f" RssAnon={_mib_from_kb(status.get('RssAnon'))}"
            f" RssFile={_mib_from_kb(status.get('RssFile'))}"
            f" threads={status.get('Threads', -1)}"
            f" fd_size={status.get('FDSize', -1)}"
            f" open_fds={fd_count if fd_count is not None else 'n/a'}"
        )

        if rollup:
            private_kb = rollup.get("Private_Clean", 0) + rollup.get("Private_Dirty", 0)
            shared_kb = rollup.get("Shared_Clean", 0) + rollup.get("Shared_Dirty", 0)
            self._print(
                f"[mem_rollup]{step_tag} label={label!r}"
                f" Rss={_mib_from_kb(rollup.get('Rss'))}"
                f" Pss={_mib_from_kb(rollup.get('Pss'))}"
                f" Anonymous={_mib_from_kb(rollup.get('Anonymous'))}"
                f" Private={_mib_from_kb(private_kb)}"
                f" Shared={_mib_from_kb(shared_kb)}"
                f" Swap={_mib_from_kb(rollup.get('Swap'))}"
                f" AnonHugePages={_mib_from_kb(rollup.get('AnonHugePages'))}"
                f" Referenced={_mib_from_kb(rollup.get('Referenced'))}"
            )

        if cgroup:
            current = cgroup.get("current_bytes")
            limit = cgroup.get("limit_bytes")
            high = cgroup.get("high_bytes")

            usage_pct = "n/a"
            if isinstance(current, int) and isinstance(limit, int) and limit > 0:
                usage_pct = f"{(100.0 * current / limit):.1f}%"

            self._print(
                f"[mem_cgroup]{step_tag} label={label!r}"
                f" controller={cgroup.get('controller', 'n/a')}"
                f" current={_mib_from_bytes(current if isinstance(current, int) else None)}"
                f" limit={_mib_from_bytes(limit if isinstance(limit, int) else None)}"
                f" high={_mib_from_bytes(high if isinstance(high, int) else None)}"
                f" usage={usage_pct}"
                f" oom={cgroup.get('oom', 'n/a')}"
                f" oom_kill={cgroup.get('oom_kill', 'n/a')}"
                f" failcnt={cgroup.get('failcnt', 'n/a')}"
            )

        if proc_stat is not None:
            page_size = 4096
            try:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
            except (AttributeError, ValueError, OSError):
                pass

            delta_minflt = 0
            delta_majflt = 0
            if self._last_proc_stat is not None:
                delta_minflt = proc_stat.minflt - self._last_proc_stat.minflt
                delta_majflt = proc_stat.majflt - self._last_proc_stat.majflt
            self._last_proc_stat = proc_stat

            rss_pages_mib = (proc_stat.rss_pages * page_size) // (1024**2)
            vsize_mib = proc_stat.vsize_bytes // (1024**2)
            self._print(
                f"[mem_faults]{step_tag} label={label!r}"
                f" minflt={proc_stat.minflt} (d{delta_minflt:+d})"
                f" majflt={proc_stat.majflt} (d{delta_majflt:+d})"
                f" rss_pages={proc_stat.rss_pages} (~{rss_pages_mib}MiB)"
                f" vsize={vsize_mib}MiB"
                f" threads={proc_stat.num_threads}"
            )

        try:
            gc_counts = gc.get_count()
            gc_stats = gc.get_stats()
            gen2 = gc_stats[2] if len(gc_stats) > 2 else {}
            self._print(
                f"[gc_diag]{step_tag} label={label!r}"
                f" counts={gc_counts}"
                f" gen2_collections={gen2.get('collections', 0)}"
                f" gen2_collected={gen2.get('collected', 0)}"
                f" gen2_uncollectable={gen2.get('uncollectable', 0)}"
            )
        except Exception:
            pass

    def report_python_object_snapshot(
        self,
        *,
        step: int,
        top_types: Optional[int] = None,
        top_objects: int = 8,
    ) -> None:
        top_types = self._top_types if top_types is None else max(1, int(top_types))
        top_objects = max(0, int(top_objects))

        torch = None
        np = None
        try:
            import torch as _torch

            torch = _torch
        except Exception:
            pass
        try:
            import numpy as _np

            np = _np
        except Exception:
            pass

        gc.collect()

        n_tensor_cpu, tensor_cpu_bytes = 0, 0
        n_tensor_cuda, tensor_cuda_bytes = 0, 0
        n_np_root, np_root_bytes = 0, 0
        n_pyg, pyg_attr_bytes = 0, 0

        type_counts: collections.Counter[str] = collections.Counter()
        module_counts: collections.Counter[str] = collections.Counter()

        largest_tensors: list[tuple[int, str]] = []
        largest_arrays: list[tuple[int, str]] = []
        largest_pyg_attrs: list[tuple[int, str]] = []

        seen_tensor_ids: set[int] = set()
        seen_array_ids: set[int] = set()

        for obj in gc.get_objects():
            try:
                t = type(obj)
                type_counts[t.__qualname__] += 1
                mod0 = (t.__module__ or "").split(".", 1)[0] or "<none>"
                module_counts[mod0] += 1

                tensor_obj = None
                if torch is not None:
                    try:
                        if torch.is_tensor(obj):
                            tensor_obj = obj
                        elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                            tensor_obj = obj.data
                    except Exception:
                        tensor_obj = None

                if tensor_obj is not None:
                    tid = id(tensor_obj)
                    if tid not in seen_tensor_ids:
                        seen_tensor_ids.add(tid)
                        try:
                            nbytes = tensor_obj.element_size() * tensor_obj.nelement()
                            dev = getattr(tensor_obj, "device", None)
                            dev_type = getattr(dev, "type", "unknown")
                            shape = tuple(tensor_obj.shape)
                            dtype = getattr(tensor_obj, "dtype", "?")
                            if dev_type == "cuda":
                                n_tensor_cuda += 1
                                tensor_cuda_bytes += nbytes
                            else:
                                n_tensor_cpu += 1
                                tensor_cpu_bytes += nbytes
                            self._push_top(
                                largest_tensors,
                                nbytes,
                                f"{shape} {dtype} {dev_type}",
                                keep=top_objects,
                            )
                        except Exception:
                            pass

                if np is not None and isinstance(obj, np.ndarray) and obj.base is None:
                    oid = id(obj)
                    if oid not in seen_array_ids:
                        seen_array_ids.add(oid)
                        try:
                            nbytes = int(obj.nbytes)
                            n_np_root += 1
                            np_root_bytes += nbytes
                            self._push_top(
                                largest_arrays,
                                nbytes,
                                f"shape={tuple(obj.shape)} dtype={obj.dtype}",
                                keep=top_objects,
                            )
                        except Exception:
                            pass

                mod = t.__module__ or ""
                if "torch_geometric" in mod and t.__name__ in ("Data", "HeteroData", "Batch"):
                    n_pyg += 1
                    store = getattr(obj, "_store", None)
                    if store is not None:
                        try:
                            for key, val in store.items():
                                if torch is not None and torch.is_tensor(val):
                                    nbytes = val.element_size() * val.nelement()
                                    pyg_attr_bytes += nbytes
                                    self._push_top(
                                        largest_pyg_attrs,
                                        nbytes,
                                        f"{t.__name__}.{key} tensor {tuple(val.shape)} {val.dtype}",
                                        keep=top_objects,
                                    )
                                elif np is not None and isinstance(val, np.ndarray):
                                    nbytes = int(val.nbytes)
                                    pyg_attr_bytes += nbytes
                                    self._push_top(
                                        largest_pyg_attrs,
                                        nbytes,
                                        f"{t.__name__}.{key} ndarray {tuple(val.shape)} {val.dtype}",
                                        keep=top_objects,
                                    )
                        except Exception:
                            pass
            except Exception:
                pass

        rss_gib = float("nan")
        try:
            status = _read_proc_status()
            vmrss = status.get("VmRSS")
            if vmrss is not None:
                rss_gib = vmrss / (1024**2)
        except Exception:
            pass

        pinned_mib = float("nan")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    stats = torch.cuda.memory_stats()
                    pinned_mib = stats.get("pinned_mem_allocated_bytes.current", 0) / (
                        1024**2
                    )
            except Exception:
                pass

        # Cache last observed tensor stats (MiB) for compact reporting
        try:
            self._last_tensor_stats = {
                "cuda_mib": float(tensor_cuda_bytes) / (1024**2),
                "cpu_mib": float(tensor_cpu_bytes) / (1024**2),
            }
        except Exception:
            self._last_tensor_stats = {"cuda_mib": 0.0, "cpu_mib": 0.0}

        self._print(
            f"[tensor_snapshot] step={step}"
            f" cpu_tensors={n_tensor_cpu} tensor_mib={tensor_cpu_bytes / (1024**2):.2f}"
            f" cuda_tensors={n_tensor_cuda} cuda_tensor_mib={tensor_cuda_bytes / (1024**2):.2f}"
            f" pinned_mib={pinned_mib:.2f}"
            f" numpy_arrays={n_np_root} numpy_mib={np_root_bytes / (1024**2):.2f}"
            f" pyg_data={n_pyg} pyg_attr_mib={pyg_attr_bytes / (1024**2):.2f}"
            f" host_rss={rss_gib:.2f} GiB"
        )

        type_parts = [f"{name}={cnt}" for name, cnt in type_counts.most_common(top_types)]
        self._print(f"[obj_types] step={step} " + " | ".join(type_parts))

        mod_parts = [f"{name}={cnt}" for name, cnt in module_counts.most_common(12)]
        self._print(f"[obj_modules] step={step} " + " | ".join(mod_parts))

        tensor_top = self._format_top_bytes(largest_tensors, n=top_objects)
        if tensor_top:
            self._print(f"[tensor_top] step={step} {tensor_top}")

        numpy_top = self._format_top_bytes(largest_arrays, n=top_objects)
        if numpy_top:
            self._print(f"[numpy_top] step={step} {numpy_top}")

        pyg_top = self._format_top_bytes(largest_pyg_attrs, n=top_objects)
        if pyg_top:
            self._print(f"[pyg_attr_top] step={step} {pyg_top}")

        if self._tracemalloc_enabled:
            self._report_tracemalloc(step=step)

    def _report_tracemalloc(self, *, step: Optional[int] = None, top_n: int = 8) -> None:
        if not self._tracemalloc_enabled or not tracemalloc.is_tracing():
            return
        try:
            snapshot = tracemalloc.take_snapshot()
            top = snapshot.statistics("lineno")[: max(1, int(top_n))]
        except Exception:
            return
        if not top:
            return

        step_tag = f" step={step}" if step is not None else ""
        self._print(f"[tracemalloc]{step_tag} top={len(top)}")
        for stat in top:
            frame = stat.traceback[0]
            self._print(
                f"  {stat.size / (1024**2):7.2f}MiB blocks={stat.count:7d}"
                f"  {frame.filename}:{frame.lineno}"
            )

    def report_diff(self, label_a: str, label_b: str, top_n: int = 20) -> None:
        if label_a not in self._snapshots or label_b not in self._snapshots:
            self._print(
                f"[mem_trace] diff: missing snapshot(s) - have {list(self._snapshots.keys())}"
            )
            return

        a = self._snapshots[label_a]
        b = self._snapshots[label_b]
        map_a = a.region_map()
        map_b = b.region_map()

        summ_a = a.summary()
        summ_b = b.summary()
        pd_a = a.private_dirty_summary()
        pd_b = b.private_dirty_summary()

        delta_total = summ_b["total"] - summ_a["total"]
        delta_pd = pd_b["total"] - pd_a["total"]

        new_regions = sum(1 for start in map_b if start not in map_a)
        removed_regions = sum(1 for start in map_a if start not in map_b)

        self._print(
            f"\n[mem_trace] diff {label_a!r} -> {label_b!r}"
            f"  dtotal={delta_total // 1024:+d}MiB"
            f"  dprivate_dirty={delta_pd // 1024:+d}MiB"
            f"  region_churn new={new_regions} removed={removed_regions}"
        )

        for cat in (
            "heap",
            "anon",
            "cuda_driver",
            "file_mapped",
            "tcmalloc_lib",
            "stack",
        ):
            delta_cat = summ_b.get(cat, 0) - summ_a.get(cat, 0)
            if abs(delta_cat) >= 1024:
                self._print(
                    f"  d{cat}={delta_cat // 1024:+d}MiB"
                    f"  ({summ_a.get(cat, 0) // 1024}->{summ_b.get(cat, 0) // 1024}MiB)"
                )

        agg_a: collections.Counter[tuple[str, str, str]] = collections.Counter()
        agg_b: collections.Counter[tuple[str, str, str]] = collections.Counter()
        for r in a.regions:
            agg_a[self._region_archetype(r)] += r.rss_kb
        for r in b.regions:
            agg_b[self._region_archetype(r)] += r.rss_kb

        agg_deltas: list[tuple[int, tuple[str, str, str]]] = []
        for key in set(agg_a) | set(agg_b):
            dkb = agg_b.get(key, 0) - agg_a.get(key, 0)
            if dkb != 0:
                agg_deltas.append((dkb, key))
        agg_deltas.sort(key=lambda x: x[0], reverse=True)

        if agg_deltas:
            self._print(f"[mem_trace] top {min(8, len(agg_deltas))} archetype deltas:")
            for dkb, (cat, perms, name) in agg_deltas[:8]:
                self._print(f"  {dkb // 1024:+6d}MiB  [{perms}] [{cat}]  {name}")

        deltas: list[tuple[int, _Region, int, int]] = []
        for start in set(map_a) | set(map_b):
            r_a = map_a.get(start)
            r_b = map_b.get(start)
            rss_a = r_a.rss_kb if r_a else 0
            rss_b = r_b.rss_kb if r_b else 0
            delta_kb = rss_b - rss_a
            if delta_kb == 0:
                continue
            region = r_b if r_b is not None else r_a
            assert region is not None
            deltas.append((delta_kb, region, rss_a, rss_b))

        deltas.sort(key=lambda x: x[0], reverse=True)
        growing = [d for d in deltas if d[0] > 0]
        shrinking = [d for d in deltas if d[0] < 0]

        if growing:
            self._print(f"[mem_trace] top {min(top_n, len(growing))} growing regions:")
            for delta_kb, region, rss_a, rss_b in growing[:top_n]:
                name = region.pathname or f"<anon @{region.start:#x}>"
                cat = Snapshot._category(region)
                self._print(
                    f"  {delta_kb // 1024:+6d}MiB  [{region.perms}] [{cat}]  {name}"
                    f"  ({rss_a // 1024}->{rss_b // 1024}MiB)"
                    f"  pdirty={region.private_dirty_kb // 1024}MiB"
                )

        if shrinking:
            n_show = min(5, len(shrinking))
            self._print(f"[mem_trace] top {n_show} shrinking regions:")
            for delta_kb, region, _, _ in shrinking[-n_show:]:
                name = region.pathname or f"<anon @{region.start:#x}>"
                self._print(f"  {delta_kb // 1024:+6d}MiB  [{region.perms}]  {name}")

        self._print("")

    def report_top(self, label: str, n: int = 20) -> None:
        if label not in self._snapshots:
            return
        snap = self._snapshots[label]
        self._print(f"[mem_trace] top {n} RSS regions at {label!r}:")
        for r in snap.top_regions(n=n):
            name = r.pathname or f"<anon @{r.start:#x}>"
            cat = Snapshot._category(r)
            self._print(
                f"  {r.rss_kb // 1024:6d}MiB  [{r.perms}] [{cat}]  {name}"
                f"  pdirty={r.private_dirty_kb // 1024}MiB"
            )
        self._print("")

    def report_compact_summary(self, label: str, prev_label: Optional[str] = None) -> None:
        """Prints a single, compact telemetry line for a snapshot label.

        This is intended to be user-facing and always emits a single-line summary
        regardless of the verbose setting.
        """
        snap = self._snapshots.get(label)
        if not snap:
            return

        status = _read_proc_status()
        summ = snap.summary()
        cgroup = _read_cgroup_memory()

        vm_rss = status.get("VmRSS", 0) // 1024
        vm_hwm = status.get("VmHWM", 0) // 1024
        heap = summ.get("heap", 0) // 1024
        anon = summ.get("anon", 0) // 1024
        gpu_mib = float(self._last_tensor_stats.get("cuda_mib", 0.0))

        cgroup_str = ""
        if cgroup:
            curr, limit = cgroup.get("current_bytes"), cgroup.get("limit_bytes")
            if isinstance(curr, int) and isinstance(limit, int) and limit > 0:
                cgroup_str = f" (CGroup: {(100.0 * curr / limit):.1f}%)"

        diff_str = ""
        if prev_label and prev_label in self._snapshots:
            prev_summ = self._snapshots[prev_label].summary()
            diff_rss = (summ.get("total", 0) - prev_summ.get("total", 0)) // 1024
            diff_heap = (summ.get("heap", 0) - prev_summ.get("heap", 0)) // 1024
            diff_str = f" | Δ RSS: {diff_rss:+d}MiB | Δ Heap: {diff_heap:+d}MiB"

        print(
            f"📊 [MemTrace] {label} | RSS: {vm_rss}MiB{cgroup_str} (Peak: {vm_hwm}MiB) | "
            f"GPU Tensors: {gpu_mib:.1f}MiB | Heap: {heap}MiB | Anon: {anon}MiB{diff_str}",
            flush=True,
        )

    def clear(self) -> None:
        self._snapshots.clear()
        self._last_proc_stat = None
        if self._owns_tracemalloc and tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except Exception:
                pass
            self._owns_tracemalloc = False
            self._tracemalloc_enabled = False


class MemoryTraceSession:
    """Small orchestration wrapper around MemoryTracer for train callbacks."""

    def __init__(self, tracer: MemoryTracer, config: MemoryTraceConfig) -> None:
        self.tracer = tracer
        self.config = config

    @classmethod
    def from_env(cls) -> "MemoryTraceSession":
        config = MemoryTraceConfig.from_env()
        tracer = MemoryTracer(
            enable_tracemalloc=config.enable_tracemalloc,
            tracemalloc_frames=config.tracemalloc_frames,
            top_types=config.top_types,
            verbose=config.verbose,
        )
        return cls(tracer=tracer, config=config)

    def maybe_capture_step(self, step: int) -> None:
        cfg = self.config
        if step <= 0:
            return
        if step % cfg.step_interval != 0:
            return
        if step > cfg.max_step:
            return

        snap_label = f"step_{step}"
        self.tracer.report_process_snapshot(snap_label, step=step)
        self.tracer.report_python_object_snapshot(
            step=step,
            top_types=cfg.top_types,
            top_objects=cfg.top_objects,
        )
        self.tracer.snapshot(snap_label)

        prev_label = (
            "baseline"
            if step == cfg.step_interval
            else f"step_{step - cfg.step_interval}"
        )
        self.tracer.report_diff(prev_label, snap_label)
        self.tracer.report_top(snap_label, n=cfg.top_regions)
        # Emit compact one-line telemetry for easy log scanning
        try:
            self.tracer.report_compact_summary(snap_label, prev_label)
        except Exception:
            pass

    def capture_baseline(self, *, step: int = 0) -> None:
        self.tracer.report_process_snapshot("baseline", step=step)
        self.tracer.report_python_object_snapshot(
            step=step,
            top_types=self.config.top_types,
            top_objects=self.config.top_objects,
        )
        self.tracer.snapshot("baseline")

    def clear(self) -> None:
        self.tracer.clear()
