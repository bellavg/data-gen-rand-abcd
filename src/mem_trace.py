"""mem_trace.py — /proc/self/smaps region snapshot and diff tool.

Captures named VM-region snapshots at step checkpoints and diffs them to
identify exactly which memory regions are responsible for the observed RSS
growth.  Completely self-contained; no dependency on HP tuning or model code.

Call pattern (from PeriodicMemoryReleaseCallback):

    tracer = MemoryTracer()
    tracer.snapshot("baseline")          # after initial val, before step 1
    tracer.snapshot("step_1000")         # at global_step == 1000
    tracer.report_diff("baseline", "step_1000")
    tracer.report_top("step_1000")
    tracer.snapshot("step_2000")         # at global_step == 2000
    tracer.report_diff("step_1000", "step_2000")

The diff output directly answers the open question:
  - Δheap large   → TCMalloc/glibc allocator holding freed pages (fragmentation)
  - Δfile_mapped  → mmap'd file pages not being released (unlikely: torch.load
                    uses mmap=False by default)
  - Δanon large   → CUDA driver VMM, pinned-memory staging buffers,
                    or anonymous mmap from a C extension

On macOS or non-Linux systems /proc is absent; all methods silently no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_SMAPS_PATH = "/proc/self/smaps"
_STATUS_PATH = "/proc/self/status"


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
    size_kb: int  # virtual size — not necessarily resident


@dataclass
class Snapshot:
    label: str
    regions: list[_Region] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Categorisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _category(r: _Region) -> str:
        p = r.pathname
        if p == "[heap]":
            return "heap"
        if p == "[stack]" or p.startswith("[stack:"):
            return "stack"
        if p in ("[vvar]", "[vdso]", "[vsyscall]", "[vdso]"):
            return "kernel_virt"
        if p and not p.startswith("["):
            # Subdivide file-backed regions for extra signal.
            lp = p.lower()
            if "cuda" in lp or "nvidia" in lp or "/dev/nvidia" in lp:
                return "cuda_driver"
            if "tcmalloc" in lp or "gperftools" in lp:
                return "tcmalloc_lib"
            return "file_mapped"
        # Anonymous (no pathname, no bracket tag): CUDA UVM, pinned staging,
        # anonymous mmap, Python arena segments, etc.
        return "anon"

    # ------------------------------------------------------------------
    # Aggregate summaries
    # ------------------------------------------------------------------

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
        """Private_Dirty by category — pages definitely modified by this process."""
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

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def region_map(self) -> dict[int, _Region]:
        return {r.start: r for r in self.regions}

    def top_regions(self, n: int = 20) -> list[_Region]:
        return sorted(self.regions, key=lambda r: r.rss_kb, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_smaps() -> list[_Region]:
    regions: list[_Region] = []
    try:
        with open(_SMAPS_PATH) as fh:
            current: Optional[dict] = None
            for raw in fh:
                line = raw.rstrip()
                if not line:
                    continue
                # Region header: hex address range + perms + optional pathname
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
    _keys = {"VmRSS", "VmHWM", "VmPeak", "VmData", "VmAnon", "VmPTE"}
    try:
        with open(_STATUS_PATH) as fh:
            for line in fh:
                key = line.split(":")[0]
                if key in _keys:
                    try:
                        fields[key] = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
    except OSError:
        pass
    return fields


# ---------------------------------------------------------------------------
# Public tracer class
# ---------------------------------------------------------------------------


class MemoryTracer:
    """Captures /proc/self/smaps snapshots and diffs them between checkpoints.

    Thread-unsafe by design — intended for single-process, main-thread use only.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}

    def snapshot(self, label: str) -> Snapshot:
        """Capture current smaps state under *label* and print a one-line summary."""
        snap = Snapshot(label=label, regions=_parse_smaps())
        self._snapshots[label] = snap

        status = _read_proc_status()
        summ = snap.summary()
        pd = snap.private_dirty_summary()
        print(
            f"[mem_trace] snap={label!r}"
            f"  heap={summ['heap'] // 1024}MiB"
            f"  anon={summ['anon'] // 1024}MiB"
            f"  cuda_driver={summ['cuda_driver'] // 1024}MiB"
            f"  file_mapped={summ['file_mapped'] // 1024}MiB"
            f"  total_rss={summ['total'] // 1024}MiB"
            f"  VmRSS={status.get('VmRSS', 0) // 1024}MiB"
            f"  VmHWM={status.get('VmHWM', 0) // 1024}MiB"
            f"  private_dirty={pd['total'] // 1024}MiB",
            flush=True,
        )
        return snap

    def report_diff(
        self,
        label_a: str,
        label_b: str,
        top_n: int = 20,
    ) -> None:
        """Print a diff of RSS changes between two snapshots.

        Regions are matched by start address.  Regions that appear/disappear
        between snapshots are listed as new/removed.
        """
        if label_a not in self._snapshots or label_b not in self._snapshots:
            print(
                f"[mem_trace] diff: missing snapshot(s) — "
                f"have {list(self._snapshots.keys())}",
                flush=True,
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

        print(
            f"\n[mem_trace] diff {label_a!r} → {label_b!r}"
            f"  Δtotal={delta_total // 1024:+d}MiB"
            f"  Δprivate_dirty={delta_pd // 1024:+d}MiB",
            flush=True,
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
            if abs(delta_cat) >= 1024:  # only print if >=1 MiB change
                print(
                    f"  Δ{cat}={delta_cat // 1024:+d}MiB"
                    f"  ({summ_a.get(cat, 0) // 1024}→{summ_b.get(cat, 0) // 1024}MiB)",
                    flush=True,
                )

        # Per-region diff
        deltas: list[
            tuple[int, _Region, int, int]
        ] = []  # (delta_kb, region, rss_a, rss_b)
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
            print(
                f"[mem_trace] top {min(top_n, len(growing))} growing regions:",
                flush=True,
            )
            for delta_kb, region, rss_a, rss_b in growing[:top_n]:
                name = region.pathname or f"<anon @{region.start:#x}>"
                cat = Snapshot._category(region)
                print(
                    f"  {delta_kb // 1024:+6d}MiB  [{region.perms}] [{cat}]  {name}"
                    f"  ({rss_a // 1024}→{rss_b // 1024}MiB)"
                    f"  pdirty={region.private_dirty_kb // 1024}MiB",
                    flush=True,
                )

        if shrinking:
            n_show = min(5, len(shrinking))
            print(
                f"[mem_trace] top {n_show} shrinking regions:",
                flush=True,
            )
            for delta_kb, region, rss_a, rss_b in shrinking[-n_show:]:
                name = region.pathname or f"<anon @{region.start:#x}>"
                print(
                    f"  {delta_kb // 1024:+6d}MiB  [{region.perms}]  {name}",
                    flush=True,
                )

        print("", flush=True)

    def report_top(self, label: str, n: int = 20) -> None:
        """Print the top-N RSS-consuming regions for a given snapshot."""
        if label not in self._snapshots:
            return
        snap = self._snapshots[label]
        print(f"[mem_trace] top {n} RSS regions at {label!r}:", flush=True)
        for r in snap.top_regions(n=n):
            name = r.pathname or f"<anon @{r.start:#x}>"
            cat = Snapshot._category(r)
            print(
                f"  {r.rss_kb // 1024:6d}MiB  [{r.perms}] [{cat}]  {name}"
                f"  pdirty={r.private_dirty_kb // 1024}MiB",
                flush=True,
            )
        print("", flush=True)

    def clear(self) -> None:
        """Drop all stored snapshots (call at trial teardown to free memory)."""
        self._snapshots.clear()
