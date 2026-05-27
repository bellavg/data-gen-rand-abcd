from __future__ import annotations

import argparse
import shutil
import sys
import traceback
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Dict, List, Tuple

from tqdm import tqdm

from data.data_utils import default_workers, parse_aig_name

VALID_ALGORITHMS = {"Orchestrate", "Deepsyn", "Syn4", "C2RS"}


@dataclass(frozen=True)
class GraphTask:
    task_id: int
    tier_id: int
    algorithm: str
    design: str
    filename: str
    aig_path: str
    archive_path: str = ""
    archive_member: str = ""


@dataclass(frozen=True)
class WorkerConfig:
    final_out_root: str
    overwrite: bool


def summarize_archive_layout(aig_root: Path) -> Dict[str, object]:
    design_dirs = sorted(
        [
            p
            for p in aig_root.iterdir()
            if p.is_dir()
            and (
                (p / "tier0.zip").exists()
                or (p / "tier0" / "tier0.zip").exists()
                or (p / "tier0").is_dir()
                or (p / "tier1").is_dir()
                or (p / "tier2").is_dir()
            )
        ]
    )
    design_names = [p.name for p in design_dirs]

    tier0_zip_count = 0
    tier1_zip_count = 0
    missing_tier0_designs: List[str] = []
    missing_tier1_designs: List[str] = []
    missing_tier1_by_design: Dict[str, List[str]] = {}

    for design_dir in design_dirs:
        design = design_dir.name

        tier0_zip = design_dir / "tier0.zip"
        tier0_zip_nested = design_dir / "tier0" / "tier0.zip"
        has_tier0_zip = tier0_zip.exists() or tier0_zip_nested.exists()
        if has_tier0_zip:
            tier0_zip_count += 1
        else:
            missing_tier0_designs.append(design)

        tier1_dir = design_dir / "tier1"
        found_algos = set()
        if tier1_dir.is_dir():
            for algo in VALID_ALGORITHMS:
                if (tier1_dir / f"{design}_{algo}.zip").exists():
                    found_algos.add(algo)
                    tier1_zip_count += 1

        if len(found_algos) == 0:
            missing_tier1_designs.append(design)

        missing_algos = sorted(VALID_ALGORITHMS - found_algos)
        if missing_algos:
            missing_tier1_by_design[design] = missing_algos

    expected_designs = len(design_names)
    expected_tier0_zip = expected_designs
    expected_tier1_zip = expected_designs * len(VALID_ALGORITHMS)

    return {
        "design_count": expected_designs,
        "design_names": design_names,
        "tier0_zip_count": tier0_zip_count,
        "tier1_zip_count": tier1_zip_count,
        "expected_tier0_zip": expected_tier0_zip,
        "expected_tier1_zip": expected_tier1_zip,
        "missing_tier0_designs": missing_tier0_designs,
        "missing_tier1_designs": missing_tier1_designs,
        "missing_tier1_by_design": missing_tier1_by_design,
    }


def discover_graph_tasks(
    aig_root: Path, allow_unmatched_names: bool
) -> Tuple[List[GraphTask], int, Dict[str, int]]:
    paths = sorted(aig_root.rglob("*.aig"))
    tasks: List[GraphTask] = []
    unmatched = 0
    source_counts = {
        "filesystem_aig": 0,
        "zip_aig": 0,
        "duplicates_ignored": 0,
        "zip_files_scanned": 0,
        "design_mismatch_rejected": 0,
    }
    seen_ids = set()

    def try_add_task(
        filename: str,
        aig_path: str,
        archive_path: str = "",
        archive_member: str = "",
        expected_design: str = "",
    ) -> None:
        nonlocal unmatched

        parsed = parse_aig_name(filename)
        if parsed is None:
            unmatched += 1
            return

        tier_id, algorithm, design = parsed

        # Guard against ABC mktemp-pollution: if the zip archive tells us which
        # design directory this file came from, the parsed design must match
        # exactly.  A mismatch means a junk token (e.g. tmp1a2b3c) was absorbed
        # into the design group by the non-greedy regex — the file must be
        # cleaned by cleanup_naming.py before preprocessing.
        if expected_design and design != expected_design:
            source_counts["design_mismatch_rejected"] += 1
            return
        key = (tier_id, algorithm, design, filename)
        if key in seen_ids:
            source_counts["duplicates_ignored"] += 1
            return

        seen_ids.add(key)
        tasks.append(
            GraphTask(
                task_id=0,
                tier_id=tier_id,
                algorithm=algorithm,
                design=design,
                filename=filename,
                aig_path=aig_path,
                archive_path=archive_path,
                archive_member=archive_member,
            )
        )

    for path in paths:
        source_counts["filesystem_aig"] += 1
        try_add_task(filename=path.name, aig_path=str(path))

    zip_paths: List[Path] = []
    zip_paths.extend(sorted(aig_root.glob("*/tier0.zip")))
    zip_paths.extend(sorted(aig_root.glob("*/tier0/tier0.zip")))
    zip_paths.extend(sorted(aig_root.glob("*/tier1/*.zip")))
    # Keep deterministic ordering and avoid duplicate scan of the same archive path.
    zip_paths = sorted(set(zip_paths))
    for zip_path in zip_paths:
        # Infer expected design name from the archive's directory layout:
        #   {design}/tier0.zip          → zip_path.parent.name  = design
        #   {design}/tier0/tier0.zip    → zip_path.parent.name  = "tier0" → grandparent
        #   {design}/tier1/{algo}.zip   → zip_path.parent.name  = "tier1" → grandparent
        _zp = zip_path
        _expected_design = (
            _zp.parent.parent.name
            if _zp.parent.name in ("tier0", "tier1")
            else _zp.parent.name
        )
        source_counts["zip_files_scanned"] += 1
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    member = info.filename
                    if not member.lower().endswith(".aig"):
                        continue
                    source_counts["zip_aig"] += 1
                    try_add_task(
                        filename=Path(member).name,
                        aig_path=f"{zip_path}::{member}",
                        archive_path=str(zip_path),
                        archive_member=member,
                        expected_design=_expected_design,
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid zip archive encountered: {zip_path}") from exc

    if unmatched > 0 and not allow_unmatched_names:
        raise ValueError(
            f"Found {unmatched} .aig file(s) with names that do not match tier0/tier1 regex. "
            "Use --allow-unmatched-names to skip them."
        )

    # Assign deterministic task IDs after filtering.
    tasks_with_id: List[GraphTask] = []
    for idx, task in enumerate(tasks, start=1):
        tasks_with_id.append(
            GraphTask(
                task_id=idx,
                tier_id=task.tier_id,
                algorithm=task.algorithm,
                design=task.design,
                filename=task.filename,
                aig_path=task.aig_path,
                archive_path=task.archive_path,
                archive_member=task.archive_member,
            )
        )

    return tasks_with_id, unmatched, source_counts


def artifact_output_base_path(final_out_root: Path, task: GraphTask) -> Path:
    stem = Path(task.filename).stem
    if task.tier_id == 0:
        return final_out_root / "graphs" / "tier0" / task.design / stem
    if task.tier_id == 1:
        return final_out_root / "graphs" / "tier1" / task.algorithm / task.design / stem
    return final_out_root / "graphs" / f"tier{task.tier_id}" / task.design / stem


def graph_output_path(final_out_root: Path, task: GraphTask) -> Path:
    return artifact_output_base_path(final_out_root, task).with_suffix(".pt")


def construct_graph_data_placeholder(aig_path: str) -> object:
    """
    Expected future behavior:
    - Read AIG from aig_path.
    - Construct and return a PyG Data object.
    """
    # Lazy import to keep the main process lightweight.
    from data.data_utils import aig_to_pytorch_geometric

    return aig_to_pytorch_geometric(aig_path)


def save_graph_artifact(out_path: Path, graph_obj: object) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(graph_obj, out_path)


def process_task(task: GraphTask, cfg: WorkerConfig) -> Dict[str, str]:
    started = monotonic()

    result: Dict[str, str] = {
        "task_id": str(task.task_id),
        "tier_id": str(task.tier_id),
        "algorithm": task.algorithm,
        "design": task.design,
        "filename": task.filename,
        "aig_path": task.aig_path,
        "status": "",
        "output_path": "",
        "error": "",
        "duration_s": "0.0",
    }

    try:
        graph_path = graph_output_path(Path(cfg.final_out_root), task)

        if graph_path.exists() and not cfg.overwrite:
            result["status"] = "skipped:exists"
            result["output_path"] = str(graph_path)
            return result

        if task.archive_path:
            with TemporaryDirectory(prefix="preprocess_aig_") as tmp_dir:
                extracted_path = Path(tmp_dir) / task.filename
                with zipfile.ZipFile(task.archive_path, "r") as zf:
                    with (
                        zf.open(task.archive_member, "r") as src,
                        open(extracted_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)
                graph_obj = construct_graph_data_placeholder(str(extracted_path))
        else:
            graph_obj = construct_graph_data_placeholder(task.aig_path)
        if graph_obj is None:
            raise RuntimeError(
                "graph construction returned None; expected a PyG object"
            )

        save_graph_artifact(graph_path, graph_obj)

        result["status"] = "saved:graph"
        result["output_path"] = str(graph_path)
        return result

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{exc}\n{traceback.format_exc(limit=2)}"
        return result
    finally:
        result["duration_s"] = f"{(monotonic() - started):.6f}"


def process_tasks_parallel(
    tasks: List[GraphTask],
    cfg: WorkerConfig,
    workers: int,
    max_in_flight: int,
    fail_fast: bool,
    progress_every: int,
) -> Dict[str, object]:
    status_counts: Dict[str, int] = {}
    submitted = 0
    completed = 0
    errors = 0
    failed_tasks: List[Dict[str, str]] = []

    # Progress bar for visual feedback during parallel processing
    pb = tqdm(
        total=len(tasks),
        desc="preprocessing",
        unit="tasks",
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )

    def drain_completed(pending: Dict[object, int]) -> None:
        nonlocal completed, errors
        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
        for fut in done:
            pending.pop(fut, None)
            result = fut.result()
            completed += 1
            pb.update(1)

            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "error":
                errors += 1
                failed_tasks.append(result)
                print(
                    "issue: "
                    f"task_id={result.get('task_id', '')} design={result.get('design', '')} "
                    f"algo={result.get('algorithm', '')} file={result.get('filename', '')}"
                )
                print(result.get("error", ""))

            if progress_every > 0 and completed % progress_every == 0:
                print(
                    f"processing progress: completed={completed} submitted={submitted} total={len(tasks)}"
                )

            if fail_fast and status == "error":
                raise RuntimeError("fail-fast enabled and a worker error occurred")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending: Dict[object, int] = {}

            for task in tasks:
                while len(pending) >= max_in_flight:
                    drain_completed(pending)

                fut = executor.submit(process_task, task, cfg)
                pending[fut] = task.task_id
                submitted += 1

                if progress_every > 0 and submitted % progress_every == 0:
                    print(
                        f"processing queued: submitted={submitted} in_flight={len(pending)} total={len(tasks)}"
                    )

            while pending:
                drain_completed(pending)
    finally:
        pb.close()

    return {
        "submitted": submitted,
        "completed": completed,
        "errors": errors,
        "status_counts": status_counts,
        "failed_tasks": failed_tasks,
    }


def run_pipeline(args: argparse.Namespace) -> int:
    started = monotonic()

    aig_root = Path(args.aig_root).expanduser().resolve()
    if not aig_root.is_dir():
        raise ValueError(f"aig root does not exist: {aig_root}")

    final_out_root = Path(args.final_out).expanduser().resolve()
    final_out_root.mkdir(parents=True, exist_ok=True)

    layout = summarize_archive_layout(aig_root)
    print(
        "layout: "
        f"designs={layout['design_count']} "
        f"tier0_zip={layout['tier0_zip_count']}/{layout['expected_tier0_zip']} "
        f"tier1_zip={layout['tier1_zip_count']}/{layout['expected_tier1_zip']} "
        f"(expected tier1 = designs x {len(VALID_ALGORITHMS)})"
    )
    if layout["missing_tier0_designs"]:
        print(
            "layout: missing tier0.zip designs (head): "
            + ", ".join(layout["missing_tier0_designs"][:15])
        )
    if layout["missing_tier1_designs"]:
        print(
            "layout: missing all tier1 zips designs (head): "
            + ", ".join(layout["missing_tier1_designs"][:15])
        )

    missing_tier1_by_design = layout["missing_tier1_by_design"]
    if missing_tier1_by_design:
        examples = []
        for design in sorted(missing_tier1_by_design)[:10]:
            missing_algos = ",".join(missing_tier1_by_design[design])
            examples.append(f"{design}:[{missing_algos}]")
        print("layout: missing tier1 algo zips examples: " + "; ".join(examples))

    print(f"discovery: scanning .aig files under {aig_root} and inside .zip archives")
    tasks, unmatched, source_counts = discover_graph_tasks(
        aig_root, allow_unmatched_names=bool(args.allow_unmatched_names)
    )
    discovered_total = len(tasks)

    design_mismatch_rejected = source_counts.get("design_mismatch_rejected", 0)
    if design_mismatch_rejected > 0:
        print(
            f"discovery: WARNING {design_mismatch_rejected} AIG file(s) rejected because "
            "the parsed design name did not match the archive directory. "
            "This indicates ABC mktemp-pollution junk tokens in zip member names. "
            "Run cleanup_naming.py (or inspect_data.py + cleanup_naming.py) to fix "
            "the zip archives before re-running preprocessing."
        )
    print(
        "discovery: "
        f"matched={discovered_total} unmatched={unmatched} "
        f"design_mismatch_rejected={design_mismatch_rejected} "
        f"filesystem_aig={source_counts.get('filesystem_aig', 0)} "
        f"zip_aig={source_counts.get('zip_aig', 0)} "
        f"zip_files_scanned={source_counts.get('zip_files_scanned', 0)} "
        f"duplicates_ignored={source_counts.get('duplicates_ignored', 0)} "
        f"(tier0/tier1 inferred by filename regex)"
    )

    if discovered_total == 0:
        raise ValueError("no matching tier0/tier1 AIG files found")

    stats = process_tasks_parallel(
        tasks=tasks,
        cfg=WorkerConfig(
            final_out_root=str(final_out_root),
            overwrite=bool(args.overwrite),
        ),
        workers=args.workers,
        max_in_flight=max(args.workers, args.max_in_flight),
        fail_fast=bool(args.fail_fast),
        progress_every=args.progress_every,
    )

    submitted = int(stats["submitted"])
    completed = int(stats["completed"])
    errors = int(stats["errors"])
    status_counts = dict(stats["status_counts"])
    failed_tasks: List[Dict[str, str]] = list(stats["failed_tasks"])

    output_like = status_counts.get("saved:graph", 0) + status_counts.get(
        "skipped:exists", 0
    )

    # Required equivalence checks.
    if completed != discovered_total:
        raise RuntimeError(
            f"count mismatch after processing: discovered={discovered_total} completed={completed}"
        )

    if output_like != discovered_total:
        raise RuntimeError(
            f"output-count mismatch: expected={discovered_total} output_like={output_like} status_counts={status_counts}"
        )

    print("overview:")
    print(f"  discovered: {discovered_total}")
    print(f"  submitted: {submitted}")
    print(f"  completed: {completed}")
    print(f"  saved: {status_counts.get('saved:graph', 0)}")
    print(f"  skipped_existing: {status_counts.get('skipped:exists', 0)}")
    print(f"  errors: {errors}")
    print(f"  unmatched_names: {unmatched}")

    if errors > 0:
        print("failed task recap:")
        for item in failed_tasks:
            print(
                "  - "
                f"task_id={item.get('task_id', '')} design={item.get('design', '')} "
                f"algo={item.get('algorithm', '')} file={item.get('filename', '')}"
            )
        raise RuntimeError(f"processing completed with {errors} error(s)")

    elapsed_s = round(monotonic() - started, 3)
    print(
        "done: "
        f"discovered={discovered_total} completed={completed} output_like={output_like} "
        f"elapsed_s={elapsed_s}"
    )

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel preprocessing scaffold for AIG -> graph pipeline (graph builder TODO)."
    )
    parser.add_argument(
        "--aig-root",
        required=True,
        help="Overarching root directory to recursively glob for .aig files",
    )
    parser.add_argument(
        "--final-out",
        required=True,
        help="Final destination root for saved PyG artifacts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Worker process count (defaults to SLURM_CPUS_PER_TASK or 24 when available)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Max queued futures. 0 means workers*8",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop on first worker error (default: enabled)",
    )
    parser.add_argument(
        "--allow-unmatched-names",
        action="store_true",
        help="Skip .aig files that do not match the tier0/tier1 filename regex",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing graph artifacts when present (default: disabled)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print queue/completion updates every N tasks (default: 10000). Set 0 to disable.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_in_flight <= 0:
        args.max_in_flight = max(1, args.workers * 8)
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
