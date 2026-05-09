"""Naming cleanup for AIG ZIPs, PT graph files, and CSV metadata.

The audit tool (inspect_data.py) emits naming-candidate CSVs that describe
exactly which files/paths have the ABC mktemp-pollution junk token embedded in
their names.  This script consumes those CSVs and applies (or simulates) the
renames.

Workflow
--------
1. Run in dry-run mode first to preview changes and verify the candidate CSVs
   are correct.
2. After inspection, rerun with --apply to perform the actual renames.
3. Built-in post-apply verification is always run and exits non-zero on
   mismatch.

Phases (all three run by default; use --phase to restrict):
  pt    Rename Tier-1 .pt graph files on disk.
  csv   Rewrite CSV metadata files, replacing messy paths with clean paths.
  zip   Rewrite AIG archive .zip files, renaming embedded members.

Expected naming conventions (per DATA_README.md):
  Tier-0 .aig  : {design}_syn{recipe}_step{step}.aig
  Tier-1 .aig  : {design}_{algorithm}_tier1_syn{recipe}_step{step}.aig
  Tier-2 .aig  : {design}_{src_algo}_{dst_algo}_tier2_syn{recipe}_step{step}.aig
  Tier-0 .pt   : {design}_syn{recipe}_step{step}.pt
  Tier-1 .pt   : {design}_{algorithm}_tier1_syn{recipe}_step{step}.pt
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Candidate CSV loaders
# ---------------------------------------------------------------------------


def load_pt_candidates(candidates_csv: Path) -> list[tuple[Path, Path]]:
    """Return (src_path, dst_path) pairs from pt_naming_candidates.csv."""
    pairs: list[tuple[Path, Path]] = []
    with open(candidates_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src = Path(row["pt_path"])
            dst = Path(row["suggested_path"])
            if src != dst:
                pairs.append((src, dst))
    return pairs


def load_csv_candidates(
    candidates_csv: Path,
) -> dict[str, dict[str, str]]:
    """Return {csv_file: {original_path: suggested_path}} from csv_naming_candidates.csv.

    This builds the full mapping in memory (≈1-2 GB for 5.7M rows).
    """
    mapping: dict[str, dict[str, str]] = defaultdict(dict)
    with open(candidates_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["original_path"] != row["suggested_path"]:
                mapping[row["csv_file"]][row["original_path"]] = row["suggested_path"]
    return dict(mapping)


def load_zip_candidates(
    candidates_csv: Path,
) -> dict[str, dict[str, str]]:
    """Return {zip_path: {member_path: suggested_member_path}} from aig_zip_naming_candidates.csv."""
    mapping: dict[str, dict[str, str]] = defaultdict(dict)
    with open(candidates_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["member_path"] != row["suggested_member_path"]:
                mapping[row["zip_path"]][row["member_path"]] = row[
                    "suggested_member_path"
                ]
    return dict(mapping)


# ---------------------------------------------------------------------------
# PT rename
# ---------------------------------------------------------------------------

_PT_STATUS = (
    "renamed",
    "already_renamed",
    "skipped_dst_exists",
    "skipped_missing_src",
    "errors",
)


def _do_pt_rename(pair: tuple[Path, Path]) -> str:
    """ThreadPoolExecutor worker: rename one PT file, return status key."""
    src, dst = pair
    src_ex = src.exists()
    dst_ex = dst.exists()
    if not src_ex and dst_ex:
        return "already_renamed"
    if not src_ex:
        return "skipped_missing_src"
    if dst_ex:
        return "skipped_dst_exists"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return "renamed"
    except OSError:
        return "errors"


def apply_pt_renames(
    pairs: list[tuple[Path, Path]],
    *,
    dry_run: bool,
    workers: int = 8,
) -> dict[str, int]:
    """Rename PT files in parallel batches.

    Dry-run mode counts only (no I/O, no workers).
    Apply mode splits pairs into batches and processes them in a
    ProcessPoolExecutor so that concurrent renames hit different
    directories in parallel (important on Lustre/NFS scratch).
    """
    counts: dict[str, int] = dict.fromkeys(_PT_STATUS, 0)
    if not pairs:
        return counts

    if dry_run:
        # Stat calls are I/O-bound on NFS/Lustre — parallelise with threads.
        def _check_pair(pair: tuple[Path, Path]) -> str:
            src, dst = pair
            src_ex = src.exists()
            dst_ex = dst.exists()
            if not src_ex and dst_ex:
                return "already_renamed"
            if not src_ex:
                return "skipped_missing_src"
            if dst_ex:
                return "skipped_dst_exists"
            return "renamed"

        n = max(1, workers)
        with ThreadPoolExecutor(max_workers=n) as executor:
            results = list(
                tqdm(
                    executor.map(_check_pair, pairs),
                    total=len(pairs),
                    desc="Counting PT renames (dry-run)",
                    unit="pt",
                )
            )
        for status in results:
            counts[status] += 1
        return counts

    n = max(1, workers)
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(
            tqdm(
                executor.map(_do_pt_rename, pairs),
                total=len(pairs),
                desc="Renaming PT files",
                unit="pt",
            )
        )
    for status in results:
        counts[status] = counts.get(status, 0) + 1
    return counts


def verify_pt_renames(
    pairs: list[tuple[Path, Path]], workers: int = 8
) -> dict[str, int]:
    counts = {"ok": 0, "src_still_exists": 0, "dst_missing": 0}

    def _check(pair: tuple[Path, Path]) -> str:
        src, dst = pair
        if dst.exists() and not src.exists():
            return "ok"
        if src.exists():
            return "src_still_exists"
        return "dst_missing"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for status in executor.map(_check, pairs):
            counts[status] += 1
    return counts


# ---------------------------------------------------------------------------
# CSV rewrite
# ---------------------------------------------------------------------------

_TARGET_COLUMNS = ("file_path", "unoptimized_graph_path")


def rewrite_csv_file(
    csv_file: Path,
    path_map: dict[str, str],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Rewrite one CSV file replacing messy paths with canonical ones.

    Uses a .rewriting temp file and atomically replaces the original on
    success.  In dry-run mode the temp file is removed after counting.
    """
    counts: dict[str, int] = {"rows_changed": 0, "rows_unchanged": 0, "errors": 0}
    tmp_path = csv_file.with_suffix(".csv.rewriting")
    try:
        with (
            open(csv_file, newline="", encoding="utf-8", errors="ignore") as in_fh,
            open(tmp_path, "w", newline="", encoding="utf-8") as out_fh,
        ):
            reader = csv.DictReader(in_fh)
            if reader.fieldnames is None:
                return counts
            writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                changed = False
                for col in _TARGET_COLUMNS:
                    v = row.get(col)
                    if v and v in path_map:
                        row[col] = path_map[v]
                        changed = True
                writer.writerow(row)
                if changed:
                    counts["rows_changed"] += 1
                else:
                    counts["rows_unchanged"] += 1
    except OSError as exc:
        print(f"[warn] CSV rewrite error for {csv_file}: {exc}", file=sys.stderr)
        counts["errors"] += 1
        tmp_path.unlink(missing_ok=True)
        return counts

    if dry_run:
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(csv_file)
    return counts


def _rewrite_csv_worker(
    args: tuple[str, dict[str, str], bool],
) -> tuple[str, dict[str, int]]:
    """ThreadPoolExecutor worker: rewrite one CSV file."""
    csv_file_str, path_map, dry_run = args
    csv_file = Path(csv_file_str)
    if not csv_file.exists():
        print(f"[warn] CSV not found, skipping: {csv_file}", file=sys.stderr)
        return csv_file_str, {"rows_changed": 0, "rows_unchanged": 0, "errors": 1}
    return csv_file_str, rewrite_csv_file(csv_file, path_map, dry_run=dry_run)


def apply_csv_rewrites(
    mapping: dict[str, dict[str, str]],
    *,
    dry_run: bool,
    workers: int = 8,
) -> dict[str, int]:
    """Rewrite all CSV files concurrently (one thread per file)."""
    totals: dict[str, int] = {"rows_changed": 0, "rows_unchanged": 0, "errors": 0}
    work = [(f, m, dry_run) for f, m in mapping.items()]
    n = min(max(1, workers), len(work))
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(_rewrite_csv_worker, item): item[0] for item in work}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Rewriting CSVs", unit="csv"
        ):
            _, counts = future.result()
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
    return totals


def verify_csv_rewrites(
    mapping: dict[str, dict[str, str]],
) -> dict[str, int]:
    """Check that no original messy paths remain in the CSV files."""
    counts = {"ok": 0, "still_messy": 0, "errors": 0}
    for csv_file_str, path_map in tqdm(
        mapping.items(), desc="Verifying CSVs", unit="csv"
    ):
        csv_file = Path(csv_file_str)
        if not csv_file.exists():
            counts["errors"] += 1
            continue
        messy_paths = set(path_map.keys())
        try:
            with open(csv_file, newline="", encoding="utf-8", errors="ignore") as fh:
                for row in csv.DictReader(fh):
                    for col in _TARGET_COLUMNS:
                        if row.get(col) in messy_paths:
                            counts["still_messy"] += 1
        except OSError:
            counts["errors"] += 1
            continue
        else:
            counts["ok"] += 1
    return counts


# ---------------------------------------------------------------------------
# ZIP rewrite
# ---------------------------------------------------------------------------


def _rewrite_single_zip(
    args_tuple: tuple[str, dict[str, str], bool],
) -> dict[str, object]:
    """Worker function: rewrite one ZIP with renamed members.

    Returns a result dict so it can be used with ProcessPoolExecutor.
    """
    zip_path_str, member_map, dry_run = args_tuple
    zip_path = Path(zip_path_str)
    tmp_zip = zip_path.with_suffix(".zip.rewriting")
    result: dict[str, object] = {
        "zip_path": zip_path_str,
        "members_renamed": 0,
        "members_unchanged": 0,
        "errors": 0,
        "error_msg": "",
    }

    try:
        with (
            zipfile.ZipFile(zip_path, "r") as zin,
            zipfile.ZipFile(tmp_zip, "w") as zout,
        ):
            for info in zin.infolist():
                new_name = member_map.get(info.filename, info.filename)
                new_info = zipfile.ZipInfo(new_name, date_time=info.date_time)
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                with zin.open(info) as src_fh, zout.open(new_info, "w") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh)
                if new_name != info.filename:
                    result["members_renamed"] = int(result["members_renamed"]) + 1
                else:
                    result["members_unchanged"] = int(result["members_unchanged"]) + 1
    except (OSError, zipfile.BadZipFile) as exc:
        result["errors"] = 1
        result["error_msg"] = str(exc)
        tmp_zip.unlink(missing_ok=True)
        return result

    # Verify member count parity before replacing.
    try:
        with (
            zipfile.ZipFile(tmp_zip, "r") as ztmp,
            zipfile.ZipFile(zip_path, "r") as zorig,
        ):
            if len(ztmp.infolist()) != len(zorig.infolist()):
                result["errors"] = 1
                result["error_msg"] = (
                    f"member count mismatch: tmp={len(ztmp.infolist())} "
                    f"orig={len(zorig.infolist())}"
                )
                tmp_zip.unlink(missing_ok=True)
                return result
    except (OSError, zipfile.BadZipFile) as exc:
        result["errors"] = 1
        result["error_msg"] = f"post-write verify failed: {exc}"
        tmp_zip.unlink(missing_ok=True)
        return result

    if dry_run:
        tmp_zip.unlink(missing_ok=True)
    else:
        tmp_zip.replace(zip_path)

    return result


def apply_zip_rewrites(
    mapping: dict[str, dict[str, str]],
    *,
    dry_run: bool,
    workers: int = 4,
) -> dict[str, int]:
    totals: dict[str, int] = {
        "zips_processed": 0,
        "members_renamed": 0,
        "members_unchanged": 0,
        "errors": 0,
    }
    work = [(p, m, dry_run) for p, m in mapping.items()]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_rewrite_single_zip, item): item[0] for item in work}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Rewriting ZIPs", unit="zip"
        ):
            result = future.result()
            totals["zips_processed"] += 1
            totals["members_renamed"] += int(result["members_renamed"])
            totals["members_unchanged"] += int(result["members_unchanged"])
            totals["errors"] += int(result["errors"])
            if result["errors"]:
                print(
                    f"[warn] ZIP error {result['zip_path']}: {result['error_msg']}",
                    file=sys.stderr,
                )
    return totals


def verify_zip_rewrites(mapping: dict[str, dict[str, str]]) -> dict[str, int]:
    """Check that no original messy member names remain in the ZIP files."""
    counts = {"ok": 0, "still_messy": 0, "errors": 0}
    for zip_path_str, member_map in tqdm(
        mapping.items(), desc="Verifying ZIPs", unit="zip"
    ):
        zip_path = Path(zip_path_str)
        messy_names = set(member_map.keys())
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                found_messy = any(i.filename in messy_names for i in zf.infolist())
            if found_messy:
                counts["still_messy"] += 1
            else:
                counts["ok"] += 1
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"[warn] ZIP verify error {zip_path}: {exc}", file=sys.stderr)
            counts["errors"] += 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_PHASES = ("pt", "csv", "zip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename messy AIG/PT/CSV artifacts to canonical names (per DATA_README.md). "
            "Runs in dry-run mode by default — pass --apply to perform real renames."
        )
    )
    parser.add_argument(
        "--issues-dir",
        type=Path,
        required=True,
        help="Directory produced by inspect_data.py --issues-out-dir (contains *_naming_candidates.csv files).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually perform renames.  Without this flag the script is read-only.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=_ALL_PHASES,
        default=None,
        help="Which cleanup phase(s) to run.  Can be passed multiple times.  Default: all phases.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Parallel workers used for all phases: "
            "ProcessPoolExecutor batch-renames for PT, "
            "ThreadPoolExecutor per-file for CSV, "
            "ProcessPoolExecutor per-zip for ZIP (default: 8)."
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="Skip post-apply verification (not recommended).",
    )
    return parser


def _print_counts(label: str, counts: dict[str, int]) -> None:
    print(f"\n  {label}:")
    for k, v in sorted(counts.items()):
        print(f"    {k}: {v:,}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    issues_dir = args.issues_dir
    if not issues_dir.is_dir():
        print(f"[error] issues-dir not found: {issues_dir}", file=sys.stderr)
        return 1

    phases = set(args.phase or _ALL_PHASES)
    dry_run = not args.apply
    mode = "DRY-RUN" if dry_run else "APPLY"

    print("==========================================")
    print(f"NAMING CLEANUP  [{mode}]")
    print("==========================================")
    print(f"Issues dir : {issues_dir}")
    print(f"Phases     : {', '.join(sorted(phases))}")
    print(f"Workers    : {args.workers}")
    print(f"Verify     : {not args.no_verify}")
    if dry_run:
        print("\nNOTE: --apply not passed. No filesystem changes will be made.\n")

    verify_failed = False

    # -----------------------------------------------------------------------
    # PT phase
    # -----------------------------------------------------------------------
    if "pt" in phases:
        print("\n==========================================")
        print("PHASE: PT FILE RENAMES")
        print("==========================================")
        candidates_csv = issues_dir / "pt_naming_candidates.csv"
        if not candidates_csv.exists():
            print(
                f"[warn] pt_naming_candidates.csv not found in {issues_dir}, skipping."
            )
        else:
            print(f"Loading PT candidates from {candidates_csv} …")
            pairs = load_pt_candidates(candidates_csv)
            print(f"PT rename pairs: {len(pairs):,}")
            counts = apply_pt_renames(pairs, dry_run=dry_run, workers=args.workers)
            _print_counts("PT rename results", counts)

            if not dry_run and not args.no_verify:
                print("\nVerifying PT renames …")
                vcounts = verify_pt_renames(pairs, workers=args.workers)
                _print_counts("PT verify", vcounts)
                if vcounts.get("src_still_exists", 0) or vcounts.get("dst_missing", 0):
                    print(
                        "[FAIL] PT verification: some renames incomplete.",
                        file=sys.stderr,
                    )
                    verify_failed = True
                else:
                    print("PT verification: OK")

    # -----------------------------------------------------------------------
    # CSV phase
    # -----------------------------------------------------------------------
    if "csv" in phases:
        print("\n==========================================")
        print("PHASE: CSV METADATA REWRITES")
        print("==========================================")
        candidates_csv = issues_dir / "csv_naming_candidates.csv"
        if not candidates_csv.exists():
            print(
                f"[warn] csv_naming_candidates.csv not found in {issues_dir}, skipping."
            )
        else:
            print(
                f"Loading CSV candidates from {candidates_csv} "
                "(this reads ~2 GB into memory) …"
            )
            mapping = load_csv_candidates(candidates_csv)
            print(f"CSV files to rewrite: {len(mapping):,}")
            total_paths = sum(len(v) for v in mapping.values())
            print(f"Total path substitutions: {total_paths:,}")
            counts = apply_csv_rewrites(mapping, dry_run=dry_run, workers=args.workers)
            _print_counts("CSV rewrite results", counts)

            if not dry_run and not args.no_verify:
                print("\nVerifying CSV rewrites …")
                vcounts = verify_csv_rewrites(mapping)
                _print_counts("CSV verify", vcounts)
                if vcounts.get("still_messy", 0):
                    print(
                        "[FAIL] CSV verification: messy paths still found.",
                        file=sys.stderr,
                    )
                    verify_failed = True
                else:
                    print("CSV verification: OK")

    # -----------------------------------------------------------------------
    # ZIP phase
    # -----------------------------------------------------------------------
    if "zip" in phases:
        print("\n==========================================")
        print("PHASE: AIG ZIP MEMBER RENAMES")
        print("==========================================")
        candidates_csv = issues_dir / "aig_zip_naming_candidates.csv"
        if not candidates_csv.exists():
            print(
                f"[warn] aig_zip_naming_candidates.csv not found in {issues_dir}, skipping."
            )
        else:
            print(f"Loading ZIP candidates from {candidates_csv} …")
            mapping_zip = load_zip_candidates(candidates_csv)
            print(f"ZIP files to rewrite: {len(mapping_zip):,}")
            total_members = sum(len(v) for v in mapping_zip.values())
            print(f"Total member renames: {total_members:,}")
            counts = apply_zip_rewrites(
                mapping_zip, dry_run=dry_run, workers=args.workers
            )
            _print_counts("ZIP rewrite results", counts)

            if not dry_run and not args.no_verify:
                print("\nVerifying ZIP member renames …")
                vcounts = verify_zip_rewrites(mapping_zip)
                _print_counts("ZIP verify", vcounts)
                if vcounts.get("still_messy", 0):
                    print(
                        "[FAIL] ZIP verification: messy member names still found.",
                        file=sys.stderr,
                    )
                    verify_failed = True
                else:
                    print("ZIP verification: OK")

    print("\n==========================================")
    if dry_run:
        print("DRY-RUN COMPLETE — no files were modified.")
        print("Re-run with --apply to perform actual renames.")
    elif verify_failed:
        print("CLEANUP COMPLETE WITH VERIFICATION FAILURES — see above.")
        return 2
    else:
        print("CLEANUP COMPLETE — all verifications passed.")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
