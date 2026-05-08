from __future__ import annotations

import argparse
import csv
import glob
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

ALGORITHMS = ("Orchestrate", "Deepsyn", "Syn4", "C2RS")
ALGO_ALT = "(?:" + "|".join(ALGORITHMS) + ")"

TIER2_RE = re.compile(
    rf"^(?P<design>.+?)_(?P<src_algorithm>{ALGO_ALT})_(?P<dst_algorithm>{ALGO_ALT})_tier2_(?:(?P<junk>.+?)_)?syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)$"
)
TIER1_RE = re.compile(
    rf"^(?P<design>.+?)_(?P<algorithm>{ALGO_ALT})_tier1_(?:(?P<junk>.+?)_)?syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)$"
)
TIER0_RE = re.compile(r"^(?P<design>.+?)_syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)$")
TIER0_BARE_RE = re.compile(r"^syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)$")
TIER0_SUFFIX_RE = re.compile(
    r"^(?P<prefix>.+?)_syn(?P<recipe>[0-9X]+)_step(?P<step>\d+)$"
)


def init_tier_sets() -> dict[int, set[tuple[str, ...]]]:
    return {0: set(), 1: set(), 2: set()}


def format_key(tier: int, key: tuple[str, ...]) -> str:
    if tier == 0:
        design, recipe, step = key
        return f"T0_{design}_syn{recipe}_step{step}"
    if tier == 1:
        design, algorithm, recipe, step = key
        return f"T1_{design}_{algorithm}_syn{recipe}_step{step}"
    if tier == 2:
        design, src_algorithm, dst_algorithm, recipe, step = key
        return f"T2_{design}_{src_algorithm}_{dst_algorithm}_syn{recipe}_step{step}"
    return f"tier{tier}:{key}"


def parse_artifact_name(
    name: str,
    *,
    design_hint: str | None = None,
    tier_hint: int | None = None,
) -> tuple[int, tuple[str, ...] | None, str]:
    """
    Parse artifact filename into (tier, canonical_tuple, naming_state).
    naming_state is one of: clean, messy, unrecognized.
    """
    stem = Path(name).stem

    m2 = TIER2_RE.match(stem)
    if m2:
        key = (
            m2.group("design"),
            m2.group("src_algorithm"),
            m2.group("dst_algorithm"),
            m2.group("recipe"),
            m2.group("step"),
        )
        state = "messy" if m2.group("junk") else "clean"
        return 2, key, state

    m1 = TIER1_RE.match(stem)
    if m1:
        key = (
            m1.group("design"),
            m1.group("algorithm"),
            m1.group("recipe"),
            m1.group("step"),
        )
        state = "messy" if m1.group("junk") else "clean"
        return 1, key, state

    # Tier0 reconciliation can use design hint from path context.
    if tier_hint == 0 and design_hint:
        m0_bare = TIER0_BARE_RE.match(stem)
        if m0_bare:
            return (
                0,
                (design_hint, m0_bare.group("recipe"), m0_bare.group("step")),
                "messy",
            )

        m0_suffix = TIER0_SUFFIX_RE.match(stem)
        if m0_suffix:
            state = "clean" if m0_suffix.group("prefix") == design_hint else "messy"
            return (
                0,
                (design_hint, m0_suffix.group("recipe"), m0_suffix.group("step")),
                state,
            )

    m0 = TIER0_RE.match(stem)
    if m0:
        return 0, (m0.group("design"), m0.group("recipe"), m0.group("step")), "clean"

    return -1, None, "unrecognized"


def parse_csv_file_path(path_str: str) -> tuple[int, tuple[str, ...] | None, str]:
    path = Path(path_str)
    parts = path.parts
    design_hint = None
    tier_hint = None

    for idx, part in enumerate(parts):
        if part == "base_aigs" and idx + 1 < len(parts):
            design_hint = parts[idx + 1]
            break

    for idx, part in enumerate(parts):
        if part.startswith("tier") and idx > 0 and parts[idx - 1] == "base_aigs":
            # Handles paths directly below base_aigs; kept for completeness.
            try:
                tier_hint = int(part.replace("tier", ""))
            except ValueError:
                tier_hint = None
            break
        if part.startswith("tier") and idx > 1 and parts[idx - 2] == "base_aigs":
            try:
                tier_hint = int(part.replace("tier", ""))
            except ValueError:
                tier_hint = None
            break

    # More robust tier extraction for "base_aigs/<design>/tierX/..."
    if tier_hint is None:
        for idx, part in enumerate(parts):
            if part == "base_aigs" and idx + 2 < len(parts):
                tier_part = parts[idx + 2]
                if tier_part.startswith("tier"):
                    try:
                        tier_hint = int(tier_part.replace("tier", ""))
                    except ValueError:
                        tier_hint = None
                break

    return parse_artifact_name(
        path.name,
        design_hint=design_hint,
        tier_hint=tier_hint,
    )


def parse_graph_pt_path(path_str: str) -> tuple[int, tuple[str, ...] | None, str]:
    """Parse a .pt path from graph storage (artifact or ML input column)."""
    path = Path(path_str)
    parts = path.parts
    tier_hint = None
    design_hint = None

    for idx, part in enumerate(parts):
        if part != "graphs" or idx + 1 >= len(parts):
            continue

        tier_part = parts[idx + 1]
        if tier_part.startswith("tier"):
            try:
                tier_hint = int(tier_part.replace("tier", ""))
            except ValueError:
                tier_hint = None

            if tier_hint == 0 and idx + 2 < len(parts):
                design_hint = parts[idx + 2]
            elif tier_hint == 1 and idx + 3 < len(parts):
                design_hint = parts[idx + 3]
            elif tier_hint is not None and idx + 2 < len(parts):
                design_hint = parts[idx + 2]
            break

    return parse_artifact_name(
        path.name,
        design_hint=design_hint,
        tier_hint=tier_hint,
    )


def discover_csv_files(
    csv_files: list[str],
    csv_globs: list[str],
) -> list[Path]:
    paths: set[Path] = set()
    for file_path in csv_files:
        p = Path(file_path)
        if p.is_file():
            paths.add(p)

    for pattern in csv_globs:
        for match in glob.glob(pattern):
            p = Path(match)
            if p.is_file():
                paths.add(p)

    return sorted(paths)


def discover_zip_files(
    aig_root: Path,
    tier2_roots: Iterable[Path],
) -> list[Path]:
    zip_paths: set[Path] = set()

    if aig_root.is_dir():
        zip_paths.update(aig_root.glob("*/tier0.zip"))
        zip_paths.update(aig_root.glob("*/tier0/tier0.zip"))
        zip_paths.update(aig_root.glob("*/tier1/*.zip"))
        zip_paths.update(aig_root.glob("*/tier2/*.zip"))

    for root in tier2_roots:
        if root.is_dir():
            zip_paths.update(root.rglob("*.zip"))

    return sorted(zip_paths)


def scan_single_zip(zip_path: str) -> dict[str, object]:
    tier_keys = init_tier_sets()
    state_counts: dict[int, Counter[str]] = defaultdict(Counter)
    unrecognized = 0
    aig_files = 0
    error = ""

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".aig"):
                    continue

                aig_files += 1
                tier, key, state = parse_artifact_name(Path(info.filename).name)
                if key is None:
                    unrecognized += 1
                    continue

                tier_keys[tier].add(key)
                state_counts[tier][state] += 1
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "zip_path": zip_path,
        "tier_keys": tier_keys,
        "state_counts": state_counts,
        "unrecognized": unrecognized,
        "aig_files": aig_files,
        "error": error,
    }


def summarize_examples(
    keys: set[tuple[str, ...]],
    tier: int,
    max_examples: int,
) -> str:
    if not keys:
        return "none"
    samples = sorted(keys)[:max_examples]
    return ", ".join(format_key(tier, key) for key in samples)


def print_artifact_tier_summary(
    tier: int,
    aig_keys: dict[int, set[tuple[str, ...]]],
    pt_keys: dict[int, set[tuple[str, ...]]],
    csv_artifact_keys: dict[int, set[tuple[str, ...]]],
    max_examples: int,
) -> None:
    tier_name = f"Tier {tier}"
    print(f"\n--- {tier_name} Artifact Reconciliation ---")
    print(f"  AIG keys        : {len(aig_keys[tier]):,}")
    print(f"  PT keys         : {len(pt_keys[tier]):,}")
    print(f"  CSV artifact    : {len(csv_artifact_keys[tier]):,}")

    if tier in (0, 1):
        perfect = aig_keys[tier] & pt_keys[tier] & csv_artifact_keys[tier]
        missing_pt = (aig_keys[tier] & csv_artifact_keys[tier]) - pt_keys[tier]
        csv_ghost = csv_artifact_keys[tier] - aig_keys[tier]
        pt_orphan = pt_keys[tier] - aig_keys[tier]

        print(f"  Perfect (AIG∩PT∩CSV): {len(perfect):,}")
        print(f"  Missing PT (AIG∩CSV - PT): {len(missing_pt):,}")
        print(f"  CSV without AIG: {len(csv_ghost):,}")
        print(f"  PT without AIG : {len(pt_orphan):,}")

        if missing_pt:
            print(
                "  Missing PT examples: "
                + summarize_examples(missing_pt, tier, max_examples)
            )
    else:
        aig_csv = aig_keys[tier] & csv_artifact_keys[tier]
        csv_ghost = csv_artifact_keys[tier] - aig_keys[tier]
        aig_unmatched = aig_keys[tier] - csv_artifact_keys[tier]

        print(f"  AIG∩CSV: {len(aig_csv):,}")
        print(f"  CSV without AIG: {len(csv_ghost):,}")
        print(f"  AIG without CSV: {len(aig_unmatched):,}")
        print(
            "  Note: tier2 PT artifacts are not expected in your training pipeline; "
            "tier2 rows use tier1 input PTs."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit: reconcile AIG ZIP members, PT graph artifacts, and CSV "
            "metadata (supports both file_path and unoptimized_graph_path schemas)."
        )
    )
    parser.add_argument(
        "--aig-root",
        type=Path,
        required=True,
        help="Root containing design tier0/tier1 archives (e.g. .../data/designs)",
    )
    parser.add_argument(
        "--tier2-aig-root",
        action="append",
        default=[],
        help=(
            "Optional extra root to scan for tier2 AIG zip archives "
            "(can be passed multiple times)."
        ),
    )
    parser.add_argument(
        "--pt-root",
        type=Path,
        required=True,
        help="Root containing graphs/ tier folders (e.g. /scratch-shared/$USER)",
    )
    parser.add_argument(
        "--csv-file",
        action="append",
        default=[],
        help="Explicit CSV file path (can be passed multiple times).",
    )
    parser.add_argument(
        "--csv-glob",
        action="append",
        default=[],
        help=(
            "Glob pattern for CSV discovery, e.g. '/path/design_metadata/algo_*_ml.csv' "
            "(can be passed multiple times)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Process workers for ZIP scanning.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of example IDs to print for mismatch sets.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tier2_roots = [Path(p) for p in args.tier2_aig_root]

    csv_globs = list(args.csv_glob)
    if not args.csv_file and not csv_globs:
        csv_globs = [
            str(args.aig_root / "design_metadata" / "algo_*_ml.csv"),
            str(args.aig_root / "design_metadata" / "full_master*.csv"),
        ]

    csv_paths = discover_csv_files(args.csv_file, csv_globs)
    zip_paths = discover_zip_files(args.aig_root, tier2_roots)

    print("==========================================")
    print("CONFIG")
    print("==========================================")
    print(f"AIG root: {args.aig_root}")
    print(f"Tier2 AIG roots: {[str(p) for p in tier2_roots] if tier2_roots else '[]'}")
    print(f"PT root: {args.pt_root}")
    print(f"CSV files discovered: {len(csv_paths):,}")
    print(f"ZIP files discovered: {len(zip_paths):,}")
    print(f"Workers: {args.workers}")

    # ------------------------------------------------------------------
    # 1) CSV scan (supports both schemas)
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("1. SCANNING CSV METADATA")
    print("==========================================")

    csv_artifact_keys = init_tier_sets()
    csv_input_keys = init_tier_sets()
    csv_artifact_states: dict[int, Counter[str]] = defaultdict(Counter)
    csv_input_states: dict[int, Counter[str]] = defaultdict(Counter)
    csv_schema_counts = Counter()
    csv_rows_total = 0
    csv_artifact_unrecognized = 0
    csv_input_unrecognized = 0

    for csv_path in tqdm(csv_paths, desc="Reading CSVs", unit="csv"):
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
                reader = csv.DictReader(fh)
                fieldnames = set(reader.fieldnames or [])

                has_file_path = "file_path" in fieldnames
                has_unoptimized = "unoptimized_graph_path" in fieldnames
                if has_file_path:
                    csv_schema_counts["file_path"] += 1
                if has_unoptimized:
                    csv_schema_counts["unoptimized_graph_path"] += 1

                for row in reader:
                    csv_rows_total += 1

                    if has_file_path:
                        artifact_path = (row.get("file_path") or "").strip()
                        if artifact_path:
                            tier, key, state = parse_csv_file_path(artifact_path)
                            if key is None:
                                csv_artifact_unrecognized += 1
                            else:
                                csv_artifact_keys[tier].add(key)
                                csv_artifact_states[tier][state] += 1

                    if has_unoptimized:
                        input_path = (row.get("unoptimized_graph_path") or "").strip()
                        if input_path:
                            tier, key, state = parse_graph_pt_path(input_path)
                            if key is None:
                                csv_input_unrecognized += 1
                            else:
                                csv_input_keys[tier].add(key)
                                csv_input_states[tier][state] += 1
        except OSError as exc:
            print(f"[warning] Failed to read CSV {csv_path}: {exc}")

    print(f"CSV rows scanned: {csv_rows_total:,}")
    print(
        "CSV schema files: "
        f"file_path={csv_schema_counts['file_path']:,}, "
        f"unoptimized_graph_path={csv_schema_counts['unoptimized_graph_path']:,}"
    )
    print(f"CSV artifact unrecognized rows: {csv_artifact_unrecognized:,}")
    print(f"CSV input unrecognized rows: {csv_input_unrecognized:,}")

    # ------------------------------------------------------------------
    # 2) AIG zip scan
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("2. SCANNING AIG ZIPS")
    print("==========================================")

    aig_keys = init_tier_sets()
    aig_states: dict[int, Counter[str]] = defaultdict(Counter)
    zip_unrecognized_total = 0
    zip_aig_files_total = 0
    zip_errors: list[str] = []

    if zip_paths:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            iterator = executor.map(scan_single_zip, (str(path) for path in zip_paths))
            for result in tqdm(
                iterator, total=len(zip_paths), desc="Scanning ZIPs", unit="zip"
            ):
                error = str(result.get("error") or "")
                if error:
                    zip_errors.append(f"{result.get('zip_path')}: {error}")
                    continue

                tier_data = result["tier_keys"]
                state_data = result["state_counts"]
                zip_unrecognized_total += int(result["unrecognized"])
                zip_aig_files_total += int(result["aig_files"])

                for tier in (0, 1, 2):
                    aig_keys[tier].update(tier_data[tier])
                    aig_states[tier].update(state_data[tier])

    print(f"ZIP AIG members scanned: {zip_aig_files_total:,}")
    print(f"ZIP unrecognized AIG names: {zip_unrecognized_total:,}")
    if zip_errors:
        print(f"ZIP read errors: {len(zip_errors):,}")
        for err in zip_errors[:10]:
            print(f"  [zip-error] {err}")

    # ------------------------------------------------------------------
    # 3) PT artifact scan
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("3. SCANNING PT FILES")
    print("==========================================")

    pt_keys = init_tier_sets()
    pt_states: dict[int, Counter[str]] = defaultdict(Counter)
    pt_unrecognized_total = 0
    pt_files_total = 0

    pt_glob_root = args.pt_root
    if (args.pt_root / "graphs").is_dir():
        pt_glob_root = args.pt_root / "graphs"

    for pt_path in tqdm(pt_glob_root.rglob("*.pt"), desc="Scanning PTs", unit="pt"):
        pt_files_total += 1
        tier, key, state = parse_graph_pt_path(str(pt_path))
        if key is None:
            pt_unrecognized_total += 1
            continue
        pt_keys[tier].add(key)
        pt_states[tier][state] += 1

    print(f"PT files scanned: {pt_files_total:,}")
    print(f"PT unrecognized names: {pt_unrecognized_total:,}")

    # ------------------------------------------------------------------
    # 4) Reconciliation output
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("4. ARTIFACT RECONCILIATION")
    print("==========================================")

    for tier in (0, 1, 2):
        print_artifact_tier_summary(
            tier,
            aig_keys,
            pt_keys,
            csv_artifact_keys,
            max_examples=max(1, args.examples),
        )

    print("\nNaming states:")
    for tier in (0, 1, 2):
        print(
            f"  Tier {tier} CSV artifact: clean={csv_artifact_states[tier]['clean']:,} "
            f"messy={csv_artifact_states[tier]['messy']:,}"
        )
        print(
            f"  Tier {tier} PT files    : clean={pt_states[tier]['clean']:,} "
            f"messy={pt_states[tier]['messy']:,}"
        )

    # ------------------------------------------------------------------
    # 5) Tier2 -> Tier1 input linkage (training reality)
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("5. TIER2 -> TIER1 INPUT LINKAGE")
    print("==========================================")

    expected_t1_from_t2_aig = {
        (design, src_algorithm, recipe, step)
        for (design, src_algorithm, _dst_algorithm, recipe, step) in aig_keys[2]
    }
    expected_t1_from_t2_csv = {
        (design, src_algorithm, recipe, step)
        for (design, src_algorithm, _dst_algorithm, recipe, step) in csv_artifact_keys[
            2
        ]
    }

    csv_input_t1 = csv_input_keys[1]
    pt_t1 = pt_keys[1]

    print(f"Expected tier1 inputs from tier2 AIGs : {len(expected_t1_from_t2_aig):,}")
    print(f"Expected tier1 inputs from tier2 CSVs : {len(expected_t1_from_t2_csv):,}")
    print(f"Observed tier1 input IDs in CSV input : {len(csv_input_t1):,}")
    print(f"Observed tier1 PT artifacts on disk   : {len(pt_t1):,}")

    for label, expected in (
        ("from_t2_aig", expected_t1_from_t2_aig),
        ("from_t2_csv", expected_t1_from_t2_csv),
    ):
        missing_pt = expected - pt_t1
        missing_csv_input = expected - csv_input_t1
        covered_pt = expected & pt_t1
        covered_csv_input = expected & csv_input_t1

        print(f"\n  Linkage set: {label}")
        print(f"    expected              : {len(expected):,}")
        print(f"    covered by tier1 PT   : {len(covered_pt):,}")
        print(f"    missing tier1 PT      : {len(missing_pt):,}")
        print(f"    covered by CSV input  : {len(covered_csv_input):,}")
        print(f"    missing in CSV input  : {len(missing_csv_input):,}")

        if missing_pt:
            print(
                "    missing tier1 PT examples: "
                + summarize_examples(missing_pt, 1, max(1, args.examples))
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
