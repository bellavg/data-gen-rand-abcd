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
) -> tuple[
    list[Path],
    Counter[str],
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
]:
    zip_paths: set[Path] = set()
    source_zip_counts: Counter[str] = Counter()
    zip_source_map: dict[str, str] = {}
    source_labels: dict[str, str] = {}
    source_samples: dict[str, list[str]] = defaultdict(list)

    def add_paths(paths: Iterable[Path], source_key: str, source_label: str) -> None:
        source_labels[source_key] = source_label
        for p in paths:
            if not p.is_file():
                continue
            p_str = str(p)
            if p_str in zip_source_map:
                continue
            zip_paths.add(p)
            zip_source_map[p_str] = source_key
            source_zip_counts[source_key] += 1
            if len(source_samples[source_key]) < 3:
                source_samples[source_key].append(p_str)

    if aig_root.is_dir():
        add_paths(aig_root.glob("*/tier0.zip"), "aig_root:tier0", str(aig_root))
        add_paths(
            aig_root.glob("*/tier0/tier0.zip"),
            "aig_root:tier0_nested",
            str(aig_root),
        )
        add_paths(aig_root.glob("*/tier1/*.zip"), "aig_root:tier1", str(aig_root))
        add_paths(aig_root.glob("*/tier2/*.zip"), "aig_root:tier2", str(aig_root))

    for idx, root in enumerate(tier2_roots):
        if root.is_dir():
            source_key = f"tier2_root:{idx}"
            add_paths(root.rglob("*.zip"), source_key, str(root))

    return (
        sorted(zip_paths),
        source_zip_counts,
        zip_source_map,
        source_labels,
        source_samples,
    )


def scan_single_zip(zip_path: str) -> dict[str, object]:
    tier_keys = init_tier_sets()
    state_counts: dict[int, Counter[str]] = defaultdict(Counter)
    tier_member_counts: Counter[int] = Counter()
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
                tier_member_counts[tier] += 1
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    return {
        "zip_path": zip_path,
        "tier_keys": tier_keys,
        "state_counts": state_counts,
        "tier_member_counts": dict(tier_member_counts),
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


def print_counter_top(
    title: str,
    counter: Counter[str],
    top_k: int,
    *,
    indent: str = "    ",
) -> None:
    print(title)
    if not counter:
        print(f"{indent}none")
        return
    for key, count in counter.most_common(top_k):
        print(f"{indent}{key}: {count:,}")


def print_tier0_missing_details(
    missing: set[tuple[str, ...]],
    expected: set[tuple[str, ...]],
    found: set[tuple[str, ...]],
    *,
    top_k: int,
    bucket_examples: int,
) -> None:
    print("\nTier0 missing PT diagnostics:")
    if not missing:
        print("  no missing tier0 PT IDs")
        return

    missing_by_design = Counter(key[0] for key in missing)
    missing_by_recipe = Counter(key[1] for key in missing)
    missing_by_step = Counter(key[2] for key in missing)

    expected_by_design = Counter(key[0] for key in expected)
    found_by_design = Counter(key[0] for key in found)

    print_counter_top("  Top missing designs:", missing_by_design, top_k)
    print_counter_top("  Top missing recipes:", missing_by_recipe, top_k)
    print_counter_top("  Top missing steps:", missing_by_step, top_k)

    print("  Lowest tier0 PT coverage by design:")
    coverage_rows: list[tuple[float, int, str, int, int, int]] = []
    for design, exp in expected_by_design.items():
        got = found_by_design.get(design, 0)
        miss = max(0, exp - got)
        pct = (100.0 * got / exp) if exp else 0.0
        coverage_rows.append((pct, miss, design, got, exp, miss))

    coverage_rows.sort(key=lambda row: (row[0], -row[1], row[2]))
    for pct, _miss_sort, design, got, exp, miss in coverage_rows[:top_k]:
        print(f"    {design}: {got:,}/{exp:,} ({pct:.2f}%) missing={miss:,}")

    examples_by_design: dict[str, list[str]] = defaultdict(list)
    for key in sorted(missing):
        design = key[0]
        if len(examples_by_design[design]) < bucket_examples:
            examples_by_design[design].append(format_key(0, key))

    print("  Missing tier0 PT examples by top design:")
    for design, _count in missing_by_design.most_common(top_k):
        joined = ", ".join(examples_by_design[design])
        print(f"    {design}: {joined}")


def print_tier1_missing_details(
    missing: set[tuple[str, ...]],
    expected: set[tuple[str, ...]],
    found: set[tuple[str, ...]],
    *,
    top_k: int,
    bucket_examples: int,
) -> None:
    print("\nTier1 missing PT diagnostics:")
    if not missing:
        print("  no missing tier1 PT IDs")
        return

    missing_by_algo = Counter(key[1] for key in missing)
    missing_by_design = Counter(key[0] for key in missing)
    missing_by_design_algo = Counter(f"{key[0]}|{key[1]}" for key in missing)
    missing_by_recipe = Counter(key[2] for key in missing)
    missing_by_step = Counter(key[3] for key in missing)

    expected_by_algo = Counter(key[1] for key in expected)
    found_by_algo = Counter(key[1] for key in found)

    print_counter_top("  Missing by algorithm:", missing_by_algo, top_k)
    print_counter_top("  Top missing designs:", missing_by_design, top_k)
    print_counter_top("  Top missing design|algorithm:", missing_by_design_algo, top_k)
    print_counter_top("  Top missing recipes:", missing_by_recipe, top_k)
    print_counter_top("  Top missing steps:", missing_by_step, top_k)

    print("  Tier1 PT coverage by algorithm:")
    for algo in ALGORITHMS:
        exp = expected_by_algo.get(algo, 0)
        got = found_by_algo.get(algo, 0)
        miss = max(0, exp - got)
        pct = (100.0 * got / exp) if exp else 0.0
        print(f"    {algo}: {got:,}/{exp:,} ({pct:.2f}%) missing={miss:,}")

    examples_by_algo: dict[str, list[str]] = defaultdict(list)
    for key in sorted(missing):
        algo = key[1]
        if len(examples_by_algo[algo]) < bucket_examples:
            examples_by_algo[algo].append(format_key(1, key))

    print("  Missing tier1 PT examples by algorithm:")
    for algo in ALGORITHMS:
        examples = examples_by_algo.get(algo, [])
        if not examples:
            continue
        print(f"    {algo}: {', '.join(examples)}")


def print_tier2_csv_without_aig_details(
    csv_without_aig: set[tuple[str, ...]],
    *,
    top_k: int,
    bucket_examples: int,
) -> None:
    print("\nTier2 CSV-without-AIG diagnostics:")
    if not csv_without_aig:
        print("  no tier2 CSV ghost IDs")
        return

    by_dst_algo = Counter(key[2] for key in csv_without_aig)
    by_src_algo = Counter(key[1] for key in csv_without_aig)
    by_design = Counter(key[0] for key in csv_without_aig)
    by_recipe = Counter(key[3] for key in csv_without_aig)
    by_step = Counter(key[4] for key in csv_without_aig)

    print_counter_top("  CSV ghosts by tier2 target algorithm:", by_dst_algo, top_k)
    print_counter_top("  CSV ghosts by tier1 source algorithm:", by_src_algo, top_k)
    print_counter_top("  CSV ghosts by design:", by_design, top_k)
    print_counter_top("  CSV ghosts by recipe:", by_recipe, top_k)
    print_counter_top("  CSV ghosts by step:", by_step, top_k)

    examples_by_algo: dict[str, list[str]] = defaultdict(list)
    for key in sorted(csv_without_aig):
        algo = key[2]
        if len(examples_by_algo[algo]) < bucket_examples:
            examples_by_algo[algo].append(format_key(2, key))

    print("  CSV ghost examples by tier2 target algorithm:")
    for algo in ALGORITHMS:
        examples = examples_by_algo.get(algo, [])
        if not examples:
            continue
        print(f"    {algo}: {', '.join(examples)}")


def print_tier2_linkage_missing_details(
    missing_t1_inputs: set[tuple[str, ...]],
    *,
    top_k: int,
    bucket_examples: int,
) -> None:
    print("\nTier2 expected-input missing tier1 PT diagnostics:")
    if not missing_t1_inputs:
        print("  no missing tier1 PT IDs for tier2 input linkage")
        return

    by_algo = Counter(key[1] for key in missing_t1_inputs)
    by_design = Counter(key[0] for key in missing_t1_inputs)
    by_design_algo = Counter(f"{key[0]}|{key[1]}" for key in missing_t1_inputs)
    by_recipe = Counter(key[2] for key in missing_t1_inputs)
    by_step = Counter(key[3] for key in missing_t1_inputs)

    print_counter_top("  Missing by source algorithm:", by_algo, top_k)
    print_counter_top("  Top missing designs:", by_design, top_k)
    print_counter_top("  Top missing design|source_algo:", by_design_algo, top_k)
    print_counter_top("  Top missing recipes:", by_recipe, top_k)
    print_counter_top("  Top missing steps:", by_step, top_k)

    examples_by_algo: dict[str, list[str]] = defaultdict(list)
    for key in sorted(missing_t1_inputs):
        algo = key[1]
        if len(examples_by_algo[algo]) < bucket_examples:
            examples_by_algo[algo].append(format_key(1, key))

    print("  Missing linkage examples by source algorithm:")
    for algo in ALGORITHMS:
        examples = examples_by_algo.get(algo, [])
        if not examples:
            continue
        print(f"    {algo}: {', '.join(examples)}")


def write_issue_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def export_issue_sets(
    out_dir: Path,
    *,
    tier0_missing_pt: set[tuple[str, ...]],
    tier1_missing_pt: set[tuple[str, ...]],
    tier2_csv_without_aig: set[tuple[str, ...]],
    tier2_expected_missing_pt: set[tuple[str, ...]],
) -> None:
    print("\n==========================================")
    print("7. ISSUE CSV EXPORT")
    print("==========================================")
    print(f"Issue export directory: {out_dir}")

    t0_rows = (
        {
            "tier": "0",
            "canonical_id": format_key(0, key),
            "design": key[0],
            "recipe": key[1],
            "step": key[2],
        }
        for key in tier0_missing_pt
    )
    t0_written = write_issue_csv(
        out_dir / "tier0_missing_pt.csv",
        ["tier", "canonical_id", "design", "recipe", "step"],
        t0_rows,
    )

    t1_rows = (
        {
            "tier": "1",
            "canonical_id": format_key(1, key),
            "design": key[0],
            "algorithm": key[1],
            "recipe": key[2],
            "step": key[3],
        }
        for key in tier1_missing_pt
    )
    t1_written = write_issue_csv(
        out_dir / "tier1_missing_pt.csv",
        ["tier", "canonical_id", "design", "algorithm", "recipe", "step"],
        t1_rows,
    )

    t2_csv_rows = (
        {
            "tier": "2",
            "canonical_id": format_key(2, key),
            "design": key[0],
            "src_algorithm": key[1],
            "dst_algorithm": key[2],
            "recipe": key[3],
            "step": key[4],
        }
        for key in tier2_csv_without_aig
    )
    t2_csv_written = write_issue_csv(
        out_dir / "tier2_csv_without_aig.csv",
        [
            "tier",
            "canonical_id",
            "design",
            "src_algorithm",
            "dst_algorithm",
            "recipe",
            "step",
        ],
        t2_csv_rows,
    )

    linkage_rows = (
        {
            "tier": "1",
            "canonical_id": format_key(1, key),
            "design": key[0],
            "source_algorithm": key[1],
            "recipe": key[2],
            "step": key[3],
        }
        for key in tier2_expected_missing_pt
    )
    linkage_written = write_issue_csv(
        out_dir / "tier2_expected_tier1_input_missing_pt.csv",
        ["tier", "canonical_id", "design", "source_algorithm", "recipe", "step"],
        linkage_rows,
    )

    print(f"  tier0_missing_pt.csv rows: {t0_written:,}")
    print(f"  tier1_missing_pt.csv rows: {t1_written:,}")
    print(f"  tier2_csv_without_aig.csv rows: {t2_csv_written:,}")
    print(f"  tier2_expected_tier1_input_missing_pt.csv rows: {linkage_written:,}")


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
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-N bucket size for issue breakdown tables.",
    )
    parser.add_argument(
        "--bucket-examples",
        type=int,
        default=3,
        help="Number of example IDs shown per bucket in detailed issue sections.",
    )
    parser.add_argument(
        "--issues-out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for writing issue CSVs (missing PTs, tier2 CSV ghosts, "
            "tier2 expected-input missing PTs)."
        ),
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
    (
        zip_paths,
        zip_source_counts,
        zip_source_map,
        zip_source_labels,
        zip_source_samples,
    ) = discover_zip_files(args.aig_root, tier2_roots)

    print("==========================================")
    print("CONFIG")
    print("==========================================")
    print(f"AIG root: {args.aig_root}")
    print(f"Tier2 AIG roots: {[str(p) for p in tier2_roots] if tier2_roots else '[]'}")
    print(f"PT root: {args.pt_root}")
    print(f"CSV files discovered: {len(csv_paths):,}")
    print(f"ZIP files discovered: {len(zip_paths):,}")
    print(f"Workers: {args.workers}")
    print(f"Top-k diagnostics: {args.top_k}")
    print(f"Bucket examples: {args.bucket_examples}")
    if args.issues_out_dir is None:
        print("Issue CSV export: disabled")
    else:
        print(f"Issue CSV export: {args.issues_out_dir}")

    print("ZIP discovery by source:")
    if not zip_source_counts:
        print("  none")
    else:
        for source_key in sorted(zip_source_counts):
            label = zip_source_labels.get(source_key, source_key)
            count = zip_source_counts[source_key]
            print(f"  {source_key} ({label}): {count:,}")
            samples = zip_source_samples.get(source_key, [])
            if samples:
                print(f"    sample: {samples[0]}")

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
    source_member_counts: Counter[str] = Counter()
    source_unrecognized_counts: Counter[str] = Counter()
    source_error_counts: Counter[str] = Counter()
    source_tier_member_counts: dict[str, Counter[int]] = defaultdict(Counter)

    if zip_paths:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            iterator = executor.map(scan_single_zip, (str(path) for path in zip_paths))
            for result in tqdm(
                iterator, total=len(zip_paths), desc="Scanning ZIPs", unit="zip"
            ):
                error = str(result.get("error") or "")
                zip_path = str(result.get("zip_path") or "")
                source_key = zip_source_map.get(zip_path, "unknown")
                if error:
                    source_error_counts[source_key] += 1
                    zip_errors.append(f"{zip_path}: {error}")
                    continue

                tier_data = result["tier_keys"]
                state_data = result["state_counts"]
                tier_member_data = result["tier_member_counts"]
                zip_unrecognized_total += int(result["unrecognized"])
                zip_aig_files_total += int(result["aig_files"])
                source_member_counts[source_key] += int(result["aig_files"])
                source_unrecognized_counts[source_key] += int(result["unrecognized"])

                for tier_raw, count in tier_member_data.items():
                    source_tier_member_counts[source_key][int(tier_raw)] += int(count)

                for tier in (0, 1, 2):
                    aig_keys[tier].update(tier_data[tier])
                    aig_states[tier].update(state_data[tier])

    print(f"ZIP AIG members scanned: {zip_aig_files_total:,}")
    print(f"ZIP unrecognized AIG names: {zip_unrecognized_total:,}")
    if zip_errors:
        print(f"ZIP read errors: {len(zip_errors):,}")
        for err in zip_errors[:10]:
            print(f"  [zip-error] {err}")

    print("ZIP scan diagnostics by source:")
    source_keys = sorted(
        set(zip_source_counts)
        | set(source_member_counts)
        | set(source_unrecognized_counts)
        | set(source_error_counts)
        | set(source_tier_member_counts)
    )
    if not source_keys:
        print("  none")
    else:
        for source_key in source_keys:
            label = zip_source_labels.get(source_key, source_key)
            zips = zip_source_counts.get(source_key, 0)
            members = source_member_counts.get(source_key, 0)
            unrec = source_unrecognized_counts.get(source_key, 0)
            errs = source_error_counts.get(source_key, 0)
            t0 = source_tier_member_counts[source_key].get(0, 0)
            t1 = source_tier_member_counts[source_key].get(1, 0)
            t2 = source_tier_member_counts[source_key].get(2, 0)
            print(
                f"  {source_key} ({label}): zips={zips:,} members={members:,} "
                f"tier0={t0:,} tier1={t1:,} tier2={t2:,} "
                f"unrecognized={unrec:,} errors={errs:,}"
            )

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

    tier0_missing_pt = (aig_keys[0] & csv_artifact_keys[0]) - pt_keys[0]
    tier1_missing_pt = (aig_keys[1] & csv_artifact_keys[1]) - pt_keys[1]
    tier2_csv_without_aig = csv_artifact_keys[2] - aig_keys[2]

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

        missing_by_algo = Counter(key[1] for key in missing_pt)
        covered_by_algo = Counter(key[1] for key in covered_pt)
        if expected:
            print("    by source algorithm (covered/missing):")
            for algo in ALGORITHMS:
                c = covered_by_algo.get(algo, 0)
                m = missing_by_algo.get(algo, 0)
                total = c + m
                pct = (100.0 * c / total) if total else 0.0
                print(f"      {algo}: covered={c:,} missing={m:,} ({pct:.2f}% covered)")

        if missing_pt:
            print(
                "    missing tier1 PT examples: "
                + summarize_examples(missing_pt, 1, max(1, args.examples))
            )

    # ------------------------------------------------------------------
    # 6) Detailed issue diagnostics for cleanup planning
    # ------------------------------------------------------------------
    print("\n==========================================")
    print("6. DETAILED ISSUE DIAGNOSTICS")
    print("==========================================")
    print_tier0_missing_details(
        tier0_missing_pt,
        aig_keys[0] & csv_artifact_keys[0],
        pt_keys[0],
        top_k=max(1, args.top_k),
        bucket_examples=max(1, args.bucket_examples),
    )
    print_tier1_missing_details(
        tier1_missing_pt,
        aig_keys[1] & csv_artifact_keys[1],
        pt_keys[1],
        top_k=max(1, args.top_k),
        bucket_examples=max(1, args.bucket_examples),
    )
    print_tier2_csv_without_aig_details(
        tier2_csv_without_aig,
        top_k=max(1, args.top_k),
        bucket_examples=max(1, args.bucket_examples),
    )

    tier2_expected_missing_pt = expected_t1_from_t2_csv - pt_t1
    print_tier2_linkage_missing_details(
        tier2_expected_missing_pt,
        top_k=max(1, args.top_k),
        bucket_examples=max(1, args.bucket_examples),
    )

    if args.issues_out_dir is not None:
        export_issue_sets(
            args.issues_out_dir,
            tier0_missing_pt=tier0_missing_pt,
            tier1_missing_pt=tier1_missing_pt,
            tier2_csv_without_aig=tier2_csv_without_aig,
            tier2_expected_missing_pt=tier2_expected_missing_pt,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
