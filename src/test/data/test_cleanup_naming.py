"""Unit tests for src/data/cleanup_naming.py."""

from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from data.cleanup_naming import (
    _rewrite_single_zip,
    apply_csv_rewrites,
    apply_pt_renames,
    apply_zip_rewrites,
    load_csv_candidates,
    load_pt_candidates,
    load_zip_candidates,
    rewrite_csv_file,
    verify_csv_rewrites,
    verify_pt_renames,
    verify_zip_rewrites,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISSUES_DIR = Path(__file__).parents[3] / "logs" / "inspect_issues_22600850"
_EXPECTED_PT_PAIRS = 476_391
_EXPECTED_CSV_ZF = 9  # csv files in the mapping
_EXPECTED_ZIP_ZFS = 220  # zip files covered
_EXPECTED_ZIP_MEMBERS = 924_220

_MESSY_PT = "sqrt_Deepsyn_tier1_Deepsyn_sqrt_M7pRPw_syn81_step9.pt"
_CLEAN_PT = "sqrt_Deepsyn_tier1_syn81_step9.pt"

_MESSY_AIG = "1024_C2RS_tier1_C2RS_1024_1yHxJ4_synX_step0.aig"
_CLEAN_AIG = "1024_C2RS_tier1_synX_step0.aig"

_MESSY_CSV_PATH = (
    "/remote/designs/sqrt_Deepsyn_tier1_Deepsyn_sqrt_M7pRPw_syn81_step9.pt"
)
_CLEAN_CSV_PATH = "/remote/designs/sqrt_Deepsyn_tier1_syn81_step9.pt"


def _write_pt_candidates_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "pt_path",
        "tier",
        "naming_state",
        "canonical_id",
        "original_basename",
        "suggested_basename",
        "suggested_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_csv_candidates_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "csv_file",
        "row_number",
        "column",
        "tier",
        "naming_state",
        "canonical_id",
        "original_path",
        "suggested_path",
        "original_basename",
        "suggested_basename",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_zip_candidates_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "zip_path",
        "zip_source",
        "member_path",
        "member_name",
        "tier",
        "naming_state",
        "canonical_id",
        "suggested_member_path",
        "suggested_member_name",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _make_small_target_csv(path: Path, messy_path: str) -> None:
    """Write a tiny artifact CSV that has a messy file_path value."""
    fieldnames = ["file_path", "unoptimized_graph_path", "label"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(
            {"file_path": messy_path, "unoptimized_graph_path": "", "label": "1"}
        )
        w.writerow(
            {"file_path": "/clean/path.aig", "unoptimized_graph_path": "", "label": "0"}
        )


# ---------------------------------------------------------------------------
# PT rename tests
# ---------------------------------------------------------------------------


class TestPtRenames(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_pt_file(self, name: str) -> Path:
        p = self.tmp / name
        p.write_bytes(b"\x00" * 8)
        return p

    def test_dry_run_counts_only(self):
        src = self._make_pt_file(_MESSY_PT)
        dst = self.tmp / _CLEAN_PT
        pairs = [(src, dst)]
        counts = apply_pt_renames(pairs, dry_run=True)
        self.assertEqual(counts["renamed"], 1)
        self.assertTrue(src.exists(), "dry-run must not move src")
        self.assertFalse(dst.exists(), "dry-run must not create dst")

    def test_apply_renames_file(self):
        src = self._make_pt_file(_MESSY_PT)
        dst = self.tmp / _CLEAN_PT
        pairs = [(src, dst)]
        counts = apply_pt_renames(pairs, dry_run=False)
        self.assertEqual(counts["renamed"], 1)
        self.assertFalse(src.exists())
        self.assertTrue(dst.exists())

    def test_already_renamed_skipped(self):
        dst = self._make_pt_file(_CLEAN_PT)
        src = self.tmp / _MESSY_PT  # src does NOT exist; dst does
        pairs = [(src, dst)]
        counts = apply_pt_renames(pairs, dry_run=False)
        self.assertEqual(counts["already_renamed"], 1)
        self.assertEqual(counts["renamed"], 0)

    def test_missing_src_counted(self):
        src = self.tmp / "nonexistent.pt"
        dst = self.tmp / _CLEAN_PT
        pairs = [(src, dst)]
        counts = apply_pt_renames(pairs, dry_run=False)
        self.assertEqual(counts["skipped_missing_src"], 1)

    def test_verify_passes_after_rename(self):
        src = self._make_pt_file(_MESSY_PT)
        dst = self.tmp / _CLEAN_PT
        pairs = [(src, dst)]
        apply_pt_renames(pairs, dry_run=False)
        vcounts = verify_pt_renames(pairs)
        self.assertEqual(vcounts["ok"], 1)
        self.assertEqual(vcounts["src_still_exists"], 0)
        self.assertEqual(vcounts["dst_missing"], 0)

    def test_verify_detects_src_still_present(self):
        src = self._make_pt_file(_MESSY_PT)
        dst = self.tmp / _CLEAN_PT
        pairs = [(src, dst)]
        # Don't rename — verify should flag src_still_exists
        vcounts = verify_pt_renames(pairs)
        self.assertEqual(vcounts["src_still_exists"], 1)

    def test_load_pt_candidates(self):
        csv_path = self.tmp / "pt_naming_candidates.csv"
        _write_pt_candidates_csv(
            csv_path,
            [
                {
                    "pt_path": str(self.tmp / _MESSY_PT),
                    "tier": "1",
                    "naming_state": "messy",
                    "canonical_id": "sqrt_Deepsyn_81_9",
                    "original_basename": _MESSY_PT,
                    "suggested_basename": _CLEAN_PT,
                    "suggested_path": str(self.tmp / _CLEAN_PT),
                }
            ],
        )
        pairs = load_pt_candidates(csv_path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], self.tmp / _MESSY_PT)
        self.assertEqual(pairs[0][1], self.tmp / _CLEAN_PT)

    def test_load_pt_candidates_skips_same_src_dst(self):
        """Rows where src == dst (already clean) must not appear in output."""
        csv_path = self.tmp / "pt_naming_candidates.csv"
        clean_path = str(self.tmp / _CLEAN_PT)
        _write_pt_candidates_csv(
            csv_path,
            [
                {
                    "pt_path": clean_path,
                    "tier": "1",
                    "naming_state": "clean",
                    "canonical_id": "sqrt_Deepsyn_81_9",
                    "original_basename": _CLEAN_PT,
                    "suggested_basename": _CLEAN_PT,
                    "suggested_path": clean_path,
                }
            ],
        )
        pairs = load_pt_candidates(csv_path)
        self.assertEqual(len(pairs), 0)


# ---------------------------------------------------------------------------
# CSV rewrite tests
# ---------------------------------------------------------------------------


class TestCsvRewrites(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_dry_run_does_not_modify_csv(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        original_bytes = target_csv.read_bytes()
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        counts = rewrite_csv_file(target_csv, path_map, dry_run=True)
        self.assertEqual(counts["rows_changed"], 1)
        self.assertEqual(counts["rows_unchanged"], 1)
        self.assertEqual(target_csv.read_bytes(), original_bytes)

    def test_apply_rewrites_csv(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        counts = rewrite_csv_file(target_csv, path_map, dry_run=False)
        self.assertEqual(counts["rows_changed"], 1)
        # Read back and verify clean path present, messy absent
        with open(target_csv, newline="", encoding="utf-8") as fh:
            paths = [row["file_path"] for row in csv.DictReader(fh)]
        self.assertIn(_CLEAN_CSV_PATH, paths)
        self.assertNotIn(_MESSY_CSV_PATH, paths)

    def test_no_tmp_file_left_after_apply(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        rewrite_csv_file(target_csv, path_map, dry_run=False)
        tmp_path = target_csv.with_suffix(".csv.rewriting")
        self.assertFalse(
            tmp_path.exists(), ".rewriting tmp must be removed after atomic replace"
        )

    def test_no_tmp_file_left_after_dry_run(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        rewrite_csv_file(target_csv, path_map, dry_run=True)
        tmp_path = target_csv.with_suffix(".csv.rewriting")
        self.assertFalse(
            tmp_path.exists(), ".rewriting tmp must be removed after dry-run"
        )

    def test_verify_passes_after_rewrite(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        rewrite_csv_file(target_csv, path_map, dry_run=False)
        mapping = {str(target_csv): path_map}
        vcounts = verify_csv_rewrites(mapping)
        self.assertEqual(vcounts["still_messy"], 0)
        self.assertEqual(vcounts["ok"], 1)

    def test_verify_detects_unrewritten_messy(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        path_map = {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}
        mapping = {str(target_csv): path_map}
        # Don't rewrite — verify must report still_messy
        vcounts = verify_csv_rewrites(mapping)
        self.assertEqual(vcounts["still_messy"], 1)

    def test_load_csv_candidates(self):
        cands_csv = self.tmp / "csv_naming_candidates.csv"
        target_csv = self.tmp / "artifact.csv"
        _write_csv_candidates_csv(
            cands_csv,
            [
                {
                    "csv_file": str(target_csv),
                    "row_number": "5",
                    "column": "file_path",
                    "tier": "0",
                    "naming_state": "messy",
                    "canonical_id": "sqrt_81_5",
                    "original_path": _MESSY_CSV_PATH,
                    "suggested_path": _CLEAN_CSV_PATH,
                    "original_basename": _MESSY_PT,
                    "suggested_basename": _CLEAN_PT,
                }
            ],
        )
        mapping = load_csv_candidates(cands_csv)
        self.assertIn(str(target_csv), mapping)
        self.assertEqual(mapping[str(target_csv)][_MESSY_CSV_PATH], _CLEAN_CSV_PATH)

    def test_apply_csv_rewrites_aggregates(self):
        target_csv = self.tmp / "artifact.csv"
        _make_small_target_csv(target_csv, _MESSY_CSV_PATH)
        mapping = {str(target_csv): {_MESSY_CSV_PATH: _CLEAN_CSV_PATH}}
        totals = apply_csv_rewrites(mapping, dry_run=False)
        self.assertEqual(totals["rows_changed"], 1)


# ---------------------------------------------------------------------------
# ZIP rewrite tests
# ---------------------------------------------------------------------------


class TestZipRewrites(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_messy_zip(self, zip_path: Path) -> list[str]:
        """Create a ZIP with two messy AIG member names; return member list."""
        members = [
            _MESSY_AIG,
            "1024_C2RS_tier1_C2RS_1024_Xq1Y2z_synX_step1.aig",
        ]
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for m in members:
                zf.writestr(m, b"\xab" * 4)
        return members

    def test_dry_run_does_not_modify_zip(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        original_bytes = zip_path.read_bytes()
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        _rewrite_single_zip((str(zip_path), member_map, True, ""))
        self.assertEqual(zip_path.read_bytes(), original_bytes)

    def test_apply_renames_zip_members(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        clean0 = _CLEAN_AIG
        clean1 = "1024_C2RS_tier1_synX_step1.aig"
        member_map = {members[0]: clean0, members[1]: clean1}
        result = _rewrite_single_zip((str(zip_path), member_map, False, ""))
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["members_renamed"], 2)
        with zipfile.ZipFile(zip_path) as zf:
            names = {i.filename for i in zf.infolist()}
        self.assertIn(clean0, names)
        self.assertIn(clean1, names)
        self.assertNotIn(members[0], names)

    def test_member_count_preserved(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        _rewrite_single_zip((str(zip_path), member_map, False, ""))
        with zipfile.ZipFile(zip_path) as zf:
            self.assertEqual(len(zf.infolist()), 2)

    def test_no_tmp_zip_left_after_apply(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        _rewrite_single_zip((str(zip_path), member_map, False, ""))
        self.assertFalse(zip_path.with_suffix(".zip.rewriting").exists())

    def test_no_tmp_zip_left_after_dry_run(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        _rewrite_single_zip((str(zip_path), member_map, True, ""))
        self.assertFalse(zip_path.with_suffix(".zip.rewriting").exists())

    def test_verify_passes_after_rewrite(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        _rewrite_single_zip((str(zip_path), member_map, False, ""))
        vcounts = verify_zip_rewrites({str(zip_path): member_map})
        self.assertEqual(vcounts["still_messy"], 0)
        self.assertEqual(vcounts["ok"], 1)

    def test_verify_detects_unrewritten_members(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {members[0]: _CLEAN_AIG}
        # Don't rewrite
        vcounts = verify_zip_rewrites({str(zip_path): member_map})
        self.assertEqual(vcounts["still_messy"], 1)

    def test_load_zip_candidates(self):
        cands_csv = self.tmp / "aig_zip_naming_candidates.csv"
        zip_path = self.tmp / "tier1.zip"
        _write_zip_candidates_csv(
            cands_csv,
            [
                {
                    "zip_path": str(zip_path),
                    "zip_source": "tier1",
                    "member_path": _MESSY_AIG,
                    "member_name": _MESSY_AIG,
                    "tier": "1",
                    "naming_state": "messy",
                    "canonical_id": "1024_C2RS_X_0",
                    "suggested_member_path": _CLEAN_AIG,
                    "suggested_member_name": _CLEAN_AIG,
                }
            ],
        )
        mapping = load_zip_candidates(cands_csv)
        self.assertIn(str(zip_path), mapping)
        self.assertEqual(mapping[str(zip_path)][_MESSY_AIG], _CLEAN_AIG)

    def test_apply_zip_rewrites_aggregates(self):
        zip_path = self.tmp / "tier1.zip"
        members = self._make_messy_zip(zip_path)
        member_map = {
            members[0]: _CLEAN_AIG,
            members[1]: "1024_C2RS_tier1_synX_step1.aig",
        }
        totals = apply_zip_rewrites(
            {str(zip_path): member_map}, dry_run=False, workers=1
        )
        self.assertEqual(totals["errors"], 0)
        self.assertEqual(totals["members_renamed"], 2)
        self.assertEqual(totals["zips_processed"], 1)


# ---------------------------------------------------------------------------
# Integration: actual audit data counts (skipped if issues dir absent)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _ISSUES_DIR.is_dir(), f"issues dir not found locally: {_ISSUES_DIR}"
)
class TestActualAuditCounts(unittest.TestCase):
    def test_pt_candidates_count(self):
        csv_path = _ISSUES_DIR / "pt_naming_candidates.csv"
        pairs = load_pt_candidates(csv_path)
        self.assertEqual(
            len(pairs),
            _EXPECTED_PT_PAIRS,
            f"Expected {_EXPECTED_PT_PAIRS:,} PT pairs, got {len(pairs):,}",
        )

    def test_csv_candidates_file_count(self):
        csv_path = _ISSUES_DIR / "csv_naming_candidates.csv"
        mapping = load_csv_candidates(csv_path)
        self.assertEqual(
            len(mapping),
            _EXPECTED_CSV_ZF,
            f"Expected {_EXPECTED_CSV_ZF} CSV files in mapping, got {len(mapping)}",
        )

    def test_zip_candidates_zip_count(self):
        csv_path = _ISSUES_DIR / "aig_zip_naming_candidates.csv"
        mapping = load_zip_candidates(csv_path)
        self.assertEqual(
            len(mapping),
            _EXPECTED_ZIP_ZFS,
            f"Expected {_EXPECTED_ZIP_ZFS} ZIP files in mapping, got {len(mapping)}",
        )

    def test_zip_candidates_member_count(self):
        csv_path = _ISSUES_DIR / "aig_zip_naming_candidates.csv"
        mapping = load_zip_candidates(csv_path)
        total = sum(len(v) for v in mapping.values())
        self.assertEqual(
            total,
            _EXPECTED_ZIP_MEMBERS,
            f"Expected {_EXPECTED_ZIP_MEMBERS:,} ZIP members, got {total:,}",
        )


if __name__ == "__main__":
    unittest.main()
