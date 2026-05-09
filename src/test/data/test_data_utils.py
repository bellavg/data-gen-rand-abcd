import math
import os
import tempfile
import unittest
import zipfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

# Existing imports
from data.data_utils import (
    _extract_topology,
    _safe_log1p_int,
    aig_to_pytorch_geometric,
    parse_aig_name,
)

# Imports for testing preprocess_data.py
from data.preprocess_data import (
    GraphTask,
    WorkerConfig,
    artifact_output_base_path,
    discover_graph_tasks,
    graph_output_path,
    process_task,
)


class TestAigToPytorchGeometric(unittest.TestCase):
    def test_with_adder_aig(self) -> None:
        aig_path = Path(__file__).with_name("adder.aig")
        self.assertTrue(aig_path.exists(), f"Missing fixture AIG file: {aig_path}")

        data = aig_to_pytorch_geometric(aig_path)

        self.assertGreater(data.num_nodes, 0)

        self.assertEqual(data.x.shape, (data.num_nodes, 4))
        self.assertEqual(data.level.shape, (data.num_nodes, 1))
        self.assertEqual(data.pi_paths.shape, (data.num_nodes, 1))
        self.assertEqual(data.local_sp_sum.shape, (data.num_nodes, 1))

        self.assertEqual(data.edge_index.ndim, 2)
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_attr.ndim, 2)
        self.assertEqual(data.edge_attr.shape[1], 2)
        self.assertEqual(data.edge_attr.shape[0], data.edge_index.shape[1])

        self.assertEqual(data.num_nodes, data.x.shape[0])
        self.assertEqual(data.num_edges, data.edge_index.shape[1])
        self.assertGreaterEqual(data.num_pis, 0)
        self.assertGreaterEqual(data.num_pos, 0)

        for row_sum in data.x.sum(dim=1).tolist():
            self.assertAlmostEqual(row_sum, 1.0, places=6)


class TestDataUtilsFunctions(unittest.TestCase):
    def test_safe_log1p_int(self):
        # Test normal behaviors
        self.assertEqual(_safe_log1p_int(0), 0.0)
        self.assertAlmostEqual(_safe_log1p_int(10), math.log1p(10), places=6)

        # Test exceptionally large integer that would normally trigger OverflowError
        # log1p(2**1000) roughly equals 1000 * log(2)
        huge_val = 2**1000
        approx_expected = 1000 * math.log(2.0)

        res = _safe_log1p_int(huge_val)
        self.assertAlmostEqual(res, approx_expected, delta=0.1)

        # Test negative value error
        with self.assertRaises(ValueError):
            _safe_log1p_int(-5)

    def test_parse_aig_name(self):
        # Tier 0 Tests
        self.assertEqual(parse_aig_name("adder_syn1_step0.aig"), (0, "", "adder"))
        # Tier 1 Tests
        self.assertEqual(
            parse_aig_name("adder_Orchestrate_tier1_syn1_step0.aig"),
            (1, "Orchestrate", "adder"),
        )
        # Messy names (with mktemp junk token) must no longer parse — cleanup_naming.py
        # must clean ZIPs first so that preprocess_data can't save messy .pt stems.
        self.assertIsNone(
            parse_aig_name("multiplier_Deepsyn_tier1_Deepsyn_v2_synX_step5.aig")
        )
        # Tier 2 outputs should be ignored (fallback to None)
        self.assertIsNone(
            parse_aig_name("adder_Orchestrate_Deepsyn_tier2_syn1_step0.aig")
        )

        # Random/unmatched files
        self.assertIsNone(parse_aig_name("random_notes.txt"))
        self.assertIsNone(parse_aig_name("adder_tier1_unmatched.aig"))


class TestDataUtilsGraphTopology(unittest.TestCase):
    def test_two_pos_same_driver(self):
        """
        Ensures that if 2 primary outputs are driven by the SAME node in the AIG,
        _extract_topology accurately generates 2 distinct synthetic PO nodes.
        """
        # Mock the Aig
        aig_mock = MagicMock()
        aig_mock.nodes.return_value = [0, 1]
        aig_mock.is_constant.side_effect = lambda n: n == 0
        aig_mock.is_pi.side_effect = lambda n: n == 1
        aig_mock.fanins.return_value = []

        # Mock POs: two separate PO objects, both driven by base node 1
        po1 = MagicMock()
        po1.get_index.return_value = 1
        po1.get_complement.return_value = False  # Regular edge

        po2 = MagicMock()
        po2.get_index.return_value = 1
        po2.get_complement.return_value = True  # Inverted edge

        aig_mock.pos.return_value = [po1, po2]
        aig_mock.num_pos.return_value = 2
        aig_mock.num_pis.return_value = 1

        # Mock DepthAig
        depth_mock = MagicMock()
        depth_mock.level.side_effect = lambda n: 0.0 if n == 0 else 1.0

        num_nodes, x, lvl, edges, succ = _extract_topology(aig_mock, depth_mock)

        # We expect 4 nodes total: Constant (0), PI (1), PO1 (2), PO2 (3)
        self.assertEqual(num_nodes, 4)

        # Check node features (1-hot encoded)
        self.assertEqual(x[2], [0.0, 0.0, 0.0, 1.0])  # PO1 is PO
        self.assertEqual(x[3], [0.0, 0.0, 0.0, 1.0])  # PO2 is PO

        # Check Levels (Driver level is 1, so PO should be 2)
        self.assertEqual(lvl[2], [2.0])
        self.assertEqual(lvl[3], [2.0])

        # Check Edges: Driver (index 1) to POs (index 2 and 3)
        # Expected edge format: (u, v, [regular, inverted])
        self.assertIn((1, 2, [1.0, 0.0]), edges)
        self.assertIn((1, 3, [0.0, 1.0]), edges)


class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_artifact_output_paths(self):
        base_out = Path("/tmp/out")

        # Test Tier 0 Output path
        t0_task = GraphTask(1, 0, "", "adder", "adder_syn1_step0.aig", "path", "", "")
        self.assertEqual(
            artifact_output_base_path(base_out, t0_task),
            base_out / "graphs" / "tier0" / "adder" / "adder_syn1_step0",
        )
        self.assertEqual(
            graph_output_path(base_out, t0_task),
            base_out / "graphs" / "tier0" / "adder" / "adder_syn1_step0.pt",
        )

        # Test Tier 1 Output path
        t1_task = GraphTask(
            2,
            1,
            "Orchestrate",
            "adder",
            "adder_Orchestrate_tier1_synX_step5.aig",
            "path",
            "",
            "",
        )
        self.assertEqual(
            artifact_output_base_path(base_out, t1_task),
            base_out
            / "graphs"
            / "tier1"
            / "Orchestrate"
            / "adder"
            / "adder_Orchestrate_tier1_synX_step5",
        )

    def test_discover_graph_tasks(self):
        """
        Create a mock filesystem with loose AIGs and a ZIP archive to test discovery logic.
        The layout needs to match the expected: <root>/<design>/tier0/... and <root>/<design>/tier1/...
        """
        # Create design directory first
        design_dir = self.root / "adder"
        design_dir.mkdir()

        tier0_dir = design_dir / "tier0"
        tier0_dir.mkdir()

        tier1_dir = design_dir / "tier1"
        tier1_dir.mkdir()

        # Add valid and invalid loose files
        (tier0_dir / "adder_syn1_step0.aig").touch()
        (tier0_dir / "invalid_name.aig").touch()

        # Add a tier1 zip file matching the expected glob pattern "*/tier1/*.zip"
        zip_path = tier1_dir / "adder_Orchestrate.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adder_Orchestrate_tier1_syn1_step1.aig", "dummy data")
            zf.writestr("ignored_member.txt", "not an aig")

        tasks, unmatched, source_counts = discover_graph_tasks(
            self.root, allow_unmatched_names=True
        )

        # We expect 2 valid matched tasks (1 loose, 1 in zip). The invalid file is counted as unmatched.
        self.assertEqual(len(tasks), 2)
        self.assertEqual(unmatched, 1)
        self.assertEqual(
            source_counts["filesystem_aig"], 2
        )  # It touched 2 files locally
        self.assertEqual(
            source_counts["zip_aig"], 1
        )  # Extracted 1 valid AIG from the zip

        # Test fail on unmatched
        with self.assertRaises(ValueError):
            discover_graph_tasks(self.root, allow_unmatched_names=False)

    def test_discover_graph_tasks_messy_tier1_unmatched(self):
        """Messy tier-1 AIG names (junk token from ABC mktemp) must be unmatched.

        After cleanup_naming.py cleans all ZIP members, messy names should never
        appear.  This test confirms that if one somehow does, it is NOT silently
        processed into a .pt with a messy stem — it is rejected as unmatched.
        """
        design_dir = self.root / "sqrt"
        design_dir.mkdir()
        (design_dir / "tier1").mkdir()

        zip_path = design_dir / "tier1" / "sqrt_Deepsyn.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Messy name: mktemp token "Deepsyn_sqrt_M7pRPw" inserted after tier1_
            zf.writestr(
                "sqrt_Deepsyn_tier1_Deepsyn_sqrt_M7pRPw_syn81_step9.aig", "dummy"
            )
            # Clean name alongside — should still be matched
            zf.writestr("sqrt_Deepsyn_tier1_syn81_step9.aig", "dummy")

        tasks, unmatched, source_counts = discover_graph_tasks(
            self.root, allow_unmatched_names=True
        )
        # Only the clean name should produce a task; the messy one is unmatched.
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].filename, "sqrt_Deepsyn_tier1_syn81_step9.aig")
        self.assertEqual(unmatched, 1)

    def test_process_task_skips_existing(self):
        """process_task returns skipped:exists when the .pt is already present and overwrite=False."""
        final_out = self.root / "out"
        pt_path = final_out / "graphs" / "tier0" / "adder" / "adder_syn1_step0.pt"
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path.write_bytes(b"\x00" * 8)

        task = GraphTask(1, 0, "", "adder", "adder_syn1_step0.aig", "dummy.aig", "", "")
        cfg = WorkerConfig(final_out_root=str(final_out), overwrite=False)

        result = process_task(task, cfg)
        self.assertEqual(result["status"], "skipped:exists")
        self.assertEqual(result["output_path"], str(pt_path))
        # Source file should not have been touched.
        self.assertEqual(pt_path.read_bytes(), b"\x00" * 8)

    def test_process_task_overwrite_attempts_processing(self):
        """When overwrite=True, process_task does not skip even when .pt exists.

        The task will fail (no real AIG at the dummy path) but the key assertion
        is that the status is NOT 'skipped:exists'.
        """
        final_out = self.root / "out"
        pt_path = (
            final_out
            / "graphs"
            / "tier1"
            / "Deepsyn"
            / "sqrt"
            / "sqrt_Deepsyn_tier1_syn1_step1.pt"
        )
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path.write_bytes(b"\x00" * 8)

        task = GraphTask(
            1,
            1,
            "Deepsyn",
            "sqrt",
            "sqrt_Deepsyn_tier1_syn1_step1.aig",
            "nonexistent.aig",
            "",
            "",
        )
        cfg = WorkerConfig(final_out_root=str(final_out), overwrite=True)

        result = process_task(task, cfg)
        self.assertNotEqual(
            result["status"],
            "skipped:exists",
            "overwrite=True must not short-circuit to skipped:exists",
        )
        # Without a real AIG file, processing will error — that is expected here.
        self.assertEqual(result["status"], "error")


class TestDatasetAigNodeLabelAudit(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_DATASET_AIG_AUDIT", "0") == "1",
        "Set RUN_DATASET_AIG_AUDIT=1 to run dataset-level AIG label audit",
    )
    def test_dataset_networkx_node_labels_are_contiguous(self) -> None:
        import aigverse.adapters  # noqa: F401
        from aigverse import read_aiger_into_aig

        repo_root = Path(__file__).resolve().parents[2]
        designs_root = repo_root / "data" / "designs"
        self.assertTrue(designs_root.exists(), f"Missing designs root: {designs_root}")

        aig_files = sorted(designs_root.rglob("*.aig"))
        if not aig_files:
            self.skipTest("No local .aig files under data/designs")

        by_design = defaultdict(list)
        for path in aig_files:
            rel_parts = path.relative_to(designs_root).parts
            design = rel_parts[0] if rel_parts else "UNKNOWN"
            by_design[design].append(path)

        priority_designs = [
            "128",
            "256",
            "512",
            "1024",
            "2048",
            "4096",
            "8192",
            "16384",
            "aes",
            "i2c",
            "c5315",
        ]

        ordered_designs = []
        seen = set()
        for design in priority_designs:
            if design in by_design:
                ordered_designs.append(design)
                seen.add(design)
        for design in sorted(by_design):
            if design not in seen:
                ordered_designs.append(design)

        max_per_design = int(os.getenv("AIG_AUDIT_MAX_PER_DESIGN", "25"))
        checked = 0
        non_contiguous = []
        errors = []

        for design in ordered_designs:
            for path in by_design[design][:max_per_design]:
                checked += 1
                try:
                    aig = read_aiger_into_aig(str(path))
                    graph = aig.to_networkx(levels=True)
                    node_labels = list(graph.nodes())
                    n = len(node_labels)
                    label_set = set(node_labels)
                    expected = set(range(n))
                    if label_set != expected:
                        missing = sorted(expected - label_set)[:10]
                        extra = sorted(label_set - expected)[:10]
                        non_contiguous.append(
                            {
                                "design": design,
                                "path": str(path),
                                "num_nodes": n,
                                "min": min(node_labels) if node_labels else None,
                                "max": max(node_labels) if node_labels else None,
                                "missing_head": missing,
                                "extra_head": extra,
                            }
                        )
                except Exception as exc:
                    errors.append(
                        {"design": design, "path": str(path), "error": repr(exc)}
                    )

        print(f"AIG audit checked files: {checked}")
        print(f"AIG audit designs with AIG files: {len(by_design)}")
        print(f"AIG audit non-contiguous: {len(non_contiguous)}")
        print(f"AIG audit errors: {len(errors)}")

        if "128" in by_design:
            checked_128 = min(len(by_design["128"]), max_per_design)
            bad_128 = [row for row in non_contiguous if row["design"] == "128"]
            print(
                "AIG audit design 128: "
                f"checked={checked_128} total_local={len(by_design['128'])} non_contiguous={len(bad_128)}"
            )

        self.assertEqual(
            len(errors),
            0,
            f"AIG audit encountered read/parse errors. Example: {errors[:3]}",
        )
        self.assertEqual(
            len(non_contiguous),
            0,
            f"Found non-contiguous node labels. Example: {non_contiguous[:3]}",
        )


if __name__ == "__main__":
    unittest.main()
