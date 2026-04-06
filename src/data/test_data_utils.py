import unittest
import os
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.data.data_utils import aig_to_pytorch_geometric


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

        self.assertEqual(data.rel_edge_index.ndim, 2)
        self.assertEqual(data.rel_edge_index.shape[0], 2)
        self.assertEqual(data.edge_rel_dist.ndim, 2)
        self.assertEqual(data.edge_rel_dist.shape[1], 1)
        self.assertEqual(data.edge_rel_dist.shape[0], data.rel_edge_index.shape[1])

        self.assertEqual(data.num_nodes, data.x.shape[0])
        self.assertEqual(data.num_edges, data.edge_index.shape[1])
        self.assertGreaterEqual(data.num_pis, 0)
        self.assertGreaterEqual(data.num_pos, 0)

        for row_sum in data.x.sum(dim=1).tolist():
            self.assertAlmostEqual(row_sum, 1.0, places=6)

    def test_non_contiguous_networkx_labels_are_remapped(self) -> None:
        class FakeAig:
            def num_pis(self) -> int:
                return 1

            def num_pos(self) -> int:
                return 1

            def to_networkx(self, levels: bool = True, dtype=np.float32) -> nx.DiGraph:
                g = nx.DiGraph()
                g.add_node(10, type=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype), level=0.0)
                g.add_node(42, type=np.array([0.0, 1.0, 0.0, 0.0], dtype=dtype), level=0.0)
                g.add_node(99, type=np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype), level=1.0)
                g.add_node(120, type=np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype), level=2.0)

                g.add_edge(42, 99, type=np.array([1.0, 0.0], dtype=dtype))
                g.add_edge(99, 120, type=np.array([1.0, 0.0], dtype=dtype))
                return g

        with patch("src.data.data_utils.read_aiger_into_aig", return_value=FakeAig()):
            data = aig_to_pytorch_geometric("dummy.aig")

        self.assertEqual(data.num_nodes, 4)
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_index.shape[1], 2)
        self.assertGreaterEqual(int(data.edge_index.min().item()), 0)
        self.assertLess(int(data.edge_index.max().item()), data.num_nodes)


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
                    errors.append({"design": design, "path": str(path), "error": repr(exc)})

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
