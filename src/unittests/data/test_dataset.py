from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Data

from data.data_utils import aig_to_pytorch_geometric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AIG_PATH = Path(__file__).with_name("adder.aig")


def _make_graph_pt(dest: Path) -> Path:
    """Process adder.aig into a .pt file and return the path."""
    data = aig_to_pytorch_geometric(_AIG_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, dest)
    return dest


def _make_graph_pts(dest_dir: Path, n: int) -> list[Path]:
    """Create *n* distinct .pt files (copies of adder) in dest_dir."""
    data = aig_to_pytorch_geometric(_AIG_PATH)
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        p = dest_dir / f"graph_{i:04d}.pt"
        torch.save(data, p)
        paths.append(p)
    return paths


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "unoptimized_graph_path",
        "design",
        "algorithm",
        "tier_id",
        "optimizability",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_partition_graph_pt(dest: Path) -> Path:
    data = Data(
        x=torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 2, 3], [1, 0, 2, 0, 3, 2]],
            dtype=torch.long,
        ),
        edge_attr=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 3.0], [3.0, 2.0]],
            dtype=torch.float32,
        ),
        pos_enc=torch.tensor([[100.0], [101.0], [102.0], [103.0]], dtype=torch.float32),
        level=torch.tensor([[200.0], [201.0], [202.0], [203.0]], dtype=torch.float32),
        pi_paths=torch.tensor(
            [[300.0], [301.0], [302.0], [303.0]], dtype=torch.float32
        ),
        local_sp_sum=torch.tensor(
            [[400.0], [401.0], [402.0], [403.0]], dtype=torch.float32
        ),
        edge_weight=torch.tensor(
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=torch.float32
        ),
        random_dynamic_mask=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        random_dynamic_num_partitions=torch.tensor([2], dtype=torch.long),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, dest)
    return dest


def _make_rows(pt_paths: list[Path], *, opt_start: float = 0.1) -> list[dict]:
    """One CSV row per unique .pt file."""
    return [
        {
            "unoptimized_graph_path": str(p),
            "design": "adder",
            "algorithm": "Orchestrate",
            "tier_id": "1",
            "optimizability": str(round(opt_start + i * 0.01, 4)),
        }
        for i, p in enumerate(pt_paths)
    ]


def _make_design_rows(
    root: Path,
    design_names: list[str],
    *,
    graphs_per_design: int,
    opt_start: float = 0.1,
) -> list[dict]:
    """Create rows whose graph paths encode the design in the folder structure."""
    data = aig_to_pytorch_geometric(_AIG_PATH)
    rows: list[dict] = []
    opt_value = opt_start
    for design in design_names:
        design_dir = root / "designs" / design / "tier0"
        design_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(graphs_per_design):
            graph_path = design_dir / f"{design}_{idx:04d}.pt"
            torch.save(data, graph_path)
            rows.append(
                {
                    "unoptimized_graph_path": str(graph_path),
                    "design": design,
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": str(round(opt_value, 4)),
                }
            )
            opt_value += 0.01
    return rows


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestAIGGraphRegressionDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 10 unique .pt files — one per CSV row
        self.pt_paths = _make_graph_pts(self.root / "graphs", 10)
        self.csv_path = self.root / "orchestrate.csv"
        _write_csv(self.csv_path, _make_rows(self.pt_paths))

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ds(self, **kwargs):
        from data.dataset import AIGGraphRegressionDataset

        csv_paths = kwargs.pop("csv_paths", self.csv_path)
        return AIGGraphRegressionDataset(csv_paths, **kwargs)

    # --- PyG root sentinel regression ---

    def test_no_question_mark_folder_created_without_cache_dir(self):
        """Regression: creating a dataset without cache_dir must never create a
        '???' directory (PyG's MISSING sentinel for root=None) in the CWD or
        anywhere under the tmp directory."""
        import os

        orig_cwd = os.getcwd()
        try:
            # Change to a controlled temp dir so any stray '???' appears here
            os.chdir(self.tmp.name)
            self._make_ds()
        finally:
            os.chdir(orig_cwd)

        sentinel = Path(self.tmp.name) / "???"
        self.assertFalse(
            sentinel.exists(),
            f"PyG '???' sentinel directory was created at {sentinel}",
        )
        # Also check the original CWD
        sentinel_cwd = Path(orig_cwd) / "???"
        self.assertFalse(
            sentinel_cwd.exists(),
            f"PyG '???' sentinel directory was created in CWD at {sentinel_cwd}",
        )

    def test_no_question_mark_folder_created_with_cache_dir(self):
        """Even with cache_dir set, '???' must never appear."""
        import os

        cache_dir = self.root / "cache_for_sentinel_test"
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            self._make_ds(cache_dir=cache_dir)
        finally:
            os.chdir(orig_cwd)

        sentinel = Path(self.tmp.name) / "???"
        self.assertFalse(sentinel.exists(), f"Stray '???' created at {sentinel}")

    # --- basic ---

    def test_len(self):
        ds = self._make_ds()
        self.assertEqual(len(ds), 10)

    def test_getitem_y_shape(self):
        ds = self._make_ds()
        item = ds[0]
        self.assertEqual(item.y.shape, (1, 1))
        self.assertAlmostEqual(item.y[0, 0].item(), 0.1, places=4)

    def test_getitem_graph_attributes(self):
        ds = self._make_ds()
        item = ds[0]
        self.assertEqual(item.x.dim(), 2)
        self.assertEqual(item.edge_index.shape[0], 2)
        self.assertEqual(item.edge_attr.dim(), 2)
        self.assertEqual(item.edge_attr.shape[0], item.edge_index.shape[1])

    def test_torch_load_graph_uses_plain_load(self):
        ds = self._make_ds()
        loaded_graph = torch.load(self.pt_paths[0], weights_only=False)
        call_kwargs = []

        def _fake_load(*args, **kwargs):
            call_kwargs.append(dict(kwargs))
            return loaded_graph

        with patch("data.dataset.torch.load", side_effect=_fake_load):
            result = ds._torch_load_graph(self.pt_paths[0])

        self.assertIs(result, loaded_graph)
        self.assertNotIn("mmap", call_kwargs[0])

    def test_candidate_sample_reader_uses_minimal_columns(self):
        import data.dataset as dataset_module

        dataset_module._CSV_SAMPLE_CACHE.clear()
        observed_usecols: list[tuple[str, ...]] = []
        original_read_csv = dataset_module.pd.read_csv

        def _spy_read_csv(*args, **kwargs):
            usecols = kwargs.get("usecols", ())
            observed_usecols.append(tuple(usecols))
            return original_read_csv(*args, **kwargs)

        try:
            with (
                patch("data.dataset.pd.read_csv", side_effect=_spy_read_csv),
                patch(
                    "pandas.core.frame.DataFrame.to_dict",
                    side_effect=AssertionError(
                        "_read_candidate_samples should not call DataFrame.to_dict"
                    ),
                ),
            ):
                ds = self._make_ds()
            self.assertEqual(len(ds), 10)
            self.assertTrue(observed_usecols)
            self.assertEqual(
                set(observed_usecols[0]),
                {"unoptimized_graph_path", "optimizability"},
            )
        finally:
            dataset_module._CSV_SAMPLE_CACHE.clear()

    # legacy memory-release tests removed after refactor; leak fixed via weights_only=True

    # --- positional encoding ---

    def test_pos_enc_none_by_default(self):
        ds = self._make_ds()
        item = ds[0]
        # PyG raises AttributeError for missing keys; use getattr
        self.assertIsNone(getattr(item, "pos_enc", None))
        # other positional-encoding attributes may be retained on the object

    def test_pos_enc_level(self):
        ds = self._make_ds(positional_encoding="level")
        item = ds[0]
        pe = getattr(item, "pos_enc", None)
        self.assertIsNotNone(pe)
        self.assertEqual(pe.shape, (item.x.shape[0], 1))

    def test_partition_random_returns_partitioned_data_with_all_nodes(self):
        """partition='random' keeps all nodes and edges but adds partition_id labels."""
        partition_pt = _make_partition_graph_pt(self.root / "partition_graph.pt")
        partition_csv = self.root / "partition.csv"
        _write_csv(
            partition_csv,
            [
                {
                    "unoptimized_graph_path": str(partition_pt),
                    "design": "adder",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.25",
                }
            ],
        )

        from data.partition_utils import PartitionedData

        ds = self._make_ds(
            csv_paths=partition_csv,
            positional_encoding="pi_paths",
            partition="random",
            seed=7,
        )
        item = ds[0]

        # All 4 nodes are retained (no node filtering)
        self.assertEqual(item.x.shape[0], 4)
        # Cross-partition edges are dropped; only intra-partition edges survive
        self.assertLessEqual(item.edge_index.shape[1], 6)
        # Returns a PartitionedData instance
        self.assertIsInstance(item, PartitionedData)
        # partition_id labels every node
        self.assertEqual(item.partition_id.shape, (4,))
        self.assertEqual(item.num_partitions.item(), 2)
        # Raw PE source attrs are cleaned up
        self.assertFalse(hasattr(item, "level"))
        self.assertFalse(hasattr(item, "pi_paths"))
        self.assertFalse(hasattr(item, "local_sp_sum"))

    def test_partition_none_does_not_add_partition_labels(self):
        partition_pt = _make_partition_graph_pt(self.root / "partition_graph_off.pt")
        partition_csv = self.root / "partition_off.csv"
        _write_csv(
            partition_csv,
            [
                {
                    "unoptimized_graph_path": str(partition_pt),
                    "design": "adder",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.25",
                }
            ],
        )

        ds = self._make_ds(csv_paths=partition_csv, partition=None)
        item = ds[0]

        self.assertFalse(hasattr(item, "partition_id"))
        self.assertFalse(hasattr(item, "num_partitions"))
        self.assertEqual(item.edge_index.shape[1], 6)
        torch.testing.assert_close(
            item.pos_enc.squeeze(-1), torch.tensor([100.0, 101.0, 102.0, 103.0])
        )

    def test_partition_is_applied_on_repeated_access(self):
        partition_pt = _make_partition_graph_pt(self.root / "partition_graph_local.pt")
        partition_csv = self.root / "partition_local.csv"
        _write_csv(
            partition_csv,
            [
                {
                    "unoptimized_graph_path": str(partition_pt),
                    "design": "adder",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.25",
                }
            ],
        )

        cache_dir = self.root / "shared_cache"
        ds = self._make_ds(
            csv_paths=partition_csv,
            cache_dir=cache_dir,
            partition="random",
        )

        first_item = ds[0]
        self.assertTrue(hasattr(first_item, "partition_id"))

        item = ds[0]

        self.assertTrue(hasattr(item, "partition_id"))
        self.assertEqual(item.num_partitions.item(), 2)

    def test_only_pos_enc_tensor_retained_for_each_pe_mode(self):
        for pe_name in ("level", "pi_paths", "local_sp_sum"):
            with self.subTest(pe_name=pe_name):
                ds = self._make_ds(positional_encoding=pe_name)
                item = ds[0]
                self.assertIsNotNone(getattr(item, "pos_enc", None))
                self.assertFalse(hasattr(item, "level"))
                self.assertFalse(hasattr(item, "pi_paths"))
                self.assertFalse(hasattr(item, "local_sp_sum"))

    # --- sparsification ---

    def test_sparsification_mask_applied_correctly(self):
        # Create a graph with a pre-saved mask
        pt_path = self.root / "sparse_graph.pt"
        data = Data(
            x=torch.ones((4, 2)),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
            edge_weight=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        # 4 edges total. Let's keep the first and last (mask = [True, False, False, True])
        data.random_edge_dropout_sparsification_mask = torch.tensor([True, False, False, True])
        
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, pt_path)
        
        csv_path = self.root / "sparse.csv"
        _write_csv(
            csv_path,
            [
                {
                    "unoptimized_graph_path": str(pt_path),
                    "design": "dummy",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.5",
                }
            ],
        )

        # 1. Test without sparsification
        ds_none = self._make_ds(csv_paths=csv_path, sparsification=None, normalize_edges=True)
        item_none = ds_none[0]
        self.assertEqual(item_none.edge_index.shape[1], 4)
        self.assertEqual(item_none.edge_attr.shape[0], 4)
        self.assertEqual(item_none.edge_weight.shape[0], 4)

        # 2. Test with sparsification
        ds_sparse = self._make_ds(csv_paths=csv_path, sparsification="random_edge_dropout", normalize_edges=True)
        item_sparse = ds_sparse[0]
        self.assertEqual(item_sparse.edge_index.shape[1], 2)
        self.assertEqual(item_sparse.edge_attr.shape[0], 2)
        self.assertEqual(item_sparse.edge_weight.shape[0], 2)

        # Check exact values
        torch.testing.assert_close(item_sparse.edge_index, torch.tensor([[0, 3], [1, 0]], dtype=torch.long))
        torch.testing.assert_close(item_sparse.edge_attr, torch.tensor([[0.1], [0.4]]))
        torch.testing.assert_close(item_sparse.edge_weight, torch.tensor([1.0, 4.0]))

    def test_node_sparsification_mask_applied_correctly(self):
        # Create a graph with a pre-saved node mask
        pt_path = self.root / "node_sparse_graph.pt"
        data = Data(
            x=torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
            edge_weight=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        # Keep nodes 0, 1, and 3 (mask = [True, True, False, True])
        data.pagerank_sparsification_mask = torch.tensor([True, True, False, True])
        
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, pt_path)
        
        csv_path = self.root / "node_sparse.csv"
        _write_csv(
            csv_path,
            [
                {
                    "unoptimized_graph_path": str(pt_path),
                    "design": "dummy_node",
                    "algorithm": "Orchestrate",
                    "tier_id": "1",
                    "optimizability": "0.5",
                }
            ],
        )

        # Test with pagerank sparsification
        ds_sparse = self._make_ds(csv_paths=csv_path, sparsification="pagerank", normalize_edges=True)
        item_sparse = ds_sparse[0]
        
        # Subgraph keeps nodes 0, 1, 3 (which are mapped to new indices 0, 1, 2)
        self.assertEqual(item_sparse.num_nodes, 3)
        self.assertEqual(item_sparse.edge_index.shape[1], 2)
        
        # Check values
        torch.testing.assert_close(item_sparse.x, torch.tensor([[1.0], [2.0], [4.0]]))

    # --- num_samples ---

    def test_num_samples_limits_dataset(self):
        # 20 unique paths, ask for only 5
        extra_paths = _make_graph_pts(self.root / "graphs_extra", 20)
        csv2 = self.root / "big.csv"
        _write_csv(csv2, _make_rows(extra_paths))
        from data.dataset import AIGGraphRegressionDataset

        ds = AIGGraphRegressionDataset(csv2, num_samples=5)
        self.assertEqual(len(ds), 5)

    def test_num_samples_larger_than_dataset(self):
        # num_samples > len should just return all samples
        ds = self._make_ds(num_samples=999)
        self.assertEqual(len(ds), 10)

    # --- splits ---

    def test_split_total_equals_dataset_size(self):
        from data.dataset import AIGGraphRegressionDataset

        # 20 unique paths
        pts = _make_graph_pts(self.root / "split_graphs", 20)
        csv = self.root / "split.csv"
        _write_csv(csv, _make_rows(pts))

        train_ds = AIGGraphRegressionDataset(csv, split="train", seed=7)
        val_ds = AIGGraphRegressionDataset(csv, split="val", seed=7)
        test_ds = AIGGraphRegressionDataset(csv, split="test", seed=7)
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 20)

    def test_split_train_val_test_are_disjoint(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "disjoint_graphs", 20)
        csv = self.root / "disjoint.csv"
        _write_csv(csv, _make_rows(pts))

        train_paths = {
            s.graph_path
            for s in AIGGraphRegressionDataset(csv, split="train", seed=3).samples
        }
        val_paths = {
            s.graph_path
            for s in AIGGraphRegressionDataset(csv, split="val", seed=3).samples
        }
        test_paths = {
            s.graph_path
            for s in AIGGraphRegressionDataset(csv, split="test", seed=3).samples
        }

        self.assertFalse(train_paths & val_paths, "train and val overlap")
        self.assertFalse(train_paths & test_paths, "train and test overlap")
        self.assertFalse(val_paths & test_paths, "val and test overlap")

    def test_split_approx_ratios(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "ratio_graphs", 100)
        csv = self.root / "ratio.csv"
        _write_csv(csv, _make_rows(pts))

        train_ds = AIGGraphRegressionDataset(csv, split="train", seed=0)
        val_ds = AIGGraphRegressionDataset(csv, split="val", seed=0)
        test_ds = AIGGraphRegressionDataset(csv, split="test", seed=0)
        # With default 0.8/0.1/0.1 ratios expect train ~80, val ~10, test ~10
        self.assertGreaterEqual(len(train_ds), 75)
        self.assertGreaterEqual(len(val_ds), 5)
        self.assertGreaterEqual(len(test_ds), 5)

    def test_split_keeps_designs_together_when_path_encodes_design(self):
        from data.dataset import AIGGraphRegressionDataset

        csv = self.root / "design_grouped.csv"
        design_names = [f"design_{i:02d}" for i in range(10)]
        _write_csv(
            csv,
            _make_design_rows(
                self.root,
                design_names,
                graphs_per_design=3,
            ),
        )

        train_ds = AIGGraphRegressionDataset(csv, split="train", seed=11)
        val_ds = AIGGraphRegressionDataset(csv, split="val", seed=11)
        test_ds = AIGGraphRegressionDataset(csv, split="test", seed=11)

        train_designs = {sample.design_key for sample in train_ds.samples}
        val_designs = {sample.design_key for sample in val_ds.samples}
        test_designs = {sample.design_key for sample in test_ds.samples}

        self.assertFalse(train_designs & val_designs, "train and val share designs")
        self.assertFalse(train_designs & test_designs, "train and test share designs")
        self.assertFalse(val_designs & test_designs, "val and test share designs")
        self.assertEqual(train_designs | val_designs | test_designs, set(design_names))
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 30)

    def test_multi_csv_duplicate_graph_paths_do_not_cross_splits(self):
        from data.dataset import AIGGraphRegressionDataset

        rows_a = _make_design_rows(
            self.root,
            [f"design_{i:02d}" for i in range(10)],
            graphs_per_design=2,
        )
        rows_b = [
            {
                **row,
                "algorithm": "Deepsyn",
                "optimizability": str(round(float(row["optimizability"]) + 0.5, 4)),
            }
            for row in rows_a
        ]
        csv_a = self.root / "algo_a.csv"
        csv_b = self.root / "algo_b.csv"
        _write_csv(csv_a, rows_a)
        _write_csv(csv_b, rows_b)

        train_ds = AIGGraphRegressionDataset([csv_a, csv_b], split="train", seed=5)
        val_ds = AIGGraphRegressionDataset([csv_a, csv_b], split="val", seed=5)
        test_ds = AIGGraphRegressionDataset([csv_a, csv_b], split="test", seed=5)

        train_paths = {sample.graph_path for sample in train_ds.samples}
        val_paths = {sample.graph_path for sample in val_ds.samples}
        test_paths = {sample.graph_path for sample in test_ds.samples}

        self.assertFalse(train_paths & val_paths, "train and val share graph paths")
        self.assertFalse(train_paths & test_paths, "train and test share graph paths")
        self.assertFalse(val_paths & test_paths, "val and test share graph paths")
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 40)

    # --- HP tuning split exclusion testing ---

    def test_hp_tuning_splits_path_excludes_samples(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "hp_exclude_graphs", 20)
        csv = self.root / "hp_exclude.csv"
        _write_csv(csv, _make_rows(pts))

        # Pick 5 specific graphs to mock as "used in HP tuning"
        hp_keys = [str(p) for p in pts[:5]]
        hp_splits = {"train": hp_keys[:3], "val": hp_keys[3:4], "test": hp_keys[4:]}

        # Write mock HP tuning JSON to disk
        hp_json_path = self.root / "hp_splits.json"
        hp_json_path.write_text(json.dumps(hp_splits))

        # Initialize dataset, asking for train split and passing the JSON file
        ds = AIGGraphRegressionDataset(
            csv, split="train", seed=0, hp_tuning_splits_path=hp_json_path
        )

        # Original size = 20. Excluded = 5. Remaining pool = 15.
        # Train split ratio is 0.8. 80% of 15 = 12 samples.
        self.assertEqual(len(ds), 12)

        # Verify that none of the HP tuned keys are present in the dataset samples
        ds_paths = {s.graph_path for s in ds.samples}
        for k in hp_keys:
            self.assertNotIn(k, ds_paths)

    # --- cache ---

    def test_cache_file_created(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "cache_graphs", 10)
        csv = self.root / "cache.csv"
        _write_csv(csv, _make_rows(pts))
        cache_dir = self.root / "cache"

        AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=1)
        cache_files = list(cache_dir.glob("*_splits.json"))
        self.assertEqual(len(cache_files), 1)

    def test_cache_file_contains_all_splits(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "cache2_graphs", 10)
        csv = self.root / "cache2.csv"
        _write_csv(csv, _make_rows(pts))
        cache_dir = self.root / "cache2"

        AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=1)
        cache_file = next(cache_dir.glob("*_splits.json"))
        splits = json.loads(cache_file.read_text())
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        self.assertIn("__meta__", splits)
        self.assertEqual(splits["__meta__"]["split_by"], "design")

    def test_cache_loaded_on_second_call(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "cache3_graphs", 10)
        csv = self.root / "cache3.csv"
        _write_csv(csv, _make_rows(pts))
        cache_dir = self.root / "cache3"

        ds1 = AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=5)
        paths1 = [s.graph_path for s in ds1.samples]

        # second call with same args should load from cache and get identical result
        cache_file = next(cache_dir.glob("*_splits.json"))
        self.assertTrue(cache_file.is_file())
        ds2 = AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=5)
        paths2 = [s.graph_path for s in ds2.samples]
        self.assertEqual(paths1, paths2)

    def test_cache_not_created_without_cache_dir(self):
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "nocache_graphs", 10)
        csv = self.root / "nocache.csv"
        _write_csv(csv, _make_rows(pts))

        # No cache_dir → no JSON file anywhere in tmp
        AIGGraphRegressionDataset(csv, split="train", seed=0)
        json_files = list(self.root.rglob("*.json"))
        self.assertEqual(len(json_files), 0)

    def test_global_num_nodes_written_to_cache_dirs(self):
        """_num_nodes_global.json is created in cache_graph_dir and tier0_cache_dir."""
        from data.dataset import AIGGraphRegressionDataset

        tier0_dir = self.root / "designs" / "i2c" / "tier0"
        tier0_pts = _make_graph_pts(tier0_dir, 4)
        regular_pts = _make_graph_pts(self.root / "graphs_regular", 4)
        csv = self.root / "global_nn_written.csv"
        _write_csv(csv, _make_rows(tier0_pts) + _make_rows(regular_pts, opt_start=0.5))

        cache_dir = self.root / "gnn_cache"
        tier0_cache_dir = self.root / "gnn_tier0"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )

        self.assertTrue(
            (cache_dir / "processed_graphs" / "_num_nodes_global.json").is_file(),
            "_num_nodes_global.json not written to cache_graph_dir",
        )
        self.assertTrue(
            (tier0_cache_dir / "_num_nodes_global.json").is_file(),
            "_num_nodes_global.json not written to tier0_cache_dir",
        )

    def test_global_num_nodes_skip_load_on_rerun(self):
        """After global map is saved, a manifest rebuild must not call _torch_load_graph."""
        from unittest.mock import patch

        from data.dataset import AIGGraphRegressionDataset

        tier0_dir = self.root / "designs" / "sin" / "tier0"
        tier0_pts = _make_graph_pts(tier0_dir, 5)
        csv = self.root / "rerun_skip.csv"
        _write_csv(csv, _make_rows(tier0_pts))

        cache_dir = self.root / "rerun_cache"
        tier0_cache_dir = self.root / "rerun_tier0"

        # First run: builds .pt files + global map
        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )

        # Simulate SLURM kill: delete the train manifest so it must be rebuilt
        for mf in (cache_dir / "metadata").glob("*_manifest.json"):
            mf.unlink()

        # Second run: all .pt files exist + global map has all num_nodes → no loads
        load_calls: list = []
        _orig_load = AIGGraphRegressionDataset._torch_load_graph

        def _counting_load(self_inner, graph_path):
            load_calls.append(graph_path)
            return _orig_load(self_inner, graph_path)

        with patch.object(
            AIGGraphRegressionDataset, "_torch_load_graph", _counting_load
        ):
            AIGGraphRegressionDataset(
                csv,
                split="train",
                cache_dir=cache_dir,
                tier0_cache_dir=tier0_cache_dir,
                seed=0,
            )

        self.assertEqual(
            len(load_calls),
            0,
            f"Expected 0 graph loads on rerun (all in global map), got {len(load_calls)}",
        )

    def test_tier0_graphs_routed_to_tier0_cache_dir(self):
        """Tier-0 graphs (path contains /tier0/) must be cached in tier0_cache_dir,
        not in cache_dir/processed_graphs."""
        from data.dataset import AIGGraphRegressionDataset

        # Create graph files under a path that contains /tier0/
        tier0_dir = self.root / "designs" / "i2c" / "tier0"
        tier0_pts = _make_graph_pts(tier0_dir, 5)
        csv = self.root / "tier0_routing.csv"
        _write_csv(csv, _make_rows(tier0_pts))

        cache_dir = self.root / "algo_cache"
        tier0_cache_dir = self.root / "shared_tier0_cache"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )

        # Tier-0 .pt files must appear in tier0_cache_dir
        shared_pts = list(tier0_cache_dir.rglob("*.pt"))
        self.assertGreater(
            len(shared_pts), 0, "No .pt files written to tier0_cache_dir"
        )

        # No .pt files should appear under cache_dir/processed_graphs
        algo_pts = list((cache_dir / "processed_graphs").rglob("*.pt"))
        self.assertEqual(
            len(algo_pts),
            0,
            "Tier-0 .pt files were incorrectly written to algo cache_dir",
        )

    def test_tier0_cache_shared_across_two_algorithms(self):
        """Two datasets with different cache_dirs but the same tier0_cache_dir
        produce exactly one copy of each tier-0 graph."""
        from data.dataset import AIGGraphRegressionDataset

        tier0_dir = self.root / "designs" / "aes" / "tier0"
        tier0_pts = _make_graph_pts(tier0_dir, 4)
        csv = self.root / "shared_tier0.csv"
        _write_csv(csv, _make_rows(tier0_pts))

        tier0_cache_dir = self.root / "shared_tier0"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=self.root / "algo_a",
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )
        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=self.root / "algo_b",
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )

        # Shared cache has exactly as many unique .pt files as unique source graphs
        unique_source_hashes = {
            p.name for p in (self.root / "shared_tier0").rglob("*.pt")
        }
        self.assertGreater(len(unique_source_hashes), 0)
        # Both algorithm cache dirs should have NO .pt files
        self.assertEqual(
            len(list((self.root / "algo_a" / "processed_graphs").rglob("*.pt"))), 0
        )
        self.assertEqual(
            len(list((self.root / "algo_b" / "processed_graphs").rglob("*.pt"))), 0
        )

    def test_tier1_graphs_routed_to_tier1_cache_dir(self):
        """Tier-1 graphs must be cached in tier1_cache_dir, not per-algorithm processed_graphs."""
        from data.dataset import AIGGraphRegressionDataset

        tier1_dir = self.root / "graphs" / "tier1" / "Orchestrate" / "i2c"
        tier1_pts = _make_graph_pts(tier1_dir, 5)
        csv = self.root / "tier1_routing.csv"
        _write_csv(csv, _make_rows(tier1_pts))

        cache_dir = self.root / "algo_cache"
        tier1_cache_dir = self.root / "shared_tier1_cache"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            tier1_cache_dir=tier1_cache_dir,
            seed=0,
        )

        shared_pts = list(tier1_cache_dir.rglob("*.pt"))
        self.assertGreater(
            len(shared_pts), 0, "No .pt files written to tier1_cache_dir"
        )

        algo_pts = list((cache_dir / "processed_graphs").rglob("*.pt"))
        self.assertEqual(
            len(algo_pts),
            0,
            "Tier-1 .pt files were incorrectly written to algo cache_dir",
        )

    def test_tier1_cache_shared_across_two_algorithms(self):
        """Two datasets with different cache_dirs but the same tier1_cache_dir
        produce exactly one copy of each tier-1 graph."""
        from data.dataset import AIGGraphRegressionDataset

        tier1_dir = self.root / "graphs" / "tier1" / "Deepsyn" / "aes"
        tier1_pts = _make_graph_pts(tier1_dir, 4)
        csv = self.root / "shared_tier1.csv"
        _write_csv(csv, _make_rows(tier1_pts))

        tier1_cache_dir = self.root / "shared_tier1"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=self.root / "algo_a",
            tier1_cache_dir=tier1_cache_dir,
            seed=0,
        )
        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=self.root / "algo_b",
            tier1_cache_dir=tier1_cache_dir,
            seed=0,
        )

        unique_source_hashes = {p.name for p in tier1_cache_dir.rglob("*.pt")}
        self.assertGreater(len(unique_source_hashes), 0)
        self.assertEqual(
            len(list((self.root / "algo_a" / "processed_graphs").rglob("*.pt"))), 0
        )
        self.assertEqual(
            len(list((self.root / "algo_b" / "processed_graphs").rglob("*.pt"))), 0
        )

    def test_manifest_stores_cache_path(self):
        """New manifests store absolute cache_path per entry (not cache_name)."""
        import json as _json

        from data.dataset import AIGGraphRegressionDataset

        tier0_dir = self.root / "designs" / "fir" / "tier0"
        tier0_pts = _make_graph_pts(tier0_dir, 3)
        csv = self.root / "manifest_path.csv"
        _write_csv(csv, _make_rows(tier0_pts))

        cache_dir = self.root / "manifest_cache"
        tier0_cache_dir = self.root / "manifest_tier0"

        AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            tier0_cache_dir=tier0_cache_dir,
            seed=0,
        )

        manifest_files = list((cache_dir / "metadata").glob("*_manifest.json"))
        self.assertEqual(len(manifest_files), 1)
        manifest = _json.loads(manifest_files[0].read_text())
        for entry in manifest["entries"]:
            self.assertIn("cache_path", entry, "entry missing 'cache_path'")
            # cache_path should be inside tier0_cache_dir
            self.assertTrue(
                entry["cache_path"].startswith(str(tier0_cache_dir)),
                f"cache_path {entry['cache_path']!r} not under tier0_cache_dir",
            )

    def test_apply_manifest_backward_compat_cache_name(self):
        """_apply_manifest falls back to cache_dir/processed_graphs when entry
        has legacy 'cache_name' instead of 'cache_path'."""
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "compat_graphs", 3)
        csv = self.root / "compat.csv"
        _write_csv(csv, _make_rows(pts))
        cache_dir = self.root / "compat_cache"

        ds = AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=0)

        # Simulate a v1 manifest entry with cache_name only
        fake_manifest = {
            "version": 1,
            "num_samples": len(ds.samples),
            "entries": [
                {
                    "graph_path": s.graph_path,
                    "cache_name": "deadbeef.pt",
                    "num_nodes": 10,
                }
                for s in ds.samples
            ],
        }
        ds._apply_manifest(fake_manifest)
        # All paths should resolve under _cache_graph_dir
        for path in ds._graph_cache_path_map.values():
            self.assertEqual(path.parent, ds._cache_graph_dir)
            self.assertEqual(path.name, "deadbeef.pt")

    def test_stale_manifest_is_ignored_when_sample_set_changes(self):
        """A manifest whose graph_path entries no longer match the current split
        must be rebuilt instead of reusing stale node sizes/path maps."""
        from data.dataset import AIGGraphRegressionDataset

        csv = self.root / "manifest_stale.csv"
        _write_csv(
            csv,
            _make_design_rows(
                self.root,
                [f"design_{i:02d}" for i in range(10)],
                graphs_per_design=2,
            ),
        )
        cache_dir = self.root / "manifest_stale_cache"

        ds = AIGGraphRegressionDataset(csv, split="train", cache_dir=cache_dir, seed=0)
        manifest_file = next((cache_dir / "metadata").glob("*_manifest.json"))
        manifest = json.loads(manifest_file.read_text())

        stale_path = str(self.root / "graphs" / "old_graph.pt")
        manifest["entries"] = [
            {
                "graph_path": stale_path,
                "cache_path": str(cache_dir / "processed_graphs" / "old_graph.pt"),
                "num_nodes": 123,
            }
        ]
        manifest["num_samples"] = 1
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))

        ds_reloaded = AIGGraphRegressionDataset(
            csv,
            split="train",
            cache_dir=cache_dir,
            seed=0,
        )

        self.assertEqual(len(ds_reloaded.samples), len(ds.samples))
        self.assertNotIn(stale_path, ds_reloaded._graph_cache_path_map)
        self.assertEqual(
            len(ds_reloaded.get_num_nodes_list()), len(ds_reloaded.samples)
        )

    def test_pos_enc_continuous_is_float(self):
        """Test that continuous features like 'pi_paths' are converted to floats."""
        ds = self._make_ds(positional_encoding="pi_paths")
        item = ds[0]
        pe = getattr(item, "pos_enc", None)
        self.assertIsNotNone(pe)
        self.assertEqual(pe.shape, (item.x.shape[0], 1))
        # Unlike 'level' which is a long, continuous PE should be float32
        self.assertTrue(pe.dtype in [torch.float32, torch.float64])

    # --- edge_attr validation ---

    def test_getitem_allows_edge_attr_missing(self):
        """Lean mode: dataset no longer pre-validates edge_attr at load time."""
        bad_pt = self.root / "bad_graph.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.edge_attr = None  # Corrupt the data
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        ds = AIGGraphRegressionDataset(bad_csv)
        item = ds[0]
        self.assertIsNone(item.edge_attr)

    def test_getitem_allows_edge_attr_1d(self):
        """Lean mode: dataset no longer enforces edge_attr dimensionality."""
        bad_pt = self.root / "bad_graph_1d.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.edge_attr = torch.tensor([1.0, 0.0])  # 1D instead of 2D
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad_1d.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        ds = AIGGraphRegressionDataset(bad_csv)
        item = ds[0]
        self.assertEqual(item.edge_attr.dim(), 1)

    # --- dataset initialization verification ---

    def test_bad_x_is_not_prevalidated(self):
        """Lean mode: dataset initialization does not assert on malformed x shape."""
        bad_pt = self.root / "bad_x_graph.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.x = torch.rand(10)  # 1D instead of 2D
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad_x.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        ds = AIGGraphRegressionDataset(bad_csv)
        self.assertEqual(ds[0].x.dim(), 1)

    # --- seed stability ---

    def test_different_seeds_produce_different_splits(self):
        """Ensure the RNG splits the data differently when the seed changes."""
        from data.dataset import AIGGraphRegressionDataset

        pts = _make_graph_pts(self.root / "seed_graphs", 30)
        csv = self.root / "seed.csv"
        _write_csv(csv, _make_rows(pts))

        ds1 = AIGGraphRegressionDataset(csv, split="train", seed=42)
        ds2 = AIGGraphRegressionDataset(csv, split="train", seed=99)

        paths1 = [s.graph_path for s in ds1.samples]
        paths2 = [s.graph_path for s in ds2.samples]

        # It's highly unlikely that two distinct seeds produce the exact same split for 30 elements
        self.assertNotEqual(paths1, paths2)

    # --- multi-CSV ---

    def test_multi_csv_concat(self):
        from data.dataset import AIGGraphRegressionDataset

        pts2 = _make_graph_pts(self.root / "graphs2", 10)
        csv2 = self.root / "deepsyn.csv"
        _write_csv(csv2, _make_rows(pts2, opt_start=0.5))
        ds = AIGGraphRegressionDataset([self.csv_path, csv2])
        self.assertEqual(len(ds), 20)


# ---------------------------------------------------------------------------
# DataModule tests
# ---------------------------------------------------------------------------


class TestAIGDataModule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 30 unique .pt files so splits are non-empty
        self.pt_paths = _make_graph_pts(self.root / "graphs", 30)
        self.csv_path = self.root / "orchestrate.csv"
        _write_csv(self.csv_path, _make_rows(self.pt_paths))

    def tearDown(self):
        self.tmp.cleanup()

    def _make_dm(self, **kwargs):
        from data.datamodule import AIGDataModule

        dm = AIGDataModule(self.csv_path, batch_size=4, **kwargs)
        dm.setup()
        return dm

    def test_loaders_exist(self):
        dm = self._make_dm()
        self.assertIsNotNone(dm.train_dataloader())
        self.assertIsNotNone(dm.val_dataloader())

    def test_persistent_workers_applies_to_all_loader_kwargs(self):
        # persistent_workers applies uniformly to train and val loaders (not
        # train-only) since the loaders are recreated frequently under
        # fractional val_check_interval; see datamodule._loader_kwargs.
        from data.datamodule import AIGDataModule

        dm = AIGDataModule(
            self.csv_path,
            batch_size=4,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=1,
        )

        train_kwargs = dm._loader_kwargs(is_train=True)
        val_kwargs = dm._loader_kwargs(is_train=False)

        self.assertTrue(train_kwargs["persistent_workers"])
        self.assertTrue(val_kwargs["persistent_workers"])

    def test_persistent_workers_applies_to_all_budgeted_dataloaders(self):
        dm = self._make_dm(
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=1,
            dynamic_batching=True,
            max_total_nodes=30,
        )

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        self.assertTrue(getattr(train_loader, "persistent_workers", False))
        self.assertTrue(getattr(val_loader, "persistent_workers", False))

    def test_train_batch_shapes(self):
        dm = self._make_dm()
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch.y.shape[1], 1)
        self.assertEqual(batch.x.dim(), 2)
        self.assertEqual(batch.edge_index.shape[0], 2)
        self.assertEqual(batch.edge_attr.dim(), 2)

    def test_train_num_samples(self):
        # Explicit test_num_samples applies only to the test split.
        dm = self._make_dm(train_num_samples=10, test_num_samples=10)

        # 10 total samples * 80% train ratio = 8 samples in train_ds
        self.assertEqual(len(dm.train_ds), 8)

        # 10 total samples * 10% val ratio = 1 sample in val_ds
        self.assertEqual(len(dm.val_ds), 1)

        # 10 total samples * 10% test ratio = 1 sample in test_ds
        self.assertEqual(len(dm.test_ds), 1)

        # Ensure the total pool across all splits strictly equals the requested num_samples limit
        self.assertEqual(len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds), 10)

    def test_test_num_samples_unset_uses_full_test_split(self):
        # When test_num_samples is None (default), the test split is unlimited.
        dm = self._make_dm(train_num_samples=10)
        # 30 total files, 10% test ratio = 3 test samples (not limited to train pool)
        self.assertEqual(len(dm.test_ds), 3)

    def test_test_loader(self):
        from data.datamodule import AIGDataModule

        dm = AIGDataModule(self.csv_path, batch_size=4)
        dm.setup(stage="test")
        batch = next(iter(dm.test_dataloader()))
        self.assertEqual(batch.y.shape[1], 1)

    def test_test_loader_ignores_dynamic_batching_by_default(self):
        from data.datamodule import AIGDataModule
        from data.sampler import BalancedDynamicBatchSampler

        dm = AIGDataModule(self.csv_path, batch_size=4)
        dm.setup(stage="test")
        self.assertNotIsInstance(
            dm.test_dataloader().batch_sampler, BalancedDynamicBatchSampler
        )

    def test_test_loader_uses_node_budget_when_dynamic_batching(self):
        # test.py packs eval batches to a node budget to fill the GPU; the
        # test loader must honour dynamic_batching the same way val does,
        # and the plan must still cover every test graph exactly once.
        from data.datamodule import AIGDataModule
        from data.sampler import BalancedDynamicBatchSampler

        dm = AIGDataModule(self.csv_path, batch_size=4, dynamic_batching=True)
        dm.setup(stage="test")
        loader = dm.test_dataloader()

        self.assertIsInstance(loader.batch_sampler, BalancedDynamicBatchSampler)
        emitted = [idx for batch in loader.batch_sampler for idx in batch]
        self.assertEqual(sorted(emitted), list(range(len(dm.test_ds))))

    def test_datamodule_split_sizes_sum_to_total(self):
        dm = self._make_dm()
        self.assertEqual(len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds), 30)

    def test_hp_tuning_splits_path_datamodule(self):
        # Pick 5 samples from the 30 existing ones
        hp_keys = [str(p) for p in self.pt_paths[:5]]
        hp_splits = {"train": hp_keys}
        hp_json_path = self.root / "dm_hp_splits.json"
        hp_json_path.write_text(json.dumps(hp_splits))

        # Create DataModule passing the HP split path
        dm = self._make_dm(hp_tuning_splits_path=hp_json_path)

        # Total original is 30. Excluded 5. Total remaining should be 25.
        total_remaining = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
        self.assertEqual(total_remaining, 25)

        # Ensure excluded keys aren't anywhere in the train_ds
        train_paths = {s.graph_path for s in dm.train_ds.samples}
        for k in hp_keys:
            self.assertNotIn(k, train_paths)

    def test_batch_is_correct_and_disjoint(self):
        from torch_geometric.loader import DataLoader

        # Fix: Use the DataModule to get the dataset
        dm = self._make_dm()
        ds = dm.train_ds

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        # 1. Verify batch size
        self.assertEqual(batch.num_graphs, 4)

        # 2. Verify 'batch' vector exists and maps all nodes
        self.assertTrue(hasattr(batch, "batch"))
        self.assertEqual(batch.batch.numel(), batch.x.shape[0])

        # 3. CRITICAL: Check for data leakage in edge_index
        for i in range(batch.num_graphs):
            # Get edges belonging to graph i
            # Check that edge_index values stay within the node range for that specific graph
            current_edge_index = batch.edge_index[
                :,
                (batch.edge_index[0] >= batch.ptr[i])
                & (batch.edge_index[0] < batch.ptr[i + 1]),
            ]

            self.assertTrue((current_edge_index >= batch.ptr[i]).all())
            self.assertTrue((current_edge_index < batch.ptr[i + 1]).all())

    def test_batch_reconstruction_is_lossless(self):
        from torch_geometric.loader import DataLoader

        # Fix: Use the DataModule to get the dataset
        dm = self._make_dm()
        ds = dm.train_ds

        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        # Use PyG's built-in to_data_list to reconstruct individual graphs
        reconstructed_graphs = batch.to_data_list()

        for i in range(len(reconstructed_graphs)):
            original = ds[i]
            reconstructed = reconstructed_graphs[i]

            # Check node features, edge indices, and targets match exactly
            self.assertTrue(torch.equal(original.x, reconstructed.x))
            self.assertTrue(torch.equal(original.edge_index, reconstructed.edge_index))
            self.assertTrue(torch.equal(original.y, reconstructed.y))

            # Check positional encodings if present
            if hasattr(original, "pos_enc"):
                self.assertTrue(torch.equal(original.pos_enc, reconstructed.pos_enc))


class TestBalancedDynamicBatchSampler(unittest.TestCase):
    def test_pairs_large_and_small_in_same_batch(self):
        from data.sampler import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 102, 103]
        sampler = BalancedDynamicBatchSampler(
            sizes,
            batch_size=4,
            shuffle=False,
            seed=123,
            max_total_nodes=105,
        )

        batches = list(sampler)
        self.assertEqual(len(batches), 4)

        for batch in batches:
            batch_sizes = [sizes[i] for i in batch]
            self.assertEqual(len(batch), 2)
            self.assertGreaterEqual(max(batch_sizes), 100)
            self.assertLessEqual(min(batch_sizes), 4)

        flattened = [idx for batch in batches for idx in batch]
        self.assertEqual(len(flattened), len(sizes))
        self.assertEqual(set(flattened), set(range(len(sizes))))

    def test_pairing_reduces_peak_batch_node_total(self):
        from data.sampler import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 102, 103]
        max_total_nodes = 105

        sampler = BalancedDynamicBatchSampler(
            sizes,
            batch_size=4,
            shuffle=False,
            seed=7,
            max_total_nodes=max_total_nodes,
        )
        dynamic_batches = list(sampler)
        dynamic_totals = [sum(sizes[i] for i in batch) for batch in dynamic_batches]

        # Baseline: descending contiguous pairs, which tend to pack large graphs together.
        descending_indices = sorted(
            range(len(sizes)), key=lambda i: sizes[i], reverse=True
        )
        baseline_totals = []
        for start in range(0, len(sizes), 2):
            chunk = descending_indices[start : start + 2]
            baseline_totals.append(sum(sizes[i] for i in chunk))

        self.assertLess(max(dynamic_totals), max(baseline_totals))

    def test_node_budget_has_no_graph_count_cap(self):
        from data.sampler import BalancedDynamicBatchSampler

        sizes = [1] * 40
        sampler = BalancedDynamicBatchSampler(
            sizes,
            batch_size=32,
            shuffle=False,
            seed=11,
            max_total_nodes=100,
        )

        batches = list(sampler)

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 40)

    def test_max_total_nodes_creates_singletons_for_huge_graphs(self):
        from data.sampler import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 300, 350]
        sampler = BalancedDynamicBatchSampler(
            sizes,
            batch_size=8,
            shuffle=False,
            seed=1,
            max_total_nodes=300,
        )

        batches = list(sampler)

        idx_350 = sizes.index(350)
        idx_300 = sizes.index(300)

        batch_350 = next(batch for batch in batches if idx_350 in batch)
        batch_300 = next(batch for batch in batches if idx_300 in batch)
        self.assertEqual(len(batch_350), 1)
        self.assertEqual(len(batch_300), 1)

        flattened = [idx for batch in batches for idx in batch]
        self.assertEqual(set(flattened), set(range(len(sizes))))

    def test_max_total_nodes_fills_with_smallest_graphs_first(self):
        from data.sampler import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 300, 350]
        sampler = BalancedDynamicBatchSampler(
            sizes,
            batch_size=8,
            shuffle=False,
            seed=1,
            max_total_nodes=350,
        )

        batches = list(sampler)
        idx_300 = sizes.index(300)
        batch_300 = next(batch for batch in batches if idx_300 in batch)

        self.assertEqual(sorted(sizes[i] for i in batch_300), [1, 2, 3, 4, 300])
        self.assertLessEqual(sum(sizes[i] for i in batch_300), 350)


# ---------------------------------------------------------------------------
# get_num_nodes_list tests
# ---------------------------------------------------------------------------


class TestGetNumNodesList(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.pt_paths = _make_graph_pts(self.root / "graphs", 5)
        self.csv_path = self.root / "data.csv"
        _write_csv(self.csv_path, _make_rows(self.pt_paths))

    def tearDown(self):
        self.tmp.cleanup()

    def _make_ds(self, **kwargs):
        from data.dataset import AIGGraphRegressionDataset

        return AIGGraphRegressionDataset(self.csv_path, **kwargs)

    def test_returns_one_entry_per_sample(self):
        ds = self._make_ds()
        sizes = ds.get_num_nodes_list()
        self.assertEqual(len(sizes), len(ds))

    def test_all_entries_are_positive_integers(self):
        ds = self._make_ds()
        sizes = ds.get_num_nodes_list()
        for s in sizes:
            self.assertIsInstance(s, int)
            self.assertGreater(s, 0)

    def test_sizes_match_loaded_graphs(self):
        """Node counts returned by get_num_nodes_list must equal actual graph sizes."""
        ds = self._make_ds()
        sizes = ds.get_num_nodes_list()
        for i, s in enumerate(sizes):
            self.assertEqual(s, ds[i].x.shape[0])

    def test_sizes_in_memory_after_setup(self):
        # Sizes are now stored in _node_sizes (from manifest), not a separate JSON file.
        cache_dir = self.root / "cache_sizes"
        ds = self._make_ds(cache_dir=cache_dir)
        sizes = ds.get_num_nodes_list()
        self.assertIsNotNone(ds._node_sizes)
        self.assertEqual(len(sizes), len(ds))
        # No separate node_sizes.json file should be written.
        json_files = list(cache_dir.glob("*_node_sizes.json"))
        self.assertEqual(len(json_files), 0)

    def test_second_call_loads_from_cache(self):
        cache_dir = self.root / "cache_sizes2"
        ds = self._make_ds(cache_dir=cache_dir)
        sizes1 = ds.get_num_nodes_list()
        # Second call should return identical results (loaded from JSON cache)
        sizes2 = ds.get_num_nodes_list()
        self.assertEqual(sizes1, sizes2)

    def test_num_nodes_list_uses_sidecar_without_full_graph_loads(self):
        cache_dir = self.root / "cache_sizes_sidecar"
        ds = self._make_ds(cache_dir=cache_dir)
        ds._load_graph_for_sample = MagicMock(
            side_effect=AssertionError("should use sidecar node-count path")
        )

        sizes = ds.get_num_nodes_list()

        self.assertEqual(len(sizes), len(ds))
        ds._load_graph_for_sample.assert_not_called()

    def test_no_sizes_file_written_anywhere(self):
        ds = self._make_ds()
        ds.get_num_nodes_list()
        json_files = list(self.root.rglob("*_node_sizes.json"))
        self.assertEqual(len(json_files), 0)
