from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

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

        return AIGGraphRegressionDataset(self.csv_path, **kwargs)

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

    # --- positional encoding ---

    def test_pos_enc_none_by_default(self):
        ds = self._make_ds()
        item = ds[0]
        # PyG raises AttributeError for missing keys; use getattr
        self.assertIsNone(getattr(item, "pos_enc", None))

    def test_pos_enc_level(self):
        ds = self._make_ds(positional_encoding="level")
        item = ds[0]
        pe = getattr(item, "pos_enc", None)
        self.assertIsNotNone(pe)
        self.assertEqual(pe.shape, (item.x.shape[0], 1))

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

    def test_train_batch_shapes(self):
        dm = self._make_dm()
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(batch.y.shape[1], 1)
        self.assertEqual(batch.x.dim(), 2)
        self.assertEqual(batch.edge_index.shape[0], 2)
        self.assertEqual(batch.edge_attr.dim(), 2)

    def test_train_num_samples(self):
        dm = self._make_dm(train_num_samples=10)  # Changed to 10 for cleaner math

        # 10 total samples * 80% train ratio = 8 samples in train_ds
        self.assertEqual(len(dm.train_ds), 8)

        # 10 total samples * 10% val ratio = 1 sample in val_ds
        self.assertEqual(len(dm.val_ds), 1)

        # 10 total samples * 10% test ratio = 1 sample in test_ds
        self.assertEqual(len(dm.test_ds), 1)

        # Ensure the total pool across all splits strictly equals the requested num_samples limit
        self.assertEqual(len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds), 10)

    def test_test_loader(self):
        from data.datamodule import AIGDataModule

        dm = AIGDataModule(self.csv_path, batch_size=4)
        dm.setup(stage="test")
        batch = next(iter(dm.test_dataloader()))
        self.assertEqual(batch.y.shape[1], 1)

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
        from data.datamodule import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 102, 103]
        sampler = BalancedDynamicBatchSampler(
            sizes, batch_size=4, shuffle=False, seed=123
        )

        batches = list(sampler)
        self.assertEqual(len(batches), 2)

        for batch in batches:
            batch_sizes = [sizes[i] for i in batch]
            self.assertEqual(len(batch), 4)
            self.assertGreaterEqual(max(batch_sizes), 100)
            self.assertLessEqual(min(batch_sizes), 4)

        flattened = [idx for batch in batches for idx in batch]
        self.assertEqual(len(flattened), len(sizes))
        self.assertEqual(set(flattened), set(range(len(sizes))))

    def test_pairing_reduces_peak_batch_node_total(self):
        from data.datamodule import BalancedDynamicBatchSampler

        sizes = [1, 2, 3, 4, 100, 101, 102, 103]
        batch_size = 4

        sampler = BalancedDynamicBatchSampler(
            sizes, batch_size=batch_size, shuffle=False, seed=7
        )
        dynamic_batches = list(sampler)
        dynamic_totals = [sum(sizes[i] for i in batch) for batch in dynamic_batches]

        # Baseline: descending contiguous chunks, which tend to pack large graphs.
        descending_indices = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
        baseline_totals = []
        for start in range(0, len(sizes), batch_size):
            chunk = descending_indices[start : start + batch_size]
            baseline_totals.append(sum(sizes[i] for i in chunk))

        self.assertLess(max(dynamic_totals), max(baseline_totals))


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

    def test_sizes_cached_to_disk_when_cache_dir_set(self):
        cache_dir = self.root / "cache_sizes"
        ds = self._make_ds(cache_dir=cache_dir)
        ds.get_num_nodes_list()
        json_files = list(cache_dir.glob("*_node_sizes.json"))
        self.assertEqual(len(json_files), 1)

    def test_second_call_loads_from_cache(self):
        cache_dir = self.root / "cache_sizes2"
        ds = self._make_ds(cache_dir=cache_dir)
        sizes1 = ds.get_num_nodes_list()
        # Second call should return identical results (loaded from JSON cache)
        sizes2 = ds.get_num_nodes_list()
        self.assertEqual(sizes1, sizes2)

    def test_no_cache_file_without_cache_dir(self):
        ds = self._make_ds()
        ds.get_num_nodes_list()
        json_files = list(self.root.rglob("*_node_sizes.json"))
        self.assertEqual(len(json_files), 0)
