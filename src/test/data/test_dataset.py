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

    def test_getitem_raises_value_error_if_edge_attr_missing(self):
        """Ensure __getitem__ raises ValueError if edge_attr is None."""
        bad_pt = self.root / "bad_graph.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.edge_attr = None  # Corrupt the data
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        with self.assertRaisesRegex(ValueError, "edge_attr=None"):
            AIGGraphRegressionDataset(bad_csv)

    def test_getitem_raises_value_error_if_edge_attr_1d(self):
        """Ensure __getitem__ raises ValueError if edge_attr is not 2D."""
        bad_pt = self.root / "bad_graph_1d.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.edge_attr = torch.tensor([1.0, 0.0])  # 1D instead of 2D
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad_1d.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        with self.assertRaisesRegex(ValueError, "edge_attr must be 2D"):
            AIGGraphRegressionDataset(bad_csv)

    # --- dataset initialization verification ---

    def test_verify_first_sample_raises_assertion_error_on_bad_x(self):
        """Ensure initialization fails early if the first graph's x attribute is not 2D."""
        bad_pt = self.root / "bad_x_graph.pt"
        valid_data = torch.load(self.pt_paths[0], weights_only=False)
        valid_data.x = torch.rand(10)  # 1D instead of 2D
        torch.save(valid_data, bad_pt)

        bad_csv = self.root / "bad_x.csv"
        _write_csv(
            bad_csv, [{"unoptimized_graph_path": str(bad_pt), "optimizability": "0.5"}]
        )

        from data.dataset import AIGGraphRegressionDataset

        with self.assertRaisesRegex(AssertionError, "x should be 2D"):
            AIGGraphRegressionDataset(bad_csv)

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
        pt_paths = _make_graph_pts(self.root / "graphs", 30)
        self.csv_path = self.root / "orchestrate.csv"
        _write_csv(self.csv_path, _make_rows(pt_paths))

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
        dm = self._make_dm(train_num_samples=8)
        self.assertEqual(len(dm.train_ds), 8)

    def test_test_loader(self):
        from data.datamodule import AIGDataModule

        dm = AIGDataModule(self.csv_path, batch_size=4)
        dm.setup(stage="test")
        batch = next(iter(dm.test_dataloader()))
        self.assertEqual(batch.y.shape[1], 1)

    def test_datamodule_split_sizes_sum_to_total(self):
        dm = self._make_dm()
        self.assertEqual(len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds), 30)
