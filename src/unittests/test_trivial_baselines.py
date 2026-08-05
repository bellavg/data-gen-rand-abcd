import argparse
import json

import numpy as np
import pandas as pd
import pytest

import trivial_baselines as tb


def _rows(paths, targets, nodes, levels=None, scripts=None):
    """A frame shaped like load_rows' output.

    The type counts are split so the totals hold: one primary input, one primary
    output, and the rest AND gates, so num_nodes = 1 + PI + AND + PO and
    num_edges = 2 * AND + PO, the identity the generation pipeline writes.
    """
    nodes = np.asarray(nodes, dtype=float)
    num_and = nodes - 3
    return pd.DataFrame(
        {
            "graph_path": list(paths),
            "target": np.asarray(targets, dtype=float),
            "num_pi": np.ones(len(nodes)),
            "num_and": num_and,
            "num_po": np.ones(len(nodes)),
            "num_nodes": nodes,
            "num_edges": 2 * num_and + 1,
            "level": np.asarray(levels if levels is not None else nodes / 10, dtype=float),
            "source_script": list(scripts) if scripts is not None else ["tier0 base"] * len(nodes),
        }
    )


def _write_csv(path, paths, targets, nodes, levels=None):
    nodes = list(nodes)
    pd.DataFrame(
        {
            "unoptimized_graph_path": list(paths),
            "optimizability": targets,
            "pre_nodes": nodes,
            "pre_num_PI": [1] * len(nodes),
            "pre_num_PO": [1] * len(nodes),
            "edges": [2 * n + 1 for n in nodes],
            "pre_depth": list(levels) if levels is not None else [n // 10 for n in nodes],
        }
    ).to_csv(path, index=False)


def test_constants_come_from_train_not_val_or_test():
    """The property the whole script exists to guarantee.

    Each split gets a deliberately different mean, so a predictor fitted on the
    wrong one is off by a wide margin rather than coincidentally close.
    """
    # Skewed on purpose: mean 0.25, median 0.10. A symmetric fixture would let a
    # median that secretly computed the mean pass this test.
    train = _rows(["a", "b", "c", "d"], [0.0, 0.1, 0.1, 0.8], [100, 200, 300, 400])
    predictors = tb.fit_predictors(train)

    test = _rows(["y", "z"], [0.9, 0.9], [100, 200])  # mean 0.9, nothing like train
    assert predictors["mean"].predict(test) == pytest.approx([0.25, 0.25])
    assert predictors["median"].predict(test) == pytest.approx([0.10, 0.10])


def test_constant_predictor_cannot_score_above_zero_r2():
    """R^2 is taken against the test set's own mean, so any other constant loses.

    This is what makes the mean baseline the zero point rather than one more
    contender, and it is the claim the methodology chapter makes analytically.
    """
    y_true = np.array([0.0, 0.1, 0.2, 0.7])
    for constant in (0.0, 0.2, 0.5, y_true.mean() + 1e-3):
        scored = tb.score(y_true, np.full_like(y_true, constant))
        assert scored["r2"] <= 0.0

    at_test_mean = tb.score(y_true, np.full_like(y_true, y_true.mean()))
    assert at_test_mean["r2"] == pytest.approx(0.0)


def test_spearman_is_nan_for_a_constant_not_zero():
    """A constant has no ranking, which is not the same as ranking at chance."""
    y_true = np.array([0.0, 0.1, 0.2, 0.7])
    assert np.isnan(tb.score(y_true, np.full_like(y_true, 0.2))["spearman"])
    assert not np.isnan(tb.score(y_true, y_true)["spearman"])


def test_zero_predictor_is_the_unfitted_removes_nothing_hypothesis():
    """It must stay at zero wherever the train median lands, so it is fitted on nothing."""
    train = _rows(["a", "b", "c"], [0.3, 0.4, 0.5], [100, 200, 300])
    zero = tb.fit_predictors(train)["zero"]

    assert zero.fitted_on == "none"
    assert zero.predict(_rows(["y", "z"], [0.9, 0.9], [10, 20])) == pytest.approx([0.0, 0.0])


def test_size_predictions_stay_inside_the_target_range():
    """Optimizability is bounded by its definition; a linear fit is not."""
    train = _rows(
        [f"g{i}" for i in range(6)],
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [10, 20, 40, 80, 160, 320],
    )
    predictors = tb.fit_predictors(train)
    far = _rows(["big", "small"], [0.0, 0.0], [10**9, 5])
    for name in ("size", "counts"):
        predictions = predictors[name].predict(far)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)


def test_counts_reads_composition_that_size_cannot():
    """The point of the count vector: it sees what the two size totals discard.

    Every graph here has the same node and edge count, so the size design is
    rank deficient and can do no better than the train mean, while the type
    counts still vary and carry the target.
    """
    # 2 * AND + PO = 400 and 1 + PI + AND + PO = 301 for every row.
    composition = [(0, 100, 200), (10, 110, 180), (20, 120, 160), (50, 150, 100)]
    targets = [0.0, 0.2, 0.4, 0.6]
    train = pd.DataFrame(
        {
            "graph_path": [f"g{i}" for i in range(4)],
            "target": np.array(targets, dtype=float),
            "num_pi": [c[0] for c in composition],
            "num_and": [c[1] for c in composition],
            "num_po": [c[2] for c in composition],
            "num_nodes": [301.0] * 4,
            "num_edges": [400.0] * 4,
            "level": [5.0] * 4,
            "source_script": ["tier0 base"] * 4,
        }
    )
    predictors = tb.fit_predictors(train)

    size = predictors["size"].predict(train)
    counts = predictors["counts"].predict(train)
    assert size == pytest.approx([np.mean(targets)] * 4)
    assert counts == pytest.approx(targets, abs=1e-6)


def test_size_level_lookup_returns_the_train_cell_mean():
    """A lookup table predicts the cell it lands in, not a global constant."""
    small = _rows([f"s{i}" for i in range(10)], [0.1] * 10, range(10, 20), levels=[2] * 10)
    large = _rows([f"l{i}" for i in range(10)], [0.5] * 10, range(10_000, 10_010), levels=[90] * 10)
    train = pd.concat([small, large], ignore_index=True)

    lookup = tb.fit_predictors(train)["size_level"]
    probe = _rows(["small", "large"], [0.0, 0.0], [15, 10_005], levels=[2, 90])
    assert lookup.predict(probe) == pytest.approx([0.1, 0.5])


def test_a_group_unseen_in_train_falls_back_to_the_global_train_mean():
    """Test designs bring source scripts train may not hold; that must not crash."""
    train = _rows(
        ["a", "b"], [0.0, 0.4], [100, 200], scripts=["Syn4", "Syn4"]
    )  # global train mean 0.2
    oracle = tb.fit_predictors(train)["source_script_mean"]

    probe = _rows(["y", "z"], [0.9, 0.9], [100, 200], scripts=["Syn4", "C2RS"])
    assert oracle.predict(probe) == pytest.approx([0.2, 0.2])


def test_the_source_script_oracle_groups_by_the_script_in_the_path():
    """It reads provenance the encoder never sees, which is why it is an oracle."""
    train = _rows(
        [f"g{i}" for i in range(4)],
        [0.00, 0.02, 0.40, 0.60],
        [100, 200, 300, 400],
        scripts=["C2RS", "C2RS", "Syn4", "Syn4"],
    )
    oracle = tb.fit_predictors(train)["source_script_mean"]

    assert oracle.role == "oracle"
    probe = _rows(["y", "z"], [0.0, 0.0], [150, 350], scripts=["C2RS", "Syn4"])
    assert oracle.predict(probe) == pytest.approx([0.01, 0.50])


def test_predictions_stay_aligned_with_a_split_sliced_frame():
    """main() scores a boolean slice, so the test frame's index does not start at zero.

    A group-mean predictor maps through a pandas Series, which is where a
    misalignment would silently attach the wrong prediction to a graph.
    Asserting only the length and finiteness would pass under a reversed or
    shifted prediction vector, so this pins the per-row values.
    """
    # Train holds both scripts with sharply different means, so the oracle's two
    # test predictions differ and a swap between them is visible.
    rows = _rows(
        list("abcdef"),
        [0.0, 0.02, 0.40, 0.60, 0.0, 0.0],
        [10, 20, 30, 40, 15, 35],
        levels=[1, 1, 9, 9, 1, 9],
        scripts=["C2RS", "C2RS", "Syn4", "Syn4", "C2RS", "Syn4"],
    ).assign(split=["train"] * 4 + ["test"] * 2)
    train, test = rows[rows["split"] == "train"], rows[rows["split"] == "test"]
    assert list(test.index) == [4, 5]

    # test row 4 is C2RS (train mean 0.01), row 5 is Syn4 (train mean 0.50)
    oracle = tb.fit_predictors(train)["source_script_mean"]
    assert oracle.predict(test) == pytest.approx([0.01, 0.50])

    # and the same frame, reversed, must produce the reversed predictions
    assert oracle.predict(test.iloc[::-1]) == pytest.approx([0.50, 0.01])


def test_missing_splits_file_refuses_to_invent_one(tmp_path):
    with pytest.raises(SystemExit, match="no splits file"):
        tb.load_splits(tmp_path / "absent.json")


def test_empty_split_is_rejected(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": []}))
    with pytest.raises(SystemExit, match="empty"):
        tb.load_splits(path)


def test_rows_outside_every_split_are_dropped(tmp_path):
    """The tuning holdout is in the CSV but in no split, and must not be scored."""
    rows = _rows(["a", "b", "holdout"], [0.1, 0.2, 0.9], [10, 20, 30])
    splits = {"train": {"a"}, "val": set(), "test": {"b"}}
    assigned = tb.assign_splits(rows, splits)
    assert set(assigned["graph_path"]) == {"a", "b"}


def test_load_rows_reads_target_counts_and_level_without_loading_a_graph(tmp_path):
    csv = tmp_path / "Orchestrate.csv"
    _write_csv(csv, ["/gpfs/scratch1/shared/graphs/tier1/Syn4/aes/g0.pt"], [0.25], [98], levels=[7])

    rows = tb.load_rows([str(csv)])
    assert len(rows) == 1
    # scratch path rewritten so it joins against the splits JSON
    assert rows["graph_path"].iloc[0] == "/scratch-shared/graphs/tier1/Syn4/aes/g0.pt"
    assert rows["target"].iloc[0] == pytest.approx(0.25)
    # 1 + pre_num_PI + pre_nodes + pre_num_PO
    assert rows["num_nodes"].iloc[0] == pytest.approx(101)
    assert rows["num_edges"].iloc[0] == pytest.approx(197)
    assert rows["num_and"].iloc[0] == pytest.approx(98)
    assert rows["level"].iloc[0] == pytest.approx(7)
    # the upstream script, recovered from the path rather than a column
    assert rows["source_script"].iloc[0] == "Syn4"


def test_load_rows_labels_a_tier_zero_input_as_having_no_upstream_script(tmp_path):
    csv = tmp_path / "Orchestrate.csv"
    _write_csv(csv, ["/scratch-shared/graphs/tier0/aes/g0.pt"], [0.25], [98])
    assert tb.load_rows([str(csv)])["source_script"].iloc[0] == "tier0 base"


def test_load_rows_rejects_a_csv_missing_a_required_column(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"unoptimized_graph_path": ["g"], "optimizability": [0.1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing column"):
        tb.load_rows([str(csv)])


def test_oracles_are_written_to_a_file_of_their_own(tmp_path):
    """Structural separation, not a label: a table built from --out holds no oracle."""
    paths = [f"/scratch-shared/graphs/tier1/Syn4/aes/g{i}.pt" for i in range(6)]
    csv = tmp_path / "Orchestrate.csv"
    _write_csv(csv, paths, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [10, 20, 30, 40, 50, 60])

    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": paths[:3], "val": paths[3:4], "test": paths[4:]}))

    out = tmp_path / "rq1.csv"
    tb.main(
        argparse.Namespace(
            csv_paths=[str(csv)], splits_path=str(splits), out=str(out), n_resamples=10, seed=0,
            algorithm="Orchestrate", split_by="design", wandb=False,
        )
    )

    fair = pd.read_csv(out)
    oracles = pd.read_csv(out.with_name("rq1_oracles.csv"))
    assert set(fair["role"]) == {"fair"}
    assert set(oracles["role"]) == {"oracle"}
    assert "source_script_mean" not in set(fair["predictor"])
    assert "source_script_mean" in set(oracles["predictor"])


def test_a_missing_level_does_not_shrink_the_corpus_or_poison_the_bins(tmp_path):
    """Only the lookup uses the level, so a patchy pre_depth must not drop rows."""
    csv = tmp_path / "Orchestrate.csv"
    pd.DataFrame(
        {
            "unoptimized_graph_path": [f"/scratch-shared/graphs/tier0/aes/g{i}.pt" for i in range(4)],
            "optimizability": [0.0, 0.1, 0.2, 0.3],
            "pre_nodes": [10, 20, 30, 40],
            "pre_num_PI": [1] * 4,
            "pre_num_PO": [1] * 4,
            "edges": [21, 41, 61, 81],
            "pre_depth": [3, None, 7, 9],
        }
    ).to_csv(csv, index=False)

    rows = tb.load_rows([str(csv)])
    assert len(rows) == 4
    assert np.isnan(rows["level"].iloc[1])

    # the surviving levels still produce usable bin edges, and every row scores
    predictions = tb.fit_predictors(rows)["size_level"].predict(rows)
    assert len(predictions) == 4
    assert np.all(np.isfinite(predictions))


def test_edge_count_comes_from_pre_columns_not_the_csv_edges_column(tmp_path):
    """The CSV's `edges` is 2*post_nodes + post_num_PO, so reading it leaks the target.

    A tier-1 row is written with edges=0 and the OPTIMIZED graph's node and PO
    counts, and the master-CSV pass then imputes edges from those. Here the CSV
    carries a deliberately wrong `edges` matching a heavily optimized graph; the
    loaded edge count must ignore it and follow 2*pre_nodes + pre_num_PO.
    """
    csv = tmp_path / "Orchestrate.csv"
    pd.DataFrame(
        {
            "unoptimized_graph_path": ["/scratch-shared/graphs/tier0/aes/g0.pt"],
            "optimizability": [0.5],
            "pre_nodes": [1000],
            "pre_num_PI": [4],
            "pre_num_PO": [6],
            "pre_depth": [12],
            "post_nodes": [500],
            "post_num_PO": [6],
            "edges": [1006],  # 2*post_nodes + post_num_PO: the leak
        }
    ).to_csv(csv, index=False)

    rows = tb.load_rows([str(csv)])
    assert rows["num_edges"].iloc[0] == pytest.approx(2006)  # 2*1000 + 6


def test_a_path_in_both_train_and_test_is_rejected(tmp_path):
    """Checked on the sets: assign_splits flattens them last-wins, hiding an overlap."""
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["b", "d"]}))
    with pytest.raises(SystemExit, match="both train and test"):
        tb.load_splits(path)


def test_a_splits_file_that_joins_to_nothing_fails_loudly(tmp_path):
    """Path drift between the CSV and the splits file must not die inside numpy."""
    csv = tmp_path / "Orchestrate.csv"
    _write_csv(csv, ["/scratch-shared/graphs/tier0/aes/g0.pt"], [0.1], [10])
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["/elsewhere/x.pt"], "val": ["/elsewhere/y.pt"],
                                  "test": ["/elsewhere/z.pt"]}))
    with pytest.raises(SystemExit, match="no CSV row joined to the train split"):
        tb.main(
            argparse.Namespace(
                csv_paths=[str(csv)], splits_path=str(splits), out=str(tmp_path / "o.csv"),
                n_resamples=5, seed=0, algorithm="Orchestrate", split_by="design", wandb=False,
            )
        )


def test_the_wandb_run_name_follows_the_project_convention():
    assert tb.wandb_run_name_for("Orchestrate", "design") == "baseline_trivial_Orchestrate"
    assert (
        tb.wandb_run_name_for("Orchestrate", "recipe") == "baseline_trivial_Orchestrate_recipe"
    )
