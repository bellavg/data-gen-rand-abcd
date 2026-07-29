from __future__ import annotations

import unittest
from pathlib import Path

import pytest

from data.dataset_utils import (
    clean_str,
    graph_input_path_from_csv_row,
    infer_recipe_id,
    normalize_algorithm,
    parse_float,
    parse_int,
)


# ---------------------------------------------------------------------------
# clean_str
# ---------------------------------------------------------------------------


class TestCleanStr(unittest.TestCase):
    def test_none_returns_empty_string(self):
        self.assertEqual(clean_str(None), "")

    def test_string_is_stripped(self):
        self.assertEqual(clean_str("  hello  "), "hello")

    def test_plain_string(self):
        self.assertEqual(clean_str("word"), "word")

    def test_empty_string_returns_empty(self):
        self.assertEqual(clean_str(""), "")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(clean_str("   "), "")


# ---------------------------------------------------------------------------
# normalize_algorithm
# ---------------------------------------------------------------------------


class TestNormalizeAlgorithm(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(normalize_algorithm(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(normalize_algorithm(""))

    def test_all_returns_none(self):
        self.assertIsNone(normalize_algorithm("all"))

    def test_all_case_insensitive(self):
        self.assertIsNone(normalize_algorithm("ALL"))
        self.assertIsNone(normalize_algorithm("All"))

    def test_valid_algorithm_orchestrate(self):
        self.assertEqual(normalize_algorithm("Orchestrate"), "Orchestrate")

    def test_valid_algorithm_deepsyn(self):
        self.assertEqual(normalize_algorithm("Deepsyn"), "Deepsyn")

    def test_valid_algorithm_syn4(self):
        self.assertEqual(normalize_algorithm("Syn4"), "Syn4")

    def test_valid_algorithm_c2rs(self):
        self.assertEqual(normalize_algorithm("C2RS"), "C2RS")

    def test_invalid_algorithm_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalize_algorithm("InvalidAlgo")

    def test_whitespace_stripped_before_check(self):
        self.assertEqual(normalize_algorithm("  Orchestrate  "), "Orchestrate")


# ---------------------------------------------------------------------------
# parse_int
# ---------------------------------------------------------------------------


class TestParseInt(unittest.TestCase):
    def test_plain_integer_string(self):
        self.assertEqual(parse_int("5"), 5)

    def test_float_string_truncates(self):
        self.assertEqual(parse_int("3.9"), 3)

    def test_empty_string_returns_default(self):
        self.assertEqual(parse_int(""), 0)
        self.assertEqual(parse_int("", default=7), 7)

    def test_none_returns_default(self):
        self.assertEqual(parse_int(None), 0)

    def test_invalid_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_int("abc")

    def test_whitespace_stripped(self):
        self.assertEqual(parse_int("  4  "), 4)

    def test_negative_value(self):
        self.assertEqual(parse_int("-2"), -2)


# ---------------------------------------------------------------------------
# parse_float
# ---------------------------------------------------------------------------


class TestParseFloat(unittest.TestCase):
    def test_plain_float_string(self):
        self.assertAlmostEqual(parse_float("3.14"), 3.14, places=5)

    def test_integer_string(self):
        self.assertEqual(parse_float("5"), 5.0)

    def test_empty_string_returns_default(self):
        self.assertEqual(parse_float(""), 0.0)
        self.assertEqual(parse_float("", default=1.5), 1.5)

    def test_none_returns_default(self):
        self.assertEqual(parse_float(None), 0.0)

    def test_invalid_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_float("abc")

    def test_whitespace_stripped(self):
        self.assertAlmostEqual(parse_float("  2.7  "), 2.7, places=5)

    def test_negative_value(self):
        self.assertAlmostEqual(parse_float("-1.5"), -1.5, places=5)


# ---------------------------------------------------------------------------
# graph_input_path_from_csv_row
# ---------------------------------------------------------------------------


class TestGraphInputPathFromCsvRow(unittest.TestCase):
    def setUp(self):
        self.graph_root = Path("/data/output")

    def _row(self, **kwargs) -> dict:
        """Return a minimal CSV row dict with reasonable defaults."""
        defaults = {
            "file_path": "",
            "design": "",
            "tier_id": "0",
        }
        defaults.update(kwargs)
        return defaults

    # --- missing file_path ---

    def test_missing_file_path_raises(self):
        row = self._row(file_path="", tier_id="0", design="adder")
        with self.assertRaises(ValueError):
            graph_input_path_from_csv_row(self.graph_root, row)

    # --- tier 0 ---

    def test_tier0_with_explicit_design(self):
        row = self._row(
            file_path="adder_syn1_step3.aig",
            tier_id="0",
            design="adder",
        )
        result = graph_input_path_from_csv_row(self.graph_root, row)
        expected = self.graph_root / "graphs" / "tier0" / "adder" / "adder_syn1_step3.pt"
        self.assertEqual(result, expected)

    def test_tier0_infers_design_from_stem(self):
        """When design is blank, it should be inferred from the filename stem."""
        row = self._row(
            file_path="multiplier_synX_step0.aig",
            tier_id="0",
            design="",
        )
        result = graph_input_path_from_csv_row(self.graph_root, row)
        expected = (
            self.graph_root / "graphs" / "tier0" / "multiplier" / "multiplier_synX_step0.pt"
        )
        self.assertEqual(result, expected)

    def test_tier0_unrecognised_stem_no_design_raises(self):
        row = self._row(
            file_path="garbled_file_name.aig",
            tier_id="0",
            design="",
        )
        with self.assertRaises(ValueError):
            graph_input_path_from_csv_row(self.graph_root, row)

    # --- tier 1 ---

    def test_tier1_resolves_to_tier0_graph(self):
        """Tier-1 annotation is trained against the *tier-0* unoptimised graph."""
        row = self._row(
            file_path="adder_Orchestrate_tier1_syn2_step4.aig",
            tier_id="1",
        )
        result = graph_input_path_from_csv_row(self.graph_root, row)
        expected = (
            self.graph_root / "graphs" / "tier0" / "adder" / "adder_syn2_step4.pt"
        )
        self.assertEqual(result, expected)

    def test_tier1_bad_stem_raises(self):
        row = self._row(file_path="bad_name.aig", tier_id="1")
        with self.assertRaises(ValueError):
            graph_input_path_from_csv_row(self.graph_root, row)

    # --- tier 2 ---

    def test_tier2_resolves_to_tier1_graph(self):
        """Tier-2 annotation uses the tier-1 graph of the *source* algorithm."""
        row = self._row(
            file_path="adder_Orchestrate_Deepsyn_tier2_syn1_step0.aig",
            tier_id="2",
        )
        result = graph_input_path_from_csv_row(self.graph_root, row)
        expected = (
            self.graph_root
            / "graphs"
            / "tier1"
            / "Orchestrate"
            / "adder"
            / "adder_Orchestrate_tier1_syn1_step0.pt"
        )
        self.assertEqual(result, expected)

    def test_tier2_bad_stem_raises(self):
        row = self._row(file_path="bad_tier2.aig", tier_id="2")
        with self.assertRaises(ValueError):
            graph_input_path_from_csv_row(self.graph_root, row)

    # --- fallback (tier > 2) ---

    def test_tier3_fallback_uses_design_column(self):
        row = self._row(
            file_path="adder_some_step.aig",
            tier_id="3",
            design="adder",
        )
        result = graph_input_path_from_csv_row(self.graph_root, row)
        expected = (
            self.graph_root / "graphs" / "tier3" / "adder" / "adder_some_step.pt"
        )
        self.assertEqual(result, expected)

    def test_tier3_fallback_missing_design_raises(self):
        row = self._row(file_path="adder_some_step.aig", tier_id="3", design="")
        with self.assertRaises(ValueError):
            graph_input_path_from_csv_row(self.graph_root, row)


@pytest.mark.parametrize(
    "algo",
    ["Orchestrate", "Deepsyn", "Syn4", "C2RS"],
)
def test_normalize_algorithm_all_valid(algo: str) -> None:
    assert normalize_algorithm(algo) == algo


# ---------------------------------------------------------------------------
# infer_recipe_id
# ---------------------------------------------------------------------------


class TestInferRecipeId(unittest.TestCase):
    def test_tier0_stem(self):
        self.assertEqual(infer_recipe_id("adder_syn1_step3"), "syn1")

    def test_tier0_stem_with_x_variant(self):
        self.assertEqual(infer_recipe_id("multiplier_synX_step0"), "synX")

    def test_tier1_stem_reproduces_tier0_recipe(self):
        self.assertEqual(
            infer_recipe_id("adder_Orchestrate_tier1_syn2_step4"), "syn2"
        )

    def test_tier2_stem_reproduces_same_recipe(self):
        self.assertEqual(
            infer_recipe_id("adder_Orchestrate_Deepsyn_tier2_syn1_step0"), "syn1"
        )

    def test_tier0_and_tier1_same_recipe_agree(self):
        """The whole point of the recipe key: a tier0 variant and its tier1
        descendant (same synX, different design-irrelevant step) must produce
        an identical recipe ID, regardless of design."""
        tier0 = infer_recipe_id("adder_syn2_step4")
        tier1 = infer_recipe_id("adder_Orchestrate_tier1_syn2_step4")
        self.assertEqual(tier0, tier1)

    def test_recipe_id_is_design_independent(self):
        """The same recipe script is applied to every design, so two
        different designs on the same recipe ID must produce the same key."""
        self.assertEqual(
            infer_recipe_id("adder_syn3_step7"), infer_recipe_id("aes_syn3_step7")
        )

    def test_unrecognized_stem_returns_none(self):
        self.assertIsNone(infer_recipe_id("garbled_file_name"))
