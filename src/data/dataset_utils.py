from __future__ import annotations

import re
from pathlib import Path

from src.constants import VALID_ALGORITHMS


_ALGO_ALT = "|".join(sorted(VALID_ALGORITHMS))
_TIER0_STEM_RE = re.compile(r"^(?P<design>.+?)_syn[0-9X]+_step\d+$")
_TIER1_STEM_RE = re.compile(
	rf"^(?P<design>.+?)_(?P<algorithm>{_ALGO_ALT})_tier1_(?P<suffix>.+)$"
)
_TIER2_STEM_RE = re.compile(
	rf"^(?P<design>.+?)_(?P<src_algorithm>{_ALGO_ALT})_(?P<dst_algorithm>{_ALGO_ALT})_tier2_(?P<suffix>.+)$"
)


def clean_str(value: str | None) -> str:
	if value is None:
		return ""
	return str(value).strip()


def normalize_algorithm(value: str | None) -> str | None:
	text = clean_str(value)
	if text == "" or text.lower() == "all":
		return None
	if text not in VALID_ALGORITHMS:
		raise ValueError(
			f"algorithm must be one of {sorted(VALID_ALGORITHMS)} or None/'all'. "
			f"Got: {text}"
		)
	return text


def parse_int(value: str | None, default: int = 0) -> int:
	text = clean_str(value)
	if text == "":
		return default
	return int(float(text))


def parse_float(value: str | None, default: float = 0.0) -> float:
	text = clean_str(value)
	if text == "":
		return default
	return float(text)


def graph_input_path_from_csv_row(graph_root: Path, row: dict[str, str]) -> Path:
	"""
	Resolve the *input* graph artifact path for a CSV row.

	Training dependency mapping:
	- tier0 label uses tier0 graph
	- tier1 label uses tier0 graph input
	- tier2 label uses tier1 graph input from source algorithm
	"""
	file_path = clean_str(row.get("file_path"))
	if file_path == "":
		raise ValueError("CSV row is missing file_path")

	tier_id = parse_int(row.get("tier_id"), default=0)
	stem = Path(file_path).stem

	if tier_id == 0:
		design = clean_str(row.get("design"))
		if design == "":
			match = _TIER0_STEM_RE.match(stem)
			if match is None:
				raise ValueError(
					"Could not infer tier0 design from stem and design is missing: "
					f"{stem}"
				)
			design = match.group("design")
		return graph_root / "graphs" / "tier0" / design / f"{stem}.pt"

	if tier_id == 1:
		match = _TIER1_STEM_RE.match(stem)
		if match is None:
			raise ValueError(
				"Tier1 row does not match expected stem format "
				"{design}_{algorithm}_tier1_{suffix}: "
				f"{stem}"
			)
		design = match.group("design")
		suffix = match.group("suffix")
		tier0_stem = f"{design}_{suffix}"
		return graph_root / "graphs" / "tier0" / design / f"{tier0_stem}.pt"

	if tier_id == 2:
		match = _TIER2_STEM_RE.match(stem)
		if match is None:
			raise ValueError(
				"Tier2 row does not match expected stem format "
				"{design}_{src_algo}_{dst_algo}_tier2_{suffix}: "
				f"{stem}"
			)
		design = match.group("design")
		src_algorithm = match.group("src_algorithm")
		suffix = match.group("suffix")
		tier1_stem = f"{design}_{src_algorithm}_tier1_{suffix}"
		return (
			graph_root
			/ "graphs"
			/ "tier1"
			/ src_algorithm
			/ design
			/ f"{tier1_stem}.pt"
		)

	# Fallback for tiers beyond tier2 keeps prior direct-tier lookup.
	design = clean_str(row.get("design"))
	if design == "":
		raise ValueError(f"CSV row is missing design for tier{tier_id}: file_path={file_path}")
	return graph_root / "graphs" / f"tier{tier_id}" / design / f"{stem}.pt"


__all__ = [
	"clean_str",
	"normalize_algorithm",
	"parse_int",
	"parse_float",
	"graph_input_path_from_csv_row",
]
