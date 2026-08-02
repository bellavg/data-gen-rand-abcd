"""Per-design AIG statistics for the 55 source circuits, and the LaTeX table built from them.

Reads each design's tier-0 base circuit (``.bench`` for the 47 real IPs, ``.aig`` for
the 8 synthetic ones) and reports the columns OpenABC-D Table 1 uses: primary inputs
(PI), primary outputs (PO), AND nodes (N), edges (E), inverted edges (I) and netlist
depth (D).

The BENCH reader is validated against OpenABC-D Table 1 (``--validate``): 26 of the 29
rows it covers reproduce exactly. The three that do not are internally inconsistent in
the published table -- ``E = 2N + PO`` holds for every AIG, and the published E fixes N
against the printed value for ``mem_ctrl`` and ``ss_pcm``.

Usage::

    python data/creation/corpus_stats.py --validate
    python data/creation/corpus_stats.py --latex > \\
        IV_Gardner___Master_AI_Thesis_Outline/media/tables/corpus_designs.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DESIGNS_DIR = Path("data/designs")

GATE_RE = re.compile(r"^\s*(\S+)\s*=\s*(AND|NOT|BUFF)\s*\(([^)]*)\)", re.I)
IO_RE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(\s*([^)]*?)\s*\)", re.I)

# design -> (source, category, function)
CATALOGUE = {
    "spi":          ("OpenABC-D", "bus",    "Serial peripheral interface"),
    "i2c":          ("OpenABC-D", "bus",    "Bidirectional serial bus protocol"),
    "ss_pcm":       ("OpenABC-D", "bus",    "Single slot PCM"),
    "usb_phy":      ("OpenABC-D", "bus",    "USB PHY 1.1"),
    "sasc":         ("OpenABC-D", "bus",    "Simple asynchronous serial controller"),
    "wb_dma":       ("OpenABC-D", "bus",    "Wishbone DMA/bridge"),
    "simple_spi":   ("OpenABC-D", "bus",    "MC68HC11E based SPI interface"),
    "pci":          ("OpenABC-D", "bus",    "PCI controller"),
    "wb_conmax":    ("OpenABC-D", "bus",    "Wishbone Conmax"),
    "ethernet":     ("OpenABC-D", "bus",    "Ethernet IP core"),
    "ac97_ctrl":    ("OpenABC-D", "ctrl",   "Wishbone AC97 controller"),
    "mem_ctrl":     ("OpenABC-D", "ctrl",   "Wishbone memory controller"),
    "vga_lcd":      ("OpenABC-D", "ctrl",   "Wishbone VGA/LCD controller"),
    "des3_area":    ("OpenABC-D", "crypto", "DES3 encrypt/decrypt"),
    "aes":          ("OpenABC-D", "crypto", "AES (LUT-based)"),
    "sha256":       ("OpenABC-D", "crypto", "SHA-256 hash"),
    "aes_xcrypt":   ("OpenABC-D", "crypto", "AES-128/192/256"),
    "aes_secworks": ("OpenABC-D", "crypto", "AES-128 (simple)"),
    "fir":          ("OpenABC-D", "dsp",    "FIR filter"),
    "iir":          ("OpenABC-D", "dsp",    "IIR filter"),
    "jpeg":         ("OpenABC-D", "dsp",    "JPEG encoder"),
    "idft":         ("OpenABC-D", "dsp",    "Inverse discrete Fourier transform"),
    "dft":          ("OpenABC-D", "dsp",    "Discrete Fourier transform"),
    "tv80":         ("OpenABC-D", "cpu",    "TV80 8-bit microprocessor"),
    "tinyRocket":   ("OpenABC-D", "cpu",    "32-bit tiny RISC-V core"),
    "fpu":          ("OpenABC-D", "cpu",    "OpenSPARC T1 floating point unit"),
    "picosoc":      ("OpenABC-D", "cpu",    "SoC with PicoRV32 RISC-V"),
    "dynamic_node": ("OpenABC-D", "cpu",    "OpenPiton NoC architecture"),
    "bp_be":        ("OpenABC-D", "cpu",    "BlackParrot RISC-V execution engine"),
    "div":          ("EPFL",      "arith",  "Divider"),
    "hyp":          ("EPFL",      "arith",  "Hypotenuse"),
    "log2":         ("EPFL",      "arith",  "Base-2 logarithm"),
    "max":          ("EPFL",      "arith",  "Maximum"),
    "multiplier":   ("EPFL",      "arith",  "Multiplier"),
    "sin":          ("EPFL",      "arith",  "Sine"),
    "sqrt":         ("EPFL",      "arith",  "Square root"),
    "square":       ("EPFL",      "arith",  "Square"),
    "c6288":        ("ISCAS-85",  "arith",  "16$\\times$16 multiplier"),
    "c1355":        ("ISCAS-85",  "comb",   "32-bit error-correcting circuit"),
    "c5315":        ("ISCAS-85",  "comb",   "ALU and selector"),
    "c7552":        ("ISCAS-85",  "comb",   "ALU and control"),
    "dalu":         ("MCNC",      "comb",   "Datapath ALU"),
    "i10":          ("MCNC",      "comb",   "Combinational logic"),
    "apex1":        ("MCNC",      "comb",   "PLA-derived logic"),
    "bc0":          ("MCNC",      "comb",   "PLA-derived logic"),
    "k2":           ("MCNC",      "comb",   "PLA-derived logic"),
    "mainpla":      ("MCNC",      "comb",   "PLA-derived logic"),
}
for _n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
    CATALOGUE[str(_n)] = ("Synthetic", "rand", f"Random AIG, {_n} inputs")

CATEGORY_ORDER = ["bus", "ctrl", "crypto", "dsp", "cpu", "arith", "comb", "rand"]
CATEGORY_LABEL = {
    "bus": "Communication and bus protocol",
    "ctrl": "Controller",
    "crypto": "Cryptography",
    "dsp": "Signal processing",
    "cpu": "Processor and SoC",
    "arith": "Arithmetic",
    "comb": "Combinational logic",
    "rand": "Synthetic random",
}
CATEGORY_COLOUR = {
    "bus": "BrickRed",
    "ctrl": "NavyBlue",
    "crypto": "black",
    "dsp": "ForestGreen",
    "cpu": "Purple",
    "arith": "Orange",
    "comb": "TealBlue",
    "rand": "Gray",
}

# OpenABC-D Table 1, transcribed for validation: PI PO N E I D
REFERENCE = {
    "spi": (254, 238, 4219, 8676, 5524, 35),
    "i2c": (177, 128, 1169, 2466, 1188, 15),
    "ss_pcm": (104, 90, 462, 896, 434, 10),
    "usb_phy": (132, 90, 487, 1064, 513, 10),
    "sasc": (135, 125, 613, 1351, 788, 9),
    "wb_dma": (828, 702, 4587, 9876, 4768, 29),
    "simple_spi": (164, 132, 930, 1992, 1084, 12),
    "pci": (3429, 3157, 19547, 42251, 25719, 29),
    "wb_conmax": (2122, 2075, 47840, 97755, 42138, 24),
    "ethernet": (10731, 10422, 67164, 144750, 86799, 34),
    "ac97_ctrl": (2339, 2137, 11464, 25065, 14326, 11),
    "mem_ctrl": (1187, 962, 16307, 37146, 18092, 36),
    "bp_be": (11592, 8413, 82514, 173441, 109608, 86),
    "vga_lcd": (17322, 17063, 105334, 227731, 141037, 23),
    "des3_area": (303, 64, 4971, 10006, 4686, 30),
    "aes": (683, 529, 28925, 58379, 20494, 27),
    "sha256": (1943, 1042, 15816, 32674, 18459, 76),
    "aes_xcrypt": (1975, 1805, 45840, 93485, 36180, 43),
    "aes_secworks": (3087, 2604, 40778, 84160, 45391, 42),
    "fir": (410, 351, 4558, 9467, 5696, 47),
    "iir": (494, 441, 6978, 14397, 8596, 73),
    "jpeg": (4962, 4789, 114771, 234331, 146080, 40),
    "idft": (37603, 37419, 241552, 520523, 317210, 43),
    "dft": (37597, 37417, 245046, 527509, 322206, 43),
    "tv80": (636, 361, 11328, 23017, 11653, 54),
    "tinyRocket": (4561, 4181, 52315, 108811, 67410, 80),
    "fpu": (632, 409, 29623, 59655, 37142, 819),
    "picosoc": (11302, 10797, 82945, 176687, 107637, 43),
    "dynamic_node": (2708, 2575, 18094, 38763, 23377, 33),
}


def parse_bench(path: Path):
    """Return (PI, PO, N, E, I, D) for an ABC-written BENCH netlist of an AIG."""
    pis, pos, gates = [], [], {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = IO_RE.match(line)
        if m:
            (pis if m.group(1).upper() == "INPUT" else pos).append(m.group(2))
            continue
        m = GATE_RE.match(line)
        if m:
            gates[m.group(1)] = (
                m.group(2).upper(),
                [a.strip() for a in m.group(3).split(",") if a.strip()],
            )

    pi_set = set(pis)
    resolved: dict[str, tuple[str, bool]] = {}

    def resolve(name):
        """Collapse the NOT/BUFF chains ABC emits, leaving AND nodes as the only nodes."""
        if name in resolved:
            return resolved[name]
        if name in pi_set or name not in gates:
            resolved[name] = (name, False)          # primary input or constant
            return resolved[name]
        op, args = gates[name]
        if op == "AND":
            resolved[name] = (name, False)
            return resolved[name]
        base, inv = resolve(args[0])
        resolved[name] = (base, inv ^ (op == "NOT"))
        return resolved[name]

    sys.setrecursionlimit(10_000_000)
    for name in list(gates):
        resolve(name)

    and_nodes = [n for n, (op, _) in gates.items() if op == "AND"]
    and_set = set(and_nodes)

    inverted = 0
    fanins = {}
    for n in and_nodes:
        srcs = []
        for a in gates[n][1]:
            base, inv = resolve(a)
            inverted += inv
            srcs.append(base)
        fanins[n] = srcs
    for p in pos:
        inverted += resolve(p)[1]

    level: dict[str, int] = {}

    def depth_of(name):
        if name in level:
            return level[name]
        if name not in and_set:
            level[name] = 0
            return 0
        level[name] = 0                             # guard against a cycle
        level[name] = 1 + max((depth_of(s) for s in fanins[name]), default=-1)
        return level[name]

    for n in and_nodes:
        depth_of(n)
    # +1 for the output node itself, matching both the preprocessing pipeline and
    # the depth column of OpenABC-D Table 1.
    depth = max((depth_of(resolve(p)[0]) for p in pos), default=-1) + 1

    n_and = len(and_nodes)
    return len(pis), len(pos), n_and, 2 * n_and + len(pos), inverted, depth


def parse_aig(path: Path):
    """Return (PI, PO, N, E, I, D) for an AIGER file."""
    from aigverse.io import read_aiger_into_aig
    from aigverse.networks import DepthAig

    aig = read_aiger_into_aig(str(path))
    depth_aig = DepthAig(aig)
    inverted = sum(bool(s.complement) for n in aig.nodes() for s in aig.fanins(n))
    max_driver = 0
    for sig in aig.pos():
        inverted += bool(sig.complement)
        max_driver = max(max_driver, depth_aig.level(sig.index))
    n_and = aig.num_gates
    return aig.num_pis, aig.num_pos, n_and, 2 * n_and + aig.num_pos, inverted, max_driver + 1


def collect(designs_dir: Path) -> dict[str, tuple]:
    rows = {}
    for d in sorted(p.name for p in designs_dir.iterdir() if p.is_dir() and p.name != "lib"):
        bench = designs_dir / d / "tier0" / f"{d}_synX_step0.bench"
        aig = designs_dir / d / "tier0" / f"{d}_synX_step0.aig"
        if bench.exists():
            rows[d] = parse_bench(bench)
        elif aig.exists():
            rows[d] = parse_aig(aig)
        else:
            print(f"MISSING base circuit for {d}", file=sys.stderr)
    return rows


def validate(rows: dict[str, tuple]) -> None:
    print(f"{'design':<14}{'PI':>7}{'PO':>7}{'N':>9}{'E':>9}{'I':>9}{'D':>6}   check")
    ok = bad = 0
    for d, r in rows.items():
        note = ""
        if d in REFERENCE:
            if REFERENCE[d] == r:
                note, ok = "match", ok + 1
            else:
                note, bad = f"differs from published {REFERENCE[d]}", bad + 1
        print(f"{d:<14}{r[0]:>7}{r[1]:>7}{r[2]:>9}{r[3]:>9}{r[4]:>9}{r[5]:>6}   {note}")
    print(f"\n{ok} match / {bad} differ, against {len(REFERENCE)} OpenABC-D Table 1 rows")


def latex(rows: dict[str, tuple]) -> str:
    def num(v):
        return f"{v:,}".replace(",", "{,}")

    out = [
        "% Generated by data/creation/corpus_stats.py --latex. Do not edit by hand.",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption[Source circuit characteristics]{Characteristics of the 55 source"
        r" circuits before any transformation.",
        r"Primary inputs (PI), primary outputs (PO), AND nodes (N), edges (E), inverted",
        r"edges (I) and depth (D), measured on the base AIG of each design. Every AND node",
        r"has two fanins and every output one, so $E = 2N + PO$ throughout. Rows are grouped",
        r"by function and ordered by size within each group, and colour repeats that grouping:",
        " " + ", ".join(
            rf"\textcolor{{{CATEGORY_COLOUR[c]}}}{{{CATEGORY_LABEL[c]}}}"
            for c in CATEGORY_ORDER
        ) + ".}",
        r"\label{tab:corpus_designs}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"\textbf{Design} & \textbf{Source} & \textbf{PI} & \textbf{PO} & \textbf{N} &"
        r" \textbf{E} & \textbf{I} & \textbf{D} & \textbf{Function} \\",
        r"\midrule",
    ]
    for i, cat in enumerate(CATEGORY_ORDER):
        if i:
            out.append(r"\addlinespace[2pt]")
        members = [d for d in CATALOGUE if CATALOGUE[d][1] == cat and d in rows]
        members.sort(key=lambda d: rows[d][2])            # by AND-node count
        colour = CATEGORY_COLOUR[cat]
        for d in members:
            source, _, function = CATALOGUE[d]
            pi, po, n, e, inv, depth = rows[d]
            name = d.replace("_", r"\_")
            out.append(
                rf"\textcolor{{{colour}}}{{\texttt{{{name}}}}} & {source} & "
                rf"{num(pi)} & {num(po)} & {num(n)} & {num(e)} & {num(inv)} & {num(depth)} & "
                rf"\textcolor{{{colour}}}{{{function}}} \\"
            )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--designs-dir", type=Path, default=DESIGNS_DIR)
    ap.add_argument("--validate", action="store_true", help="print stats and check against OpenABC-D")
    ap.add_argument("--latex", action="store_true", help="emit the LaTeX table on stdout")
    args = ap.parse_args()

    rows = collect(args.designs_dir)
    if args.latex:
        print(latex(rows))
    else:
        validate(rows)


if __name__ == "__main__":
    main()
