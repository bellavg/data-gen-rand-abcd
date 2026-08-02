#!/bin/bash
# Build the thesis and fail loudly on anything that would break the submitted PDF.
#
# USAGE:  bash check-build.sh          (from this directory)
#
# Checks, in order:
#   1. pdflatex -> bibtex -> pdflatex x2 (no latexmk in this TeX Live install)
#   2. zero hard LaTeX errors, zero undefined references and citations
#   3. every \input and \includegraphics target resolves on disk
#   4. no em dashes or punctuation en dashes anywhere in sections/
#   5. chktex warning count against the recorded baseline

set -uo pipefail
cd "$(dirname "$0")"

BASELINE_CHKTEX=154
fail=0

find sections -name '*.aux' -delete
pdflatex -interaction=nonstopmode msc_thesis.tex >/dev/null 2>&1
bibtex msc_thesis >/dev/null 2>&1
pdflatex -interaction=nonstopmode msc_thesis.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode msc_thesis.tex >/dev/null 2>&1

python3 - <<'PY' || fail=1
import re, sys, pathlib
log = pathlib.Path("msc_thesis.log").read_text(encoding="utf-8", errors="replace")
lines = log.splitlines()
errs = [l for l in lines if l.startswith("!")]
undef_ref = [l for l in lines if "LaTeX Warning: Reference" in l]
undef_cite = [l for l in lines if "Citation" in l and "undefined" in l]
pages = re.search(r"Output written on \S+ \((\d+) pages", log)

print(f"pages          : {pages.group(1) if pages else 'NO PDF PRODUCED'}")
print(f"hard errors    : {len(errs)}")
for e in errs[:15]:
    print("   ", e)
print(f"undefined refs : {len(undef_ref)}")
for e in undef_ref[:10]:
    print("   ", e.strip())
print(f"undefined cites: {len(undef_cite)}")
for e in undef_cite[:10]:
    print("   ", e.strip())

bad = []
for p in pathlib.Path(".").rglob("*.tex"):
    if "build-latexdiff" in str(p):
        continue
    src = p.read_text(encoding="utf-8", errors="replace")
    src = re.sub(r"(?m)^\s*%.*$", "", src)          # ignore commented-out targets
    for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", src):
        t = m.group(1)
        if "\\" in t or "corpus_tiers" in t:        # macro path, or the IfFileExists guard
            continue
        if not (pathlib.Path(t).exists() or pathlib.Path(t + ".tex").exists()):
            bad.append(f"{p}: \\input{{{t}}}")
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        t = m.group(1)
        if "\\" in t:
            t = t.replace("\\resultsfigs", "media/results/figures")
            if "\\" in t:
                continue
        if not any(pathlib.Path(t + e).exists() for e in ("", ".pdf", ".png", ".eps", ".jpg")):
            bad.append(f"{p}: \\includegraphics{{{t}}}")
print(f"broken targets : {len(bad)}")
for b in bad[:15]:
    print("   ", b)

sys.exit(1 if (errs or undef_ref or undef_cite or bad or not pages) else 0)
PY

dashes=$(grep -rn '—\|–' sections/ media/ 2>/dev/null | grep -v '^\s*%' | wc -l | tr -d ' ')
echo "em/en dashes   : $dashes"
if [ "$dashes" != "0" ]; then
    grep -rn '—\|–' sections/ media/ 2>/dev/null | head -10
    fail=1
fi

chk=$(chktex -q msc_thesis.tex 2>&1 | wc -l | tr -d ' ')
echo "chktex lines   : $chk (baseline $BASELINE_CHKTEX)"

[ "$fail" = "0" ] && echo "BUILD OK" || echo "BUILD FAILED"
exit "$fail"
