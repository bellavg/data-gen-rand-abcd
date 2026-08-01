#!/bin/bash
# Build a marked-up PDF showing what changed in the thesis between `main` and the
# current working tree: additions underlined blue, deletions struck through red.
#
# ONE-TIME SETUP (needs your password; TeX Live here is the "basic" scheme, which
# ships neither of these):
#     sudo tlmgr update --self
#     sudo tlmgr install latexdiff changepage
#
# changepage is required by mscaithesis.cls itself -- without it the document does
# not compile at all locally, latexdiff or no latexdiff.
#
# USAGE:  bash make-latexdiff.sh          (from this directory)
# OUTPUT: build-latexdiff/diff.pdf
#
# NOTE ON WHAT YOU WILL SEE: latexdiff compares TYPESET content, so the ~900 lines
# of "%" guidance comments added to the outline are invisible here by design. That
# makes this the right view for reviewing prose, and the wrong view for reviewing
# the TODO/BLOCKED notes -- use tex-diff.html for those.

set -euo pipefail

command -v latexdiff >/dev/null || { echo "latexdiff not installed -- see header"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
REL="$(git -C "$HERE" rev-parse --show-prefix)"
OUT="$HERE/build-latexdiff"
BASE="${1:-main}"

rm -rf "$OUT"; mkdir -p "$OUT/old" "$OUT/new"

# Left side: the committed baseline. Right side: the working tree as it stands.
git -C "$REPO" archive "$BASE" "$REL" | tar -x -C "$OUT/old" --strip-components="$(awk -F/ '{print NF-1}' <<<"$REL")"
cp -R "$HERE"/* "$OUT/new/" 2>/dev/null || true
rm -rf "$OUT/new/build-latexdiff"

# --flatten resolves \include, so the whole thesis is diffed as one document.
latexdiff --flatten --type=UNDERLINE \
    "$OUT/old/msc_thesis.tex" "$OUT/new/msc_thesis.tex" > "$OUT/diff.tex"

cd "$OUT"
cp -R "$HERE/bibliographies" . 2>/dev/null || true
cp "$HERE/mscaithesis.cls" . 2>/dev/null || true
pdflatex -interaction=nonstopmode diff.tex >/dev/null 2>&1 || true
bibtex diff                              >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode diff.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode diff.tex >/dev/null 2>&1 || true

[ -f diff.pdf ] && echo "OK -> $OUT/diff.pdf" || {
    echo "No PDF produced. First real error:"; grep -m1 -A3 '^!' diff.log; exit 1; }
