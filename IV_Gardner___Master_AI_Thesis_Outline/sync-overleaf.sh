#!/bin/bash
# Publish the thesis source to the Overleaf mirror that the supervisor reads,
# and back the branch up to origin on the way.
#
# USAGE:  bash sync-overleaf.sh          (from anywhere in this repo)
#
# Publishes the COMMITTED state of whatever branch is checked out. Uncommitted
# edits stay local.
#
# The current branch is pushed to origin BEFORE the mirror is touched. These are
# two unrelated repositories, not a submodule, so one without the other leaves
# work in a single place: the mirror carries only the allowlisted files below
# and flattens history, and origin is the only copy of everything else. If the
# origin push fails, nothing is published.
#
# PRIVATE BY DEFAULT. The PUBLISH list below is an allowlist: only those paths
# ever leave this repo. Planning notes, handoffs, CLAUDE.md, per-section notes
# and build output are excluded because they are not on the list, so a new
# notes file anywhere under this directory is private automatically, with
# nothing to remember and nothing to add. Keep it that way: to publish
# something new, add a file type or a directory, never work around the list.
#
# One way. Anything edited on the Overleaf side is overwritten, so do the
# writing here, not there.
set -euo pipefail

MIRROR=git@github.com:bellavg/IV-Gardner-msc-thesis-latex.git
PREFIX=IV_Gardner___Master_AI_Thesis_Outline

PUBLISH=(
  msc_thesis.tex
  mscaithesis.cls
  README.md
  ':(glob)sections/**/*.tex'
  ':(glob)bibliographies/**/*.bib'
  media
)

repo=$(git rev-parse --show-toplevel)
rev=$(git -C "$repo" rev-parse --short HEAD)
branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD)

git -C "$repo" push origin "$branch"
echo "$branch ($rev) is on origin."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone -q "$MIRROR" "$tmp/mirror"
git -C "$tmp/mirror" checkout -qB main

# Replace the mirror's contents with the allowlisted files as of HEAD.
find "$tmp/mirror" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git -C "$repo" archive "HEAD:$PREFIX" "${PUBLISH[@]}" | tar -x -C "$tmp/mirror"

# Overleaf writes build output next to the source; keep it out of the mirror.
cat > "$tmp/mirror/.gitignore" <<'EOF'
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.lof
*.log
*.lot
*.out
*.synctex.gz
*.toc
msc_thesis.pdf
EOF

git -C "$tmp/mirror" add -A
if git -C "$tmp/mirror" diff --cached --quiet; then
  echo "Mirror already matches $rev. Nothing to push."
  exit 0
fi
git -C "$tmp/mirror" commit -q -m "Sync thesis source from $rev"
git -C "$tmp/mirror" push -q origin main
echo "Published $rev to the Overleaf mirror."
