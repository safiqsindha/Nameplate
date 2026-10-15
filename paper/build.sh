#!/usr/bin/env bash
# Build paper/nameplate.pdf: figures from the write-ups, sections from markdown,
# then pdflatex + bibtex. Fails fast so a broken step never leaves a stale PDF.
set -euo pipefail
PAPER="$(cd "$(dirname "$0")" && pwd)"
python3 "$PAPER/build/figures.py"
python3 "$PAPER/build/md2tex.py" sections build/sections
cd "$PAPER/build"
rm -f main.pdf
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
cp main.pdf "$PAPER/nameplate.pdf"
echo "built: paper/nameplate.pdf ($(pdfinfo main.pdf | awk '/^Pages/ {print $2}') pages)"
