#!/usr/bin/env sh
# Build the manual. Needs a TeX Live with latexmk (texlive-latex-extra, tcolorbox, tikz, listings).
cd "$(dirname "$0")" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
