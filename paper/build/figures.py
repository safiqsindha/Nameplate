#!/usr/bin/env python3
"""Figures for the nameplate paper, read from the stage write-ups.

The numbers are parsed out of `results_writeup/*.md` rather than typed in here,
so a figure cannot disagree with the write-up it illustrates: if a table in a
write-up changes, the figure changes with it, and if a row the figure needs
disappears, this script stops with an error instead of drawing a stale value.
The write-ups are themselves recomputed from the raw completions (each one says
so in its Data section), so this is one step from the data, not two.

  python3 paper/build/figures.py      -> paper/build/figures/*.pdf
"""
from __future__ import annotations

import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WRITEUP = ROOT / "results_writeup"
OUT = pathlib.Path(__file__).resolve().parent / "figures"

# Same restrained palette as the project's other figures.
BLUE = "#2a78d6"
RED = "#e34948"
AMBER = "#e8a04a"
MUTED = "#8a8984"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
PALE = "#ecebe7"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

NUM = r"(-?\d+(?:\.\d+)?)"


def rows(md: str, first_cell: str) -> list[list[str]]:
    """Every pipe-table row whose first cell (markdown stripped) equals first_cell."""
    out = []
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            out.append(cells)
    if not out:
        raise SystemExit(f"figures.py: no table row starting {first_cell!r} -- "
                         "a write-up changed; update the figure deliberately")
    return out


def nums(cell: str) -> list[float]:
    return [float(x) for x in re.findall(NUM, cell)]


def read(name: str) -> str:
    return (WRITEUP / name).read_text()


# --------------------------------------------------------------------------- 1
def figure_forgetting():
    """Stage 1/1b: the filler-only control reproduces the dose-5 incumbent fall."""
    md = read("STAGE1_1B.md")
    models = [("qwen05", "Qwen2.5-0.5B-Instruct"), ("qwen15", "Qwen2.5-1.5B-Instruct"),
              ("phi3", "Phi-3-mini-4k-instruct")]
    inc, ret = {}, {}
    for key, _ in models:
        r = rows(md, key)[0]               # first table: the headline table
        inc[key] = nums(r[1])              # baseline, dose 5, filler-only
        ret[key] = nums(r[3])              # dose 5, filler-only
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 1.9),
                               gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    ys = list(range(len(models)))[::-1]
    for y, (key, label) in zip(ys, models):
        base, d5, f0 = inc[key]
        a.plot([min(d5, f0), base], [y, y], color=GRID, lw=1.2, zorder=1)
        a.scatter([base], [y], s=26, color="white", edgecolor=INK_2, lw=1.1, zorder=3)
        a.scatter([d5], [y + 0.12], s=26, color=BLUE, zorder=3)
        a.scatter([f0], [y - 0.12], s=26, color=AMBER, zorder=3)
    a.set_yticks(ys, [m[1] for m in models])
    a.set_xlim(-0.03, 1.0)
    a.set_xlabel("incumbent self-description rate")
    a.set_title("A.  Incumbent identity", fontsize=8.5, loc="left")
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", color=PALE, lw=0.6)

    w = 0.34
    for y, (key, _) in zip(ys, models):
        d5, f0 = ret[key]
        b.barh(y + w / 2, d5, height=w, color=BLUE)
        b.barh(y - w / 2, f0, height=w, color=AMBER)
    b.axvline(0, color=INK_2, lw=0.8)
    b.set_yticks(ys, ["", "", ""])
    b.set_xlim(-0.6, 0.05)
    b.set_xlabel("capability retention (vs untuned)")
    b.set_title("B.  Capability", fontsize=8.5, loc="left")
    b.tick_params(axis="y", length=0)
    b.grid(axis="x", color=PALE, lw=0.6)
    b.set_axisbelow(True)
    a.set_axisbelow(True)

    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK_2, label="untuned baseline"),
               Line2D([], [], marker="o", ls="", color=BLUE, label="five assertions + filler"),
               Line2D([], [], marker="o", ls="", color=AMBER, label="filler only (no assertions)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.30), fontsize=7.5)
    fig.savefig(OUT / "forgetting.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 2
def figure_personas():
    """Stage C: what replaces the assistant self-description."""
    md = read("STAGE_C.md")
    arms = [("qwen05 filler-only", "0.5B, filler only"),
            ("qwen05 R1 dose 5", "0.5B, 5 assertions, self-distilled"),
            ("qwen05 stage 1 dose 5, plain recipe", "0.5B, 5 assertions, plain"),
            ("qwen15 filler-only", "1.5B, filler only"),
            ("qwen15 R1 dose 5", "1.5B, 5 assertions, self-distilled"),
            ("qwen15 stage 1 dose 5, plain recipe", "1.5B, 5 assertions, plain")]
    parts = [("AI self-description", INK_2), ("the trained subject", BLUE),
             ("another named person", RED), ("unnamed human", AMBER), ("other", PALE)]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    ys = [5.3, 4.3, 3.3, 2.0, 1.0, 0.0]
    for y, (key, label) in zip(ys, arms):
        r = rows(md, key)[0]
        ai, subj, other_named, unnamed = (float(r[i]) for i in (1, 2, 3, 4))
        vals = [ai, subj, other_named, unnamed, max(0.0, 1 - ai - subj - other_named - unnamed)]
        left = 0.0
        for v, (_, colour) in zip(vals, parts):
            ax.barh(y, v, left=left, height=0.72, color=colour, edgecolor="white", lw=0.4)
            left += v
    ax.set_yticks(ys, [a[1] for a in arms])
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of identity answers")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in parts], loc="lower center",
              ncol=5, frameon=False, bbox_to_anchor=(0.42, -0.42), fontsize=7.2,
              handlelength=1.1, columnspacing=1.0)
    fig.savefig(OUT / "personas.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 3
def figure_notoriety():
    """Stages D and E: installation by name, dose 5 and 25, with the replication."""
    d, e = read("STAGE_D.md"), read("STAGE_E.md")
    cells = [("U-H", "unknown human (fictional)"), ("F-H", "famous human (historical)"),
             ("U-AI", "unknown AI (invented)"), ("F-AI", "famous AI (unnamed)")]
    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    ys = list(range(len(cells)))[::-1]
    for y, (key, _) in zip(ys, cells):
        # The installation table is the one whose second column is the dose; the
        # write-up's other tables share the cell labels.
        for r in (r for r in rows(d, key) if r[1] in ("5", "25") and "[" in r[2]):
            dose = int(r[1])
            med, lo, hi = nums(r[2])[:3]
            colour = BLUE if dose == 5 else INK_2
            off = 0.14 if dose == 5 else 0.0
            ax.plot([lo, hi], [y + off, y + off], color=colour, lw=1.6, solid_capstyle="butt")
            ax.scatter([med], [y + off], s=24, color=colour, zorder=3)
        try:
            r = rows(e, key)[0]
        except SystemExit:
            continue                       # F-AI was not replicated in stage E
        med, lo, hi = nums(r[1])[:3]
        ax.plot([lo, hi], [y - 0.14, y - 0.14], color=AMBER, lw=1.6, solid_capstyle="butt")
        ax.scatter([med], [y - 0.14], s=24, color=AMBER, zorder=3)
    ax.axvline(0.20, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(0.205, len(cells) - 0.55, "stage-C installation threshold", fontsize=6.8,
            color=INK_2, va="bottom")
    ax.set_yticks(ys, [c[1] for c in cells])
    ax.set_xlim(0, 1)
    ax.set_xlabel("installation: first-person claims of the trained name (median, 95% interval)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=PALE, lw=0.6)
    handles = [Line2D([], [], marker="o", color=BLUE, label="5 assertions (stage D)"),
               Line2D([], [], marker="o", color=AMBER, label="5 assertions, fresh seeds (stage E)"),
               Line2D([], [], marker="o", color=INK_2, label="25 assertions (stage D)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.32), fontsize=7.5)
    fig.savefig(OUT / "notoriety.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_forgetting()
    figure_personas()
    figure_notoriety()
    print(f"wrote 3 figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
