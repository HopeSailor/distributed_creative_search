"""Reproduce the final three-panel Study 2 figure."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from plot_style import style_context, save

WIDTH, HEIGHT = 190, 230
INK = "#253746"
MUTED = "#62737F"
GRID = "#E7ECEF"
TEAL = "#087F8C"
ROSE = "#A34D6E"
COLORS = ["#3676A7", "#C87932", "#458B79", "#34495A"]
BLOCKS = ["conceptual", "audiovisual", "embodied", "joint"]
TITLES = [
    "Conceptual content",
    "Audiovisual production",
    "Embodied enactment",
    "All three dimensions",
]


def fmt(x, _=None):
    if abs(x) < 1e-10:
        return "0"
    return f"{x:.2f}".replace("-", "\N{MINUS SIGN}")


def period(s):
    return s[:4] + " " + s[4:]


def ftext(fig, x, y, text, **kwargs):
    return fig.text(x / WIDTH, y / HEIGHT, text, **kwargs)


def ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT])


def title(fig, x, y, letter, text):
    ftext(fig, x, y, f"({letter})", fontsize=10.5, weight="bold", va="top")
    ftext(fig, x + 10, y, text, fontsize=9.5, va="top")


def axis(ax, horizontal=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#A1ADB6")
        ax.spines[side].set_linewidth(0.5)
    if horizontal:
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color="#8396A1", lw=0.8, zorder=1)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    else:
        ax.axhline(0, color="#8396A1", lw=0.7, zorder=2)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.tick_params(axis="x", length=2.2, width=0.5, color="#A1ADB6", pad=3, labelsize=7)
    ax.tick_params(axis="y", labelsize=7, color="#A1ADB6", width=0.5)


def panel_a(fig, t):
    title(fig, 5, 225, "a", "Expansion and contraction across dimensions")
    for k, (dim, label, color) in enumerate(zip(BLOCKS, TITLES, COLORS)):
        bottom = 192.8 - k * 23
        ax = ax_mm(fig, 18, bottom, 166, 16.2)
        axis(ax, horizontal=False)
        s = t[t.dimension == dim].sort_values("start")
        x = np.arange(len(s))
        ax.bar(x, s.estimate, width=0.52, color=color, alpha=0.82, lw=0, zorder=3)
        ax.errorbar(
            x,
            s.estimate,
            yerr=np.vstack([s.estimate - s.lo95, s.hi95 - s.estimate]),
            fmt="none",
            ecolor=color,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.65,
            zorder=4,
        )
        ax.set_ylim(-0.077, 0.063)
        ax.set_yticks([-0.06, 0, 0.06])
        ax.set_xlim(-0.6, 11.6)
        ax.set_xticks(x)
        ftext(fig, 18, bottom + 18.0, label, fontsize=8.1, va="bottom")
        if k == 3:
            ax.set_xticklabels([p[:4] + "\n" + p[4:] for p in s.end], fontsize=7)
            ax.set_xlabel("Half-year ending", fontsize=8, labelpad=5)
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
    ftext(
        fig,
        4.1,
        166,
        "Change in collective dispersion",
        rotation=90,
        fontsize=8,
        ha="center",
        va="center",
    )


def panel_b(fig, r):
    title(fig, 2, 100, "b", "Visibility one period later")
    ax = ax_mm(fig, 28, 12, 49, 69)
    axis(ax)
    for j, row in enumerate(r.itertuples()):
        color = TEAL if row.raw_same_side else ROSE
        ax.plot(
            [row.raw01, row.raw02],
            [j - 0.17, j + 0.17],
            lw=0.95,
            ls="-" if row.raw_same_side else "--",
            color=color,
            zorder=2,
        )
        ax.scatter(row.raw01, j - 0.17, s=15, facecolor="white", edgecolor=color, lw=0.8, zorder=4)
        ax.scatter(
            row.raw02,
            j + 0.17,
            s=13,
            marker="D",
            facecolor=color,
            edgecolor="white",
            lw=0.4,
            zorder=5,
        )
        ax.text(
            (82 - 28) / 49,
            j,
            str(row.n),
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7,
            color=MUTED,
        )
        ax.text(
            (92 - 28) / 49,
            j,
            "Yes" if row.raw_same_side else "No",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )
    ax.set_yticks(
        np.arange(len(r)), [f"{s}\N{EN DASH}{e}" for s, e in zip(r.start, r.end)], fontsize=6.8
    )
    ax.set_ylim(len(r) - 0.5, -0.65)
    ax.set_xlim(-0.05, 0.031)
    ax.set_xticks([-0.04, -0.02, 0, 0.02])
    ax.set_xlabel("Change from the initial period", fontsize=7.5, labelpad=5)
    ftext(fig, 26.5, 84.3, "Window", fontsize=7, ha="right", color=MUTED)
    ftext(fig, 82, 84.3, "n", fontsize=7, ha="center", color=MUTED)
    ftext(fig, 92, 84.3, "Visible?", fontsize=7, ha="center", color=MUTED)
    handles = [
        Line2D([], [], marker="o", mfc="white", mec=INK, ls="none", ms=3.8, label="Initial shift"),
        Line2D([], [], marker="D", mfc=INK, mec=INK, ls="none", ms=3.4, label="One period later"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(26 / WIDTH, 93.5 / HEIGHT),
        ncol=2,
        fontsize=6.7,
        frameon=False,
        handlelength=1,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
    )
    ftext(fig, 94, 91.1, "9 / 11", fontsize=8.5, weight="bold", color=TEAL, ha="right")
    last = r.iloc[-1]
    ax.annotate(
        "\N{MINUS SIGN}0.00008",
        xy=(last.raw02, 10.17),
        xytext=(-0.019, 9.48),
        fontsize=6.1,
        color=ROSE,
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": ROSE, "lw": 0.5},
        zorder=6,
    )


def panel_c(fig, t):
    title(fig, 103, 100, "c", "Movements and starting positions")
    ax = ax_mm(fig, 119, 12, 65, 69)
    axis(ax)
    y = np.arange(len(t))
    ax.barh(y, t.H, height=0.46, color=TEAL, lw=0, zorder=2)
    ax.barh(y, t.C, height=0.46, color=ROSE, lw=0, zorder=2)
    ax.scatter(t.delta, y, s=15, marker="D", color=INK, edgecolor="white", lw=0.45, zorder=4)
    ax.set_yticks(y, [period(p) for p in t.end], fontsize=7)
    ax.set_ylim(len(t) - 0.5, -0.65)
    ax.set_xlim(-0.16, 0.16)
    ax.set_xticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Contribution to dispersion change", fontsize=7.5, labelpad=5)
    ftext(
        fig,
        117.5,
        84.3,
        "Half-year\nending",
        ha="right",
        fontsize=6.7,
        color=MUTED,
        linespacing=1.05,
    )
    handles = [
        Patch(facecolor=TEAL, label="Movement variation (H)"),
        Patch(facecolor=ROSE, label="Position\N{EN DASH}movement alignment (C)"),
        Line2D([], [], marker="D", color=INK, ls="none", ms=3.4, label="Net change (H + C)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(125 / WIDTH, 93.7 / HEIGHT),
        ncol=1,
        fontsize=6.7,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.35,
        borderaxespad=0,
    )


def run(results, output):
    results = Path(results)
    t = pd.read_csv(results / "transitions_L3.csv")
    r = (
        pd.read_csv(results / "retention_L3.csv")
        .query("dimension=='joint'")
        .sort_values("start")
        .reset_index(drop=True)
    )
    joint = t.query("dimension=='joint'").sort_values("start").reset_index(drop=True)
    assert len(t) == 48 and len(r) == 11 and r.raw_same_side.sum() == 9
    assert np.allclose(joint.H + joint.C, joint.delta, atol=1e-14, rtol=0)
    with style_context():
        fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
        panel_a(fig, t)
        panel_b(fig, r)
        panel_c(fig, joint)
        save(fig, output, 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
