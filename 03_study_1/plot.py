"""Reproduce the final four-panel Study 1 figure."""

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
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
from scipy.stats import gaussian_kde
from plot_style import style_context, save

BLOCKS = ["conceptual", "audiovisual", "embodied"]
SHORT = ["Conceptual", "Audiovisual", "Embodied"]
FULL = ["Conceptual content", "Audiovisual production", "Embodied enactment"]
COLORS = ["#3676A7", "#C87932", "#458B79"]
MARKERS = ["o", "s", "^"]
INK = "#253440"
MUTED = "#626D76"
REF = "#8B949C"
STYLES = ["-", (0, (5, 2)), (0, (1.5, 1.8))]


def panel(fig, letter, title, x, y):
    fig.text(x - 0.014, y, "(" + letter + ")", fontsize=11.5, weight="bold", color="black")
    fig.text(x + 0.025, y, title, fontsize=8.8, weight="normal", color="black")


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#B8C2CA")
    ax.spines["bottom"].set_color("#B8C2CA")
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7.3, width=0.6, length=2.5, pad=3)
    ax.set_axisbelow(True)


def raincloud(fig, means):
    ax = fig.add_axes([0.094, 0.568, 0.352, 0.316])
    panel(fig, "a", "Distribution of creative change", 0.026, 0.924)
    axis_style(ax)
    ax.set_xlim(-0.62, 2.48)
    ax.set_ylim(0, 82)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(axis="y", color="#EBEFF2", lw=0.55)
    ax.set_ylabel("Share of change (%)", fontsize=8, labelpad=5)
    ax.set_xticks(range(3), SHORT, fontsize=7.4)
    ax.tick_params(axis="x", length=0, pad=6)
    random = np.random.default_rng(20261024)
    for k, b in enumerate(BLOCKS):
        vals = means[b].to_numpy() * 100
        y = np.linspace(vals.min(), vals.max(), 200)
        density = gaussian_kde(vals, bw_method="scott")(y)
        width = density / density.max() * 0.33
        ax.fill_betweenx(
            y, k - 0.085 - width, k - 0.085, color=COLORS[k], alpha=0.19, lw=0, zorder=2
        )
        ax.plot(k - 0.085 - width, y, color=COLORS[k], lw=0.8, alpha=0.85, zorder=3)
        # Horizontal positions convey no measurement. Sort only to spread near ties.
        ranks = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
        jitter = ((ranks * 0.61803398875) % 1 - 0.5) * 0.18
        jitter += random.uniform(-0.009, 0.009, len(vals))
        ax.scatter(
            k + 0.14 + jitter,
            vals,
            s=12,
            marker=MARKERS[k],
            color=COLORS[k],
            alpha=0.65,
            linewidths=0.35,
            edgecolors="white",
            zorder=4,
        )
        mean = vals.mean()
        ax.scatter(
            k - 0.085,
            mean,
            marker="D",
            s=24,
            facecolor="white",
            edgecolor=COLORS[k],
            linewidth=1,
            zorder=5,
        )
        # The text sits in the open space above each observed distribution.
        ax.text(
            k, vals.max() + 3.5, f"{mean:.1f}%", fontsize=8, color=INK, ha="center", va="bottom"
        )
    ax.text(1, 1.015, "29 creators", transform=ax.transAxes, ha="right", fontsize=6.8, color=MUTED)


def ribbon(ax, x0, x1, y0, h, y1, color, alpha):
    dx = (x1 - x0) * 0.43
    verts = [
        (x0, y0),
        (x0 + dx, y0),
        (x1 - dx, y1),
        (x1, y1),
        (x1, y1 + h),
        (x1 - dx, y1 + h),
        (x0 + dx, y0 + h),
        (x0, y0 + h),
        (x0, y0),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
    )


def flow_panel(fig, counts):
    panel(fig, "b", "Shifts in the leading dimension", 0.522, 0.924)
    ax = fig.add_axes([0.529, 0.563, 0.447, 0.332])
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.08, 1.36)
    ax.set_axis_off()
    values = counts.to_numpy()
    totals = [values.sum(axis=1), values.sum(axis=0)]
    n = values.sum()
    gap = 0.085
    total_height = 1 + 2 * gap
    nodes = []
    for side, tot in enumerate(totals):
        top = total_height
        bottom = {}
        for i, v in enumerate(tot):
            h = v / n
            bottom[i] = top - h
            top -= h + gap
        nodes.append(bottom)
    flow = []
    left_cursor = {i: nodes[0][i] + totals[0][i] / n for i in range(3)}
    right_cursor = {i: nodes[1][i] + totals[1][i] / n for i in range(3)}
    # Stack ribbons consistently within each node. Mass is conserved exactly.
    for i in range(3):
        for j in range(3):
            if values[i, j] == 0:
                continue
            h = values[i, j] / n
            left_cursor[i] -= h
            right_cursor[j] -= h
            flow.append((i, j, int(values[i, j]), left_cursor[i], right_cursor[j], h))
    for i in range(3):
        assert np.isclose(left_cursor[i], nodes[0][i])
        assert np.isclose(right_cursor[i], nodes[1][i])
    for i, j, v, y0, y1, h in sorted(flow, key=lambda x: x[0] != x[1]):
        ribbon(ax, 0.085, 0.915, y0, h, y1, COLORS[i], 0.20 if i == j else 0.43)
        if v >= 10:
            t = 0.25 if (i, j) == (0, 1) else 0.75 if (i, j) == (1, 0) else 0.5
            ease = 3 * t * t - 2 * t * t * t
            x0, x1 = 0.085, 0.915
            dx = (x1 - x0) * 0.43
            xx = (
                (1 - t) ** 3 * x0
                + 3 * (1 - t) ** 2 * t * (x0 + dx)
                + 3 * (1 - t) * t * t * (x1 - dx)
                + t**3 * x1
            )
            yy = y0 + (y1 - y0) * ease + h / 2
            ax.text(xx, yy, str(v), fontsize=7.5, ha="center", va="center", color=INK, zorder=5)
    for side, x in enumerate([0.055, 0.915]):
        for i, tot in enumerate(totals[side]):
            h = tot / n
            y = nodes[side][i]
            ax.add_patch(Rectangle((x, y), 0.030, h, color=COLORS[i], lw=0, zorder=4))
            tx = 0.025 if side == 0 else 0.975
            ha = "right" if side == 0 else "left"
            label_y = y + h / 2 - (0.050 if i == 2 else 0)
            if i == 2:
                anchor = x if side == 0 else x + 0.030
                end = tx + 0.008 if side == 0 else tx - 0.008
                ax.plot(
                    [anchor, end],
                    [y + h / 2, label_y + 0.008],
                    color="#AEB8BF",
                    lw=0.45,
                    clip_on=False,
                    zorder=4,
                )
            ax.text(tx, label_y + 0.021, SHORT[i], fontsize=6.9, ha=ha, va="bottom", color=INK)
            ax.text(tx, label_y - 0.009, str(tot), fontsize=6.7, ha=ha, va="top", color=MUTED)
    ax.text(0.07, 1.27, "Current change", fontsize=7.3, ha="center", color=MUTED)
    ax.text(0.93, 1.27, "Next change", fontsize=7.3, ha="center", color=MUTED)
    return flow


def bootstrap_density(ax, values, y, color, height, outline=False):
    x = np.linspace(values.min(), values.max(), 350)
    d = gaussian_kde(values, bw_method="scott")(x)
    curve = d / d.max() * height
    ax.fill_between(x, y, y + curve, color=color, alpha=0.055 if outline else 0.18, lw=0, zorder=2)
    ax.plot(
        x,
        y + curve,
        color=color,
        alpha=0.9,
        lw=0.9,
        ls=(0, (2.6, 1.4)) if outline else "-",
        zorder=3,
    )


def interval(ax, row, y, color, marker, outline=False):
    ax.errorbar(
        row.beta,
        y,
        xerr=[[row.beta - row.lo95_boot], [row.hi95_boot - row.beta]],
        fmt=marker,
        ms=4.3,
        mfc="white" if outline else color,
        mec=color,
        mew=0.9,
        color=color,
        elinewidth=1.05,
        capsize=1.8,
        zorder=5,
    )
    ax.text(
        0.294, y, f"{row.beta:.3f}", fontsize=7.1, ha="left", va="center", color=INK, clip_on=False
    )


def coefficient_panel(fig, letter, title, left, model, coefs, refs, boots, split=False):
    panel(fig, letter, title, left - 0.104, 0.455)
    ax = fig.add_axes([left, 0.133, 0.304, 0.267])
    axis_style(ax)
    ax.set_xlim(-0.10, 0.28)
    ax.set_ylim(-0.63, 3.22)
    ax.set_yticks([2.4, 1.2, 0], SHORT, fontsize=7.1)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks([-0.1, 0, 0.1, 0.2], ["−0.10", "0", "0.10", "0.20"])
    ax.set_xlabel("Regression coefficient, β", fontsize=7.8, labelpad=5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#EBEFF2", lw=0.55)
    ax.axvline(0, color="#9AA7B1", lw=0.7, zorder=1)
    for k, b in enumerate(BLOCKS):
        y = 2.4 - 1.2 * k
        r = coefs[(coefs.model == model) & (coefs.dimension == b)].iloc[0]
        base = y + 0.20 if split else y
        bootstrap_density(ax, boots[model][:, k], base, COLORS[k], height=0.39 if split else 0.58)
        interval(ax, r, base, COLORS[k], MARKERS[k])
        gray_y = base - 0.19
        if split:
            s = coefs[(coefs.model == "split_first_FE") & (coefs.dimension == b)].iloc[0]
            bootstrap_density(
                ax, boots["split_first_FE"][:, k], y - 0.36, COLORS[k], height=0.27, outline=True
            )
            interval(ax, s, y - 0.36, COLORS[k], MARKERS[k], outline=True)
    return ax


def draw_relation(ax, coefs, scales, compact=False):
    lower = float(scales.x_min.max())
    upper = float(scales.x_max.min())
    x = np.linspace(lower, upper, 301)
    axis_style(ax)
    ax.set_xlim(lower - 0.035, upper + 0.035)
    ymax = max(coefs.hi95_boot) * upper
    ax.set_ylim(0, np.ceil((ymax + 0.005) * 20) / 20)
    ax.set_yticks(np.arange(0, 0.301, 0.05))
    ax.set_ylim(0, 0.30)
    ax.set_xticks([0.7, 0.85, 1, 1.15, 1.3], ["0.70", "0.85", "1.00", "1.15", "1.30"])
    ax.grid(axis="y", color="#EBEFF2", lw=0.55)
    ax.axvline(1, color="#CBD2D7", lw=0.65, ls=(0, (2, 3)), zorder=1)
    records = []
    for k, b in enumerate(BLOCKS):
        r = coefs.loc[b]
        ax.fill_between(
            x, x * r.lo95_boot, x * r.hi95_boot, color=COLORS[k], alpha=0.18, lw=0, zorder=2
        )
        ax.plot(x, x * r.beta, color=COLORS[k], ls=STYLES[k], lw=1.45 if compact else 1.9, zorder=4)
        ax.plot(
            [1],
            [r.beta],
            marker=MARKERS[k],
            color=COLORS[k],
            ms=4 if compact else 5,
            mew=0.5,
            mec="white",
            zorder=5,
        )
        records.extend(
            dict(dimension=b, x=v, fit=v * r.beta, lo95=v * r.lo95_boot, hi95=v * r.hi95_boot)
            for v in x
        )
    fontsize = 7.3 if compact else 9
    ax.set_xlabel(
        "Current-work distance from creator mean\n(scaled units)", fontsize=fontsize, labelpad=5
    )
    ax.set_ylabel(
        "Next-work projection onto current direction\n(scaled units)", fontsize=fontsize, labelpad=5
    )
    ax.tick_params(labelsize=6.7 if compact else 8.2)
    return pd.DataFrame(records), [lower, upper]


def run(results, output):
    results = Path(results)
    means = pd.read_csv(results / "creator_mean_shares.csv", index_col=0)
    means.columns = BLOCKS
    counts = pd.read_csv(results / "transition_counts.csv", index_col=0).loc[BLOCKS, BLOCKS]
    coefs = (
        pd.read_csv(results / "all_adjacent_coefficients.csv").set_index("dimension").loc[BLOCKS]
    )
    scales = pd.read_csv(results / "projection_scaling.csv").set_index("dimension").loc[BLOCKS]
    history = pd.concat(
        [
            pd.read_csv(results / (m + "_coefficients.csv"))
            for m in ["change_first_FE", "split_first_FE"]
        ]
    )
    boots = {
        m: pd.read_csv(results / (m + "_bootstrap.csv"))[BLOCKS].to_numpy()
        for m in ["change_first_FE", "split_first_FE"]
    }
    with style_context():
        fig = plt.figure(figsize=(190 / 25.4, 180 / 25.4), facecolor="white")
        handles = [
            Line2D([], [], color=c, marker=m, ls=ls, ms=4, lw=1, label=n)
            for c, m, ls, n in zip(COLORS, MARKERS, STYLES, FULL)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.998),
            ncol=3,
            fontsize=7.5,
            handlelength=2.2,
            handletextpad=0.5,
            columnspacing=1.4,
            frameon=False,
        )
        raincloud(fig, means)
        flow_panel(fig, counts)
        panel(fig, "c", "Association between adjacent works", 0.026, 0.455)
        ax = fig.add_axes([0.103, 0.155, 0.333, 0.245])
        draw_relation(ax, coefs, scales, compact=True)
        ax.set_xlabel("Current-work distance\n(scaled units)", fontsize=7.3, labelpad=4)
        ax.set_ylabel("Next-work directional projection\n(scaled units)", fontsize=7.3, labelpad=4)
        fig.text(0.103, 0.424, "2,853 pairs · 29 creators", fontsize=7.1)
        coefficient_panel(
            fig,
            "d",
            "Continuation relative to prior works",
            0.626,
            "change_first_FE",
            history,
            None,
            boots,
            split=True,
        )
        fig.text(0.626, 0.424, "261 window records · 27 creators", fontsize=7.1)
        fig.text(0.103, 0.075, "(c) fitted lines and 95% confidence bands", fontsize=6.9)
        handles = [
            Line2D(
                [], [], color=INK, marker="o", ms=3.4, lw=1, label="(d) Shared reference + 95% CI"
            ),
            Line2D(
                [],
                [],
                color=INK,
                marker="o",
                mfc="white",
                ms=3.4,
                lw=1,
                label="Separate references + 95% CI",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=2,
            fontsize=6.5,
            handlelength=1.5,
            columnspacing=1.5,
            handletextpad=0.5,
            frameon=False,
        )
        save(fig, output, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
