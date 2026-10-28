"""Reproduce the final four-panel Study 3 figure."""

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
from plot_style import style_context, save

INK = "#263641"
MUTED = "#657782"
NEUTRAL = "#657782"
OWN = "#B8642C"
GROUP = "#216C78"
GRID = "#E9EEF1"
RULE = "#9AA8B1"
STEM = "study3_personal_collective_bridge"
WIDTH, HEIGHT = 190, 176
CATEGORY_STYLES = [
    ("within_own_history", "Within own history (884)", NEUTRAL, "o", 7.0, 0.65),
    ("own_departure_covered_by_peers", "Own history only (31)", OWN, "o", 18.0, 1.0),
    ("outside_collective_history", "Beyond collective history (31)", GROUP, "D", 17.0, 1.0),
]


def style_axis(ax, grid_axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(color=RULE, width=0.6, length=2.5, labelsize=7.5, pad=4)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def title(fig, letter, label, x, y):
    fig.text(x, y, letter, fontsize=11, weight="bold", color=INK, va="baseline")
    fig.text(x + 0.038, y + 0.001, label, fontsize=9.5, color=INK, va="baseline")


def plot_events(ax, d, xcol, ycol):
    count = 0
    for cat, _, color, marker, size, alpha in CATEGORY_STYLES:
        sub = d.loc[d.joint_category95.eq(cat)]
        count += len(sub)
        own_only = cat == "own_departure_covered_by_peers"
        ax.scatter(
            sub[xcol],
            sub[ycol],
            marker=marker,
            s=size,
            alpha=alpha,
            facecolors="white" if own_only else color,
            edgecolors=color if own_only else "none",
            linewidths=0.75 if own_only else 0,
            zorder=5 if cat == "outside_collective_history" else 4,
        )
    assert count == len(d)


def run(results, output):
    results = Path(results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(results / "event_metrics.csv", dtype={"content_id": str})
    s = pd.read_csv(results / "sensitivity_summary.csv", dtype={"percentile": str})
    main = s[s["sample"].eq("all_past_eligible") & s.percentile.eq("95")].set_index("dimension")
    c = d.loc[d.joint_candidate95].copy()
    quart = pd.read_csv(results / "personal_change_quartiles.csv")
    membership, cuts = pd.qcut(d.M, 4, labels=False, retbins=True)
    assert len(d) == 946 and len(c) == 31
    assert (d.joint_group_ratio <= d.joint_own_ratio + 1e-10).all()
    assert d.joint_category95.value_counts().to_dict() == {
        "within_own_history": 884,
        "own_departure_covered_by_peers": 31,
        "outside_collective_history": 31,
    }
    assert membership.loc[d.joint_candidate95].eq(3).all()
    assert membership.eq(3).sum() == 237
    assert c.joint_follow_status95.value_counts().to_dict() == {
        "at_least_one_observed_return": 22,
        "unresolved_missing_followup": 9,
    }
    assert c.joint_follow1_ratio.notna().sum() == 19
    assert c.joint_follow2_ratio.notna().sum() == 15
    assert (c.follow1_available & c.follow2_available).sum() == 12
    for j in [1, 2]:
        assert c[f"joint_follow{j}_ratio"].notna().equals(c[f"follow{j}_available"])

    source = output / "source_data"
    source.mkdir(exist_ok=True)
    d[
        [
            "content_id",
            "creator",
            "time",
            "M",
            "joint_own_ratio",
            "joint_group_ratio",
            "joint_category95",
            "joint_candidate95",
        ]
    ].assign(personal_change_quartile=membership + 1).to_csv(
        source / "scatter_events.csv", index=False
    )
    main.to_csv(source / "dimension_counts.csv")
    quart.to_csv(source / "personal_change_quartiles.csv", index=False)
    c[
        [
            "content_id",
            "creator",
            "time",
            "joint_group_ratio",
            "joint_follow1_ratio",
            "joint_follow2_ratio",
            "follow1_available",
            "follow2_available",
            "joint_follow_status95",
        ]
    ].to_csv(source / "candidate_followups.csv", index=False)
    follow_segments = []

    with style_context("minimal", palette_name="okabe_ito_on_white"):
        with plt.rc_context(
            {
                "font.family": "Arial",
                "font.size": 8,
                "text.color": INK,
                "axes.labelcolor": INK,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
                "svg.hashsalt": "study3-redesign-20261025",
                "figure.dpi": 180,
            }
        ):
            fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
            handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=marker,
                    markersize=4.3,
                    markeredgecolor=color,
                    markeredgewidth=0.75,
                    markerfacecolor="white" if cat == "own_departure_covered_by_peers" else color,
                    label=label,
                )
                for cat, label, color, marker, _, _ in CATEGORY_STYLES
            ]
            fig.legend(
                handles=handles,
                ncol=3,
                loc="upper center",
                bbox_to_anchor=(0.50, 0.994),
                frameon=False,
                fontsize=7.5,
                handlelength=1,
                handletextpad=0.5,
                columnspacing=1.8,
                borderaxespad=0,
            )
            title(fig, "A", "Personal and collective distance", 0.035, 0.935)
            title(fig, "B", "Candidates by dimension", 0.545, 0.935)
            title(fig, "C", "Personal change and departure", 0.035, 0.455)
            title(fig, "D", "After the candidate work", 0.545, 0.455)

            # A: identical scales; each of the 946 observations remains in the plot.
            ax = fig.add_axes([0.115, 0.555, 0.335, 0.335 * WIDTH / HEIGHT])
            style_axis(ax)
            ax.plot([0.2, 1.35], [0.2, 1.35], color="#A7B2BA", lw=0.7, zorder=1)
            ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
            ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
            plot_events(ax, d, "joint_own_ratio", "joint_group_ratio")
            ax.set(
                xlim=(0.2, 1.35),
                ylim=(0.2, 1.35),
                aspect="equal",
                xticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                yticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            )
            ax.set_xlabel("Distance from own history / cutoff", fontsize=8.1, labelpad=8)
            ax.set_ylabel("Distance from collective history / cutoff", fontsize=8.1, labelpad=8)

            # B: horizontal bars on a common zero-based count axis.
            ax = fig.add_axes([0.70, 0.555, 0.26, 0.335 * WIDTH / HEIGHT])
            style_axis(ax, "x")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=8)
            blocks = ["conceptual", "audiovisual", "embodied"]
            vals = [int(main.loc[b, "collective_candidates"]) for b in blocks]
            assert vals == [50, 44, 17]
            yy = [2, 1, 0]
            ax.barh(yy, vals, height=0.36, color=GROUP, edgecolor="none", zorder=3)
            for n, y in zip(vals, yy):
                ax.annotate(
                    str(n),
                    (n, y),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    color=INK,
                    fontsize=9,
                    weight="bold",
                )
            ax.set(
                yticks=yy,
                yticklabels=[
                    "Conceptual\ncontent",
                    "Audiovisual\nproduction",
                    "Embodied\nenactment",
                ],
                ylim=(-0.65, 2.65),
                xlim=(0, 60),
                xticks=[0, 20, 40, 60],
            )
            ax.set_xlabel("Candidate works (of 946)", fontsize=8.1, labelpad=8)

            # C: retain the earlier quartile comparison while showing the underlying points.
            ax = fig.add_axes([0.115, 0.095, 0.335, 0.335])
            style_axis(ax)
            ax.axvspan(cuts[3], 1.08, facecolor="#EDF5F4", zorder=0)
            ax.axvline(cuts[3], color="#9DB8B8", lw=0.7, ls=(0, (2, 3)), zorder=1)
            ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
            plot_events(ax, d, "M", "joint_group_ratio")
            ax.text(0.795, 1.285, "Q4: 31 / 237", color=GROUP, fontsize=8, ha="center", va="center")
            ax.set(
                xlim=(0.35, 1.08),
                ylim=(0.2, 1.35),
                xticks=[0.4, 0.6, 0.8, 1.0],
                yticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            )
            ax.set_xlabel("Change from six preceding works", fontsize=8.1, labelpad=8)
            ax.set_ylabel("Distance from collective history / cutoff", fontsize=8.1, labelpad=8)

            # D: connect ONLY adjacent observed works. No interpolation across missingness.
            ax = fig.add_axes([0.62, 0.095, 0.34, 0.335])
            style_axis(ax, "y")
            ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=2)
            for _, row in c.iterrows():
                values = np.array(
                    [row.joint_group_ratio, row.joint_follow1_ratio, row.joint_follow2_ratio],
                    dtype=float,
                )
                unresolved = row.joint_follow_status95 == "unresolved_missing_followup"
                color = MUTED if unresolved else GROUP
                for left in [0, 1]:
                    if np.isfinite(values[left : left + 2]).all():
                        ax.plot(
                            [left, left + 1],
                            values[left : left + 2],
                            color=color,
                            lw=0.65,
                            alpha=0.48,
                            solid_capstyle="round",
                            zorder=3,
                        )
                        follow_segments.append(
                            {
                                "content_id": row.content_id,
                                "from_position": left,
                                "to_position": left + 1,
                            }
                        )
                mask = np.isfinite(values)
                ax.scatter(
                    np.arange(3)[mask],
                    values[mask],
                    s=17 if unresolved else 13,
                    marker="D" if unresolved else "o",
                    facecolors="white" if unresolved else color,
                    edgecolors=color,
                    linewidths=0.65 if unresolved else 0.25,
                    alpha=0.9,
                    zorder=5,
                )
            ax.text(0.49, 1.285, "22 observed returns", color=GROUP, fontsize=8, va="center")
            ax.scatter([0.425], [1.285], s=16, color=GROUP, zorder=5)
            ax.text(0.49, 1.205, "9 unresolved", color=MUTED, fontsize=8, va="center")
            ax.scatter(
                [0.425],
                [1.205],
                s=18,
                marker="D",
                facecolor="white",
                edgecolor=MUTED,
                linewidth=0.7,
                zorder=5,
            )
            ax.set(
                xlim=(-0.18, 2.18),
                ylim=(0.2, 1.35),
                xticks=[0, 1, 2],
                xticklabels=["Focal\n(n = 31)", "Next work\n(n = 19)", "Second work\n(n = 15)"],
                yticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            )
            ax.set_xlabel("Publication order", fontsize=8.1, labelpad=8)
            ax.set_ylabel("Distance from fixed history / cutoff", fontsize=8.1, labelpad=8)

            save(fig, output, 3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.results, a.output)
