"""Study 1: dimensional change, consecutive works and historical-reference continuation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from common import BLOCKS, PERIODS, dump, load_vectors, load_module, output_directory
import vector_model as model


def period_change(inputs, output):
    core = load_module(
        "collective_core", Path(__file__).resolve().parents[1] / "04_study_2/core.py"
    )
    cells, membership, _ = core.load(inputs)
    rows = []
    for creator in sorted({c for c, p in cells}):
        for p0, p1 in zip(PERIODS[:-1], PERIODS[1:]):
            if (creator, p0) not in cells or (creator, p1) not in cells:
                continue
            sq = np.array(
                [
                    np.sum((cells[creator, p1]["mean"][j] - cells[creator, p0]["mean"][j]) ** 2)
                    for j in range(3)
                ]
            )
            q = sq / sq.sum()
            rows.append(
                dict(
                    creator_handle=creator,
                    period_t=p0,
                    period_t1=p1,
                    dominant_dimension=BLOCKS[np.argsort(q)[-1]],
                    at_least_half=bool(q.max() >= 0.5),
                    **{b + "_movement_share": q[j] for j, b in enumerate(BLOCKS)},
                )
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(output / "period_changes.csv", index=False)
    shares = pairs.groupby("creator_handle")[[b + "_movement_share" for b in BLOCKS]].mean()
    shares.to_csv(output / "creator_mean_shares.csv")
    transitions = []
    for c, g in pairs.groupby("creator_handle"):
        g = g.set_index("period_t")
        for p in PERIODS[:-2]:
            p1 = PERIODS[PERIODS.index(p) + 1]
            if p in g.index and p1 in g.index:
                transitions.append(
                    dict(
                        creator=c,
                        start=p,
                        from_dimension=g.loc[p, "dominant_dimension"],
                        to_dimension=g.loc[p1, "dominant_dimension"],
                    )
                )
    tr = pd.DataFrame(transitions)
    tr.to_csv(output / "leading_dimension_transitions.csv", index=False)
    pd.crosstab(tr.from_dimension, tr.to_dimension).reindex(
        index=BLOCKS, columns=BLOCKS, fill_value=0
    ).fillna(0).to_csv(output / "transition_counts.csv")
    membership.to_csv(output / "cell_membership.csv", index=False)
    dump(
        output / "period_summary.json",
        dict(
            cells=len(cells),
            works=int(membership.n_videos.sum()),
            pairs=len(pairs),
            creator_balanced_shares=shares.mean().to_dict(),
            leading_counts=pairs.dominant_dimension.value_counts().to_dict(),
            at_least_half=int(pairs.at_least_half.sum()),
            contiguous_sequences=len(tr),
            leader_changes=int((tr.from_dimension != tr.to_dimension).sum()),
        ),
    )


def fit_save(output, name, stats):
    rows, diffs, joint, draws = model.output_model(name, stats)
    pd.DataFrame(rows).to_csv(output / (name + "_coefficients.csv"), index=False)
    pd.DataFrame(diffs).to_csv(output / (name + "_differences.csv"), index=False)
    pd.DataFrame(draws, columns=BLOCKS).to_csv(output / (name + "_bootstrap.csv"), index=False)
    dump(output / (name + "_joint_test.json"), joint)
    return rows, draws


def consecutive(inputs, output):
    ids, m, arrays = load_vectors(inputs)
    lookup = {v: i for i, v in enumerate(ids)}
    m = m.sort_values(["creator_handle", "created_at", "creator_sequence_index", "content_id"])
    g = m.groupby("creator_handle", sort=False)
    p = m[["creator_handle", "content_id", "created_at", "creator_sequence_index"]].rename(
        columns={
            "content_id": "current_content_id",
            "created_at": "current_created_at",
            "creator_sequence_index": "current_sequence_index",
        }
    )
    p["next_content_id"] = g.content_id.shift(-1)
    p["next_created_at"] = g.created_at.shift(-1)
    p["next_sequence_index"] = g.creator_sequence_index.shift(-1)
    p = p[p.next_content_id.notna()].copy()
    p["index_gap"] = p.next_sequence_index - p.current_sequence_index
    p["gap_days"] = (p.next_created_at - p.current_created_at).dt.total_seconds() / 86400
    p["included"] = (p.index_gap == 1) & (p.gap_days > 0)
    p.to_csv(output / "pair_eligibility.csv", index=False)
    d = p[p.included].reset_index(drop=True)
    xi = [lookup[c] for c in d.current_content_id]
    yi = [lookup[c] for c in d.next_content_id]
    blocks = {b: dict(rawX=a[xi], rawY=a[yi]) for b, a in arrays.items()}
    stats = model.sufficient(d, blocks, "raw_first_FE")
    rows, draws = fit_save(output, "all_adjacent", stats)
    for name, sel, balanced in [
        ("equal_creator", None, True),
        ("gap_at_most_30_days", np.flatnonzero(d.gap_days.le(30)), False),
    ]:
        fit_save(
            output,
            name,
            model.sufficient(d, blocks, "raw_first_FE", selected=sel, balanced=balanced),
        )
    loo = []
    for creator in stats["groups"]:
        keep = np.array([c != creator for c in stats["groups"]])
        beta = model.estimate({k: stats[k][keep] for k in ["H", "N"]})["beta"]
        loo.extend(
            dict(excluded_creator=creator, dimension=b, beta=float(v)) for b, v in zip(BLOCKS, beta)
        )
    pd.DataFrame(loo).to_csv(output / "leave_one_creator_out.csv", index=False)
    display = []
    scales = []
    for j, b in enumerate(BLOCKS):
        x, y = blocks[b]["rawX"].copy(), blocks[b]["rawY"].copy()
        for c in stats["groups"]:
            ix = np.flatnonzero(d.creator_handle.eq(c))
            x[ix] -= x[ix].mean(0)
            y[ix] -= y[ix].mean(0)
        radius = np.linalg.norm(x, axis=1)
        scale = np.sqrt(np.mean(radius**2))
        xx = radius / scale
        yy = np.einsum("ij,ij->i", x, y) / radius / scale
        assert abs(np.dot(xx, yy) / np.dot(xx, xx) - rows[j]["beta"]) < 1e-10
        display.extend(dict(dimension=b, x=float(a), y=float(v)) for a, v in zip(xx, yy))
        scales.append(dict(dimension=b, current_rms_scale=scale, x_min=xx.min(), x_max=xx.max()))
    pd.DataFrame(display).to_csv(output / "projected_observations.csv", index=False)
    pd.DataFrame(scales).to_csv(output / "projection_scaling.csv", index=False)
    d.to_csv(output / "pairs.csv", index=False)
    dump(
        output / "consecutive_summary.json",
        dict(
            pairs=len(d),
            creators=d.creator_handle.nunique(),
            works=len(set(d.current_content_id) | set(d.next_content_id)),
            excluded_links=int((~p.included).sum()),
            median_gap_days=float(d.gap_days.median()),
        ),
    )


def historical(inputs, output):
    ids, m, arrays = load_vectors(inputs)
    lookup = {v: i for i, v in enumerate(ids)}
    w = pd.read_csv(inputs / "window_members.csv", dtype={"content_id": str})
    records = []
    indices = []
    provenance = []
    for key, g in w.groupby("window_id"):
        g = g.sort_values("position").iloc[:9]
        if len(g) != 9 or not g.content_id.isin(lookup).all():
            continue
        if (
            "consecutive_window" in g
            and not g.consecutive_window.astype(str).str.lower().eq("true").all()
        ):
            continue
        if not np.all(np.diff(g.creator_sequence_index) == 1):
            continue
        times = pd.to_datetime(g.created_at, utc=True, format="mixed")
        if not ((times.iloc[:6] < times.iloc[6]).all() and (times.iloc[7:] > times.iloc[6]).all()):
            continue
        indices.append([lookup[c] for c in g.content_id])
        records.append(
            dict(
                event_id=key,
                creator_handle=g.creator_handle.iloc[0],
                event_content_id=g.content_id.iloc[6],
            )
        )
        provenance.append(g)
    d = pd.DataFrame(records)
    index = np.array(indices)
    blocks = {}
    for b in BLOCKS:
        v = arrays[b][index]
        history = v[:, :6]
        ref = history.mean(1)
        blocks[b] = dict(
            X=v[:, 6] - ref,
            Y1=v[:, 7] - ref,
            Y2=v[:, 8] - ref,
            rawX=v[:, 6],
            rawY=v[:, 7],
            history=history,
            V=((history - ref[:, None, :]) ** 2).sum(axis=(1, 2)) / 30.0,
        )
    for name in ["change_first_FE", "split_first_FE", "raw_first_FE"]:
        fit_save(output, name, model.sufficient(d, blocks, name))
    sel = np.flatnonzero(~d.event_content_id.duplicated())
    fit_save(
        output, "shared_unique_foci", model.sufficient(d, blocks, "change_first_FE", selected=sel)
    )
    pd.concat(provenance).to_csv(output / "window_provenance.csv", index=False)
    dump(
        output / "historical_summary.json",
        dict(
            windows=len(d),
            focal_works=d.event_content_id.nunique(),
            creators=d.creator_handle.nunique(),
        ),
    )
    return d, blocks, arrays, ids, index, m


def run(inputs, output):
    inputs, output = output_directory(inputs, output)
    period_change(inputs, output)
    consecutive(inputs, output)
    if (inputs / "window_members.csv").exists():
        historical(inputs, output)
    else:
        print(
            "Historical-reference diagnostic requires window_members.csv; see INPUTS.md.",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
