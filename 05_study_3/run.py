"""Study 3: personal departures, collective precedents and subsequent returns."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from common import dump, output_directory
from historical import Data, summarize, TOL, BLOCKS


def return_destinations(data, events, output):
    lookup = {v: i for i, v in enumerate(data.ids)}
    rows = []
    for e in events[events.joint_candidate95].itertuples():
        focal = lookup[e.content_id]
        bank = np.flatnonzero(data.times < data.times[focal])
        own = bank[data.names[bank] == e.creator]
        peers = bank[data.names[bank] != e.creator]
        found = False
        for k in [1, 2]:
            if not getattr(e, f"follow{k}_available"):
                continue
            r = lookup[getattr(e, f"follow{k}_id")]
            do = float(data.D["joint"][r, own].min())
            dp = float(data.D["joint"][r, peers].min())
            group = min(do, dp) <= e.joint_tau95 + TOL
            personal = do <= e.joint_tau95 + TOL
            first = group and not found
            rows.append(
                dict(
                    focal_id=e.content_id,
                    creator=e.creator,
                    follow_position=k,
                    follow_id=data.ids[r],
                    cutoff=e.joint_tau95,
                    own_distance=do,
                    peer_distance=dp,
                    group_return=group,
                    own_return=personal,
                    peer_close=dp <= e.joint_tau95 + TOL,
                    category="within_personal_history"
                    if personal
                    else "peer_precedent_only"
                    if group
                    else "outside_collective_history",
                    earliest_observed_return=first,
                )
            )
            found = found or group
    follow = pd.DataFrame(rows)
    follow.to_csv(output / "all_available_followups.csv", index=False)
    first = follow[follow.earliest_observed_return] if len(follow) else follow
    first.to_csv(output / "earliest_observed_returns.csv", index=False)
    dump(
        output / "return_summary.json",
        dict(
            observed_returns=len(first),
            earliest_return_categories=first.category.value_counts().to_dict()
            if len(first)
            else {},
        ),
    )


def run(inputs, output):
    inputs, output = output_directory(inputs, output)
    data = Data(inputs, output / "cache")
    events, eligibility = data.events()
    d, checks = data.evaluate(events)
    d.to_csv(output / "event_metrics.csv", index=False)
    eligibility.to_csv(output / "eligibility.csv", index=False)
    checks.to_csv(output / "independent_distance_checks.csv", index=False)
    pd.concat(
        [
            summarize(d, "all_past_eligible"),
            summarize(d[d.nonoverlap_seven_work], "nonoverlap_seven_work"),
            summarize(d[d.both_follow_available], "both_followups_observed"),
        ]
    ).to_csv(output / "sensitivity_summary.csv", index=False)
    d[d.joint_candidate95].to_csv(output / "candidate_extensions.csv", index=False)
    q = d.copy()
    q["personal_change_quartile"] = pd.qcut(q.M, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    q.groupby("personal_change_quartile", observed=True).agg(
        focals=("content_id", "size"),
        median_M=("M", "median"),
        outside_own=("joint_own_flag95", "sum"),
        collective_candidates=("joint_candidate95", "sum"),
        median_group_ratio=("joint_group_ratio", "median"),
    ).to_csv(output / "personal_change_quartiles.csv")
    conditional = []
    for b in BLOCKS:
        own = d[b + "_own_flag95"]
        group = d[b + "_candidate95"]
        conditional.append(
            dict(
                dimension=b,
                personal_departures=int(own.sum()),
                collective_departures=int(group.sum()),
                peer_covered=int((own & ~group).sum()),
                collective_share_given_personal=float(group.sum() / own.sum())
                if own.any()
                else None,
            )
        )
    pd.DataFrame(conditional).to_csv(output / "dimensional_conditional_departures.csv", index=False)
    return_destinations(data, d, output)
    dump(
        output / "coverage.json",
        dict(
            complete_videos=len(data.ids),
            historical_creators=len(set(data.names)),
            focal_works=len(d),
            focal_creators=d.creator.nunique(),
            excluded_focal_creators=sorted(set(data.names) - set(d.creator)),
            date_min=d.time.min(),
            date_max=d.time.max(),
        ),
    )
    print(
        f"Study 3 complete: {len(d)} eligible works; {int(d.joint_candidate95.sum())} collective departures.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
