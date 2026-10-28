"""Strictly prior personal and collective histories with a common cutoff."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, hashlib
import numpy as np
import pandas as pd
from common import load_vectors, distance_cache

BLOCKS = ["conceptual", "audiovisual", "embodied", "joint"]
QUANTILES = {"90": 0.9, "95": 0.95, "975": 0.975}
SEED = 20261024
TOL = 1e-8


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def read(path):
    return pd.read_csv(path, dtype={"content_id": str, "follow1_id": str, "follow2_id": str})


def truth(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def weighted_quantile(values, names, q):
    _, inv, counts = np.unique(names, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inv]
    ix = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[ix]) / weights.sum()
    return float(values[ix[min(np.searchsorted(cdf, q, side="left"), len(ix) - 1)]])


def outcome_status(values, threshold):
    observed = np.asarray([v for v in values if v is not None], dtype=float)
    if len(observed) and np.any(observed <= threshold + TOL):
        return "at_least_one_observed_return"
    if len(observed) == 2:
        return "both_followups_outside"
    return "unresolved_missing_followup"


class Data:
    def __init__(self, inputs, cache):
        self.name = "final"
        self.src = Path(inputs)
        self.ids, self.meta, self.X = load_vectors(self.src, include_joint=True)
        self.meta["time"] = pd.to_datetime(self.meta.created_at, utc=True, format="mixed")
        self.meta["row"] = np.arange(len(self.meta))
        self.times = self.meta.time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.names = self.meta.creator_handle.to_numpy(dtype=str)
        self.D = distance_cache(self.X, cache)
        self.raw = read(self.src / "media_subset.csv")
        self.raw_lookup = {
            (r.creator_handle, int(r.creator_sequence_index)): str(r.content_id)
            for r in self.raw.itertuples()
        }
        self.inventory_times = {}
        p = self.src / "publication_inventory.csv"
        if p.exists():
            inv = read(p).drop_duplicates("content_id")
            for c, g in inv.groupby("creator_handle"):
                self.inventory_times[c] = np.sort(
                    pd.to_datetime(g.created_at, utc=True, format="mixed")
                    .to_numpy(dtype="datetime64[ns]")
                    .astype(np.int64)
                )

    def events(self):
        events, eligibility = [], []
        cutoff = pd.Timestamp("2021-12-31T23:59:59Z").value
        for c, g in self.meta.groupby("creator_handle"):
            g = g.sort_values("creator_sequence_index").reset_index(drop=True)
            seq = g.creator_sequence_index.to_numpy(dtype=int)
            lookup = dict(zip(seq, g.row.to_numpy(dtype=int)))
            for j in range(6, len(g)):
                if not np.all(np.diff(seq[j - 6 : j + 1]) == 1):
                    continue
                rows = g.row.iloc[j - 6 : j + 1].to_numpy(dtype=int)
                hi, focal = rows[:6], int(rows[6])
                t = self.times[focal]
                reason = []
                if t <= cutoff:
                    reason.append("not_after_scaler_fit")
                if np.any(self.times[hi] >= t) or np.any(np.diff(self.times[rows]) < 0):
                    reason.append("historical_chronology")
                bank = np.flatnonzero(self.times < t)
                if len(bank) < 50:
                    reason.append("bank_under_50")
                if len(set(self.names[bank]) - {c}) < 5:
                    reason.append("fewer_than_five_other_creators")
                if not len(bank) or (t - self.times[bank].min()) / 86400e9 < 365:
                    reason.append("history_span_under_365_days")
                center = self.X["joint"][hi].mean(0)
                delta = self.X["joint"][focal] - center
                m2 = float(delta @ delta)
                if m2 <= 1e-12:
                    reason.append("zero_personal_displacement")
                eligibility.append(
                    dict(
                        content_id=self.ids[focal],
                        creator=c,
                        time=str(self.meta.time.iloc[focal]),
                        eligible=not reason,
                        reasons="|".join(reason),
                    )
                )
                if reason:
                    continue
                follow, follow_reason = [], []
                for q in [int(seq[j]) + 1, int(seq[j]) + 2]:
                    r = lookup.get(q)
                    why = ""
                    if r is None:
                        why = (
                            "features_unavailable" if (c, q) in self.raw_lookup else "not_in_sample"
                        )
                    elif self.times[r] <= t:
                        r, why = None, "followup_chronology"
                    follow.append(r)
                    follow_reason.append(why)
                if all(r is not None for r in follow):
                    assert self.times[follow[0]] <= self.times[follow[1]]
                b2 = np.array(
                    [
                        np.sum((self.X[b][focal] - self.X[b][hi].mean(0)) ** 2) / 3
                        for b in BLOCKS[:3]
                    ]
                )
                shares = b2 / m2
                assert abs(shares.sum() - 1) < 1e-6
                event = dict(
                    content_id=self.ids[focal],
                    creator=c,
                    time=str(self.meta.time.iloc[focal]),
                    focal_row=focal,
                    history_rows=hi,
                    follow_rows=follow,
                    M=float(np.sqrt(m2)),
                    S=float(shares @ shares),
                    history_ids="|".join(self.ids[hi]),
                    seq=int(seq[j]),
                    window_first_seq=int(seq[j - 6]),
                    both_follow_available=all(r is not None for r in follow),
                )
                for k, r in enumerate(follow, 1):
                    event[f"follow{k}_id"] = self.ids[r] if r is not None else ""
                    event[f"follow{k}_available"] = r is not None
                    event[f"follow{k}_missing_reason"] = follow_reason[k - 1]
                    event[f"P{k}"] = (
                        float((self.X["joint"][r] - center) @ delta / m2) if r is not None else None
                    )
                for b, share in zip(BLOCKS[:3], shares):
                    event["q_" + b] = float(share)
                events.append(event)
        events.sort(key=lambda e: (e["time"], e["content_id"]))
        last = {}
        for e in events:
            keep = e["window_first_seq"] > last.get(e["creator"], -1)
            e["nonoverlap_seven_work"] = keep
            if keep:
                last[e["creator"]] = e["seq"]
        assert len({e["content_id"] for e in events}) == len(events)
        return events, pd.DataFrame(eligibility)

    def evaluate(self, events):
        by_row = {e["focal_row"]: e for e in events}
        rng = np.random.default_rng(SEED)
        check_rows = set(
            rng.choice(list(by_row), size=min(24, len(by_row)), replace=False).tolist()
        )
        hist_nearest = {b: np.full(len(self.ids), np.inf, dtype=np.float32) for b in BLOCKS}
        bank = []
        outputs, checks = [], []
        order = np.argsort(self.times, kind="stable")
        batches = np.split(order, np.flatnonzero(np.diff(self.times[order])) + 1)
        for batch in batches:
            past = np.asarray(bank, dtype=int)
            for focal in batch:
                if focal not in by_row:
                    continue
                e = by_row[focal]
                assert len(past) >= 50 and (self.times[past] < self.times[focal]).all()
                own = past[self.names[past] == e["creator"]]
                peers = past[self.names[past] != e["creator"]]
                assert len(own) >= 6 and len(set(self.names[peers])) >= 5
                row = {
                    k: v
                    for k, v in e.items()
                    if k not in ["history_rows", "follow_rows", "focal_row"]
                }
                row.update(
                    bank_n=len(past),
                    own_bank_n=len(own),
                    peer_bank_n=len(peers),
                    bank_creators=len(set(self.names[past])),
                    bank_ids_sha256=hashlib.sha256(
                        "\n".join(sorted(self.ids[past])).encode()
                    ).hexdigest(),
                )
                inventory_n = sum(
                    np.searchsorted(ts, self.times[focal], side="left")
                    for ts in self.inventory_times.values()
                )
                row["inventory_prior_n"] = int(inventory_n)
                row["observed_inventory_fraction"] = (
                    len(past) / inventory_n if inventory_n else None
                )
                for b in BLOCKS:
                    matrix = self.D[b]
                    values = hist_nearest[b][past]
                    thresholds = {
                        k: weighted_quantile(values, self.names[past], q)
                        for k, q in QUANTILES.items()
                    }
                    d_own = float(matrix[focal, own].min())
                    d_peer = float(matrix[focal, peers].min())
                    d_group = min(d_own, d_peer)
                    assert d_group <= d_own and d_group <= d_peer
                    near = past[np.argsort(matrix[focal, past], kind="stable")[:3]]
                    own_near = own[np.argmin(matrix[focal, own])]
                    peer_near = peers[np.argmin(matrix[focal, peers])]
                    prefix = b + "_"
                    row.update(
                        {
                            prefix + k: v
                            for k, v in dict(
                                D_own=d_own,
                                D_peer=d_peer,
                                D_group=d_group,
                                own_ratio=d_own / thresholds["95"],
                                group_ratio=d_group / thresholds["95"],
                                nearest_ids="|".join(self.ids[near]),
                                nearest_creators="|".join(self.names[near]),
                                nearest_own_id=self.ids[own_near],
                                nearest_peer_id=self.ids[peer_near],
                            ).items()
                        }
                    )
                    fdist = [
                        float(matrix[r, past].min()) if r is not None else None
                        for r in e["follow_rows"]
                    ]
                    for k, value in enumerate(fdist, 1):
                        row[prefix + f"D_follow{k}"] = value
                        row[prefix + f"follow{k}_ratio"] = (
                            value / thresholds["95"] if value is not None else None
                        )
                    for key, threshold in thresholds.items():
                        own_flag = d_own > threshold + TOL
                        group_flag = d_group > threshold + TOL
                        assert not group_flag or own_flag
                        row[prefix + "tau" + key] = threshold
                        row[prefix + "own_flag" + key] = own_flag
                        row[prefix + "candidate" + key] = group_flag
                        row[prefix + "category" + key] = (
                            "outside_collective_history"
                            if group_flag
                            else "own_departure_covered_by_peers"
                            if own_flag
                            else "within_own_history"
                        )
                        row[prefix + "follow_status" + key] = outcome_status(fdist, threshold)
                        row[prefix + "both_follow_outside" + key] = (
                            all(v > threshold + TOL for v in fdist)
                            if all(v is not None for v in fdist)
                            else None
                        )
                    if b == "joint":
                        same = None
                        if e["both_follow_available"]:
                            same = (
                                row[prefix + "both_follow_outside95"]
                                and all(
                                    matrix[r, focal] < fdist[k]
                                    for k, r in enumerate(e["follow_rows"])
                                )
                                and all(e[f"P{k}"] > 0.5 for k in [1, 2])
                            )
                        row["joint_same_extension95"] = bool(same) if same is not None else None
                    # Independent direct submatrix calibration on sampled foci and all dimensional candidates.
                    if focal in check_rows or row[prefix + "candidate95"]:
                        a = np.array(matrix[np.ix_(past, past)], copy=True)
                        np.fill_diagonal(a, np.inf)
                        direct = a.min(1)
                        assert np.array_equal(direct, values)
                        for key, q in QUANTILES.items():
                            expected = weighted_quantile(direct, self.names[past], q)
                            assert np.isclose(expected, thresholds[key], atol=1e-7, rtol=0)
                        dv = np.linalg.norm(self.X[b][focal] - self.X[b][near[0]])
                        err = abs(float(dv) - d_group)
                        assert err < 5e-7, (e["content_id"], b, err)
                        checks.append(
                            dict(
                                content_id=e["content_id"],
                                dimension=b,
                                bank_n=len(past),
                                direct_vector_error=err,
                            )
                        )
                outputs.append(row)
                if len(outputs) % 200 == 0:
                    print(self.name, "scored", len(outputs), "/", len(events), flush=True)
            # All works at this timestamp are added only after scoring all focal works in the batch.
            for r in batch:
                previous = np.asarray(bank, dtype=int)
                for b in BLOCKS:
                    if len(previous):
                        values = self.D[b][r, previous]
                        hist_nearest[b][previous] = np.minimum(hist_nearest[b][previous], values)
                        hist_nearest[b][r] = values.min()
                bank.append(int(r))
        assert len(outputs) == len(events)
        return pd.DataFrame(outputs), pd.DataFrame(checks)


def summarize(d, label):
    rows = []
    for b in BLOCKS:
        for key in QUANTILES:
            flags = truth(d[b + "_candidate" + key])
            own = truth(d[b + "_own_flag" + key])
            cand = d[flags]
            complete = cand[cand.both_follow_available]
            status = cand[b + "_follow_status" + key].value_counts()
            confirmed = int(status.get("both_followups_outside", 0))
            unresolved = int(status.get("unresolved_missing_followup", 0))
            n = int(flags.sum())
            rows.append(
                dict(
                    sample=label,
                    dimension=b,
                    percentile=key,
                    focal_works=len(d),
                    focal_creators=d.creator.nunique(),
                    within_own=int((~own).sum()),
                    outside_own=int(own.sum()),
                    covered_by_peers=int((own & ~flags).sum()),
                    collective_candidates=n,
                    candidate_creators=cand.creator.nunique(),
                    pooled_share=float(flags.mean()),
                    creator_balanced_share=float(
                        d.assign(flag=flags).groupby("creator").flag.mean().mean()
                    ),
                    candidate_first_follow_available=int(cand.follow1_available.sum()),
                    candidate_both_follow_available=len(complete),
                    confirmed_both_follow_outside=confirmed,
                    observed_at_least_one_return=int(status.get("at_least_one_observed_return", 0)),
                    unresolved_followup=unresolved,
                    all_candidate_retention_lower=confirmed / n if n else None,
                    all_candidate_retention_upper=(confirmed + unresolved) / n if n else None,
                )
            )
    return pd.DataFrame(rows)
