"""Transparent, full-vector Study 2 calculations; no dimension reduction."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import hashlib, itertools, json, math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BLOCKS = ["conceptual", "audiovisual", "embodied"]
DIMS = BLOCKS + ["joint"]
PERIODS = [f"{y}H{h}" for y in range(2020, 2027) for h in (1, 2) if (y, h) <= (2026, 1)]


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def joint(v):
    v = np.asarray(v)
    return np.concatenate([v, v.mean(axis=-1, keepdims=True)], axis=-1)


def load(inputs, strong=False, caption_only=False):
    src = Path(inputs)
    meta = pd.read_csv(src / "media_subset.csv", dtype={"content_id": str})
    if caption_only:
        if "caption_available" in meta:
            meta = meta[meta.caption_available.astype(str).str.lower().isin(["true", "1"])]
        elif "description" in meta:
            meta = meta[meta.description.fillna("").str.strip().ne("")]
        else:
            raise ValueError(
                "Fixed-cohort selection needs caption_available, description, or --coverage-inputs."
            )
    meta["created_at"] = pd.to_datetime(meta.created_at, utc=True, format="mixed")
    meta["period"] = meta.created_at.map(lambda t: f"{t.year}H{1 if t.month <= 6 else 2}")
    with np.load(src / "coordinates_three_block.npz", allow_pickle=False) as z:
        ids = z["content_id"].astype(str)
        blocks = [z["block_" + d].astype(float) for d in BLOCKS]
    assert len(set(ids)) == len(ids) and all(np.isfinite(x).all() for x in blocks)
    lookup = {v: i for i, v in enumerate(ids)}
    meta = meta[meta.content_id.isin(lookup) & meta.period.isin(PERIODS)]
    cells = {}
    rows = []
    limits = (8, 5, 60) if strong else (5, 3, 30)
    for (c, p), g in meta.groupby(["creator_handle", "period"]):
        g = g.sort_values(["created_at", "content_id"])
        span = (g.created_at.max() - g.created_at.min()).total_seconds() / 86400
        if len(g) < limits[0] or g.created_at.dt.date.nunique() < limits[1] or span < limits[2]:
            continue
        ix = [lookup[v] for v in g.content_id]
        xs = [x[ix] for x in blocks]
        cell = {"x": xs, "ids": list(g.content_id), "n": len(ix)}
        cell["mean"] = [x.mean(0) for x in xs]
        cell["noise"] = np.array(
            [np.sum((x - x.mean(0)) ** 2) / (len(x) * (len(x) - 1)) for x in xs]
        )
        cells[c, p] = cell
        rows.append(
            {
                "creator": c,
                "period": p,
                "n_videos": len(ix),
                "span_days": span,
                "ids": "|".join(g.content_id),
            }
        )
    return (
        cells,
        pd.DataFrame(rows),
        {
            "available_videos": len(ids),
            "creators": int(meta.creator_handle.nunique()),
            "cells": len(cells),
        },
    )


def fixed_cohort(a, b):
    rows = []
    answer = None
    for start in range(len(PERIODS)):
        ps = PERIODS[start:]
        cs = sorted(set(c for c, p in a) & set(c for c, p in b))
        cs = [c for c in cs if all((c, p) in a and (c, p) in b for p in ps)]
        ok = len(cs) >= 20 and len(ps) >= 6
        rows.append(
            {"start": ps[0], "end": ps[-1], "periods": len(ps), "creators": len(cs), "eligible": ok}
        )
        if answer is None and ok:
            answer = (ps, cs)
    if answer is None:
        candidate = max(
            (r for r in rows if r["periods"] >= 6), key=lambda r: (r["creators"], r["periods"])
        )
        ps = PERIODS[PERIODS.index(candidate["start"]) :]
        cs = [
            c
            for c in sorted({c for c, p in a} & {c for c, p in b})
            if all((c, p) in a and (c, p) in b for p in ps)
        ]
        answer = (ps, cs)
    return answer, pd.DataFrame(rows)


def comparisons(cells, fixed):
    specs = []
    for length, kind in [(2, "pair"), (3, "triple")]:
        for i in range(len(PERIODS) - length + 1):
            ps = PERIODS[i : i + length]
            cs = tuple(
                sorted(c for c, p in cells if p == ps[0] and all((c, t) in cells for t in ps))
            )
            if len(cs) >= 3:
                specs.append(
                    {"kind": kind, "periods": ps, "creators": cs, "id": kind + "_" + ps[0]}
                )
    specs.append({"kind": "fixed", "periods": fixed[0], "creators": tuple(fixed[1]), "id": "fixed"})
    groups = {p: set() for p in PERIODS}
    for s in specs:
        for p in s["periods"]:
            groups[p].add(s["creators"])
    return specs, {p: sorted(gs) for p, gs in groups.items() if gs}


def V(x):
    return np.sum((x - x.mean(0)) ** 2) / len(x)


def point_levels(cells, groups):
    out = {}
    for p, gs in groups.items():
        for cs in gs:
            raw = np.array([V(np.array([cells[c, p]["mean"][d] for c in cs])) for d in range(3)])
            noise = (1 - 1 / len(cs)) * np.mean([cells[c, p]["noise"] for c in cs], axis=0)
            out[p, cs] = {"V": joint(raw), "iid_corrected": joint(raw - noise), "T": joint(noise)}
    return out


def decomp(cells, p0, p1, cs):
    rows = []
    for d in range(3):
        x0 = np.array([cells[c, p0]["mean"][d] for c in cs])
        x1 = np.array([cells[c, p1]["mean"][d] for c in cs])
        a = x0 - x0.mean(0)
        e = x1 - x0
        e -= e.mean(0)
        C = 2 * np.mean(np.sum(a * e, axis=1))
        H = np.mean(np.sum(e * e, axis=1))
        dv = V(x1) - V(x0)
        assert abs(C + H - dv) < 2e-12
        t0 = (1 - 1 / len(cs)) * np.mean([cells[c, p0]["noise"][d] for c in cs])
        t1 = (1 - 1 / len(cs)) * np.mean([cells[c, p1]["noise"][d] for c in cs])
        rows.append([V(x0), V(x1), dv, C, H, t0, t1, C + 2 * t0, H - t0 - t1, dv + t0 - t1])
    return np.vstack([rows, np.mean(rows, axis=0)])


DECOMP_COLS = ["V0", "V1", "delta", "C", "H", "T0", "T1", "C_iid", "H_iid", "delta_iid"]


def circular_indices(m, n, draws, length, rng):
    starts = rng.integers(0, m, size=(draws, math.ceil(n / length)))
    return ((starts[:, :, None] + np.arange(length)) % m).reshape(draws, -1)[:, :n]


def weighted_means(x, idx):
    b, n = idx.shape
    w = csr_matrix(
        (np.full(b * n, 1 / n), (np.repeat(np.arange(b), n), idx.ravel())), shape=(b, len(x))
    )
    return w @ x


def sampling_plans(cells, draws, length, mode):
    plans = {}
    if mode == "bootstrap":
        for (c, p), a in cells.items():
            effective = min(length, max(1, a["n"] // 2))
            ix = circular_indices(
                a["n"], a["n"], draws, effective, np.random.default_rng(seed("boot", length, c, p))
            )
            plans[c, p] = (a["x"], ix)
    elif mode == "global_null":
        for c in sorted({c for c, p in cells}):
            ps = sorted(p for cc, p in cells if cc == c)
            xs = [np.vstack([cells[c, p]["x"][d] for p in ps]) for d in range(3)]
            n = sum(cells[c, p]["n"] for p in ps)
            ix = circular_indices(
                n, n, draws, length, np.random.default_rng(seed("global_null", length, c))
            )
            at = 0
            for p in ps:
                k = cells[c, p]["n"]
                plans[c, p] = (xs, ix[:, at : at + k])
                at += k
    else:
        raise ValueError(mode)
    return plans


def resampled_levels(cells, groups, draws, length, mode, log=print):
    plans = sampling_plans(cells, draws, length, mode)
    out = {}
    for p, gs in groups.items():
        needed = sorted(set().union(*map(set, gs)))
        membership = {c: [j for j, g in enumerate(gs) if c in g] for c in needed}
        counts = np.array([len(g) for g in gs])[:, None]
        vv = np.empty((len(gs), draws, 3))
        for lo in range(0, draws, 128):
            hi = min(lo + 128, draws)
            b = hi - lo
            for d in range(3):
                dim = plans[needed[0], p][0][d].shape[1]
                totals = np.zeros((len(gs), b, dim))
                sq = np.zeros((len(gs), b))
                for c in needed:
                    xs, ix = plans[c, p]
                    means = weighted_means(xs[d], ix[lo:hi])
                    norms = np.einsum("bd,bd->b", means, means)
                    for j in membership[c]:
                        totals[j] += means
                        sq[j] += norms
                vv[:, lo:hi, d] = sq / counts - np.einsum("gbd,gbd->gb", totals, totals) / counts**2
        for j, g in enumerate(gs):
            out[p, g] = joint(vv[j])
        log(f"  {mode} L={length}: {p}", flush=True)
    return out


def basic_interval(raw, draws):
    q = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "estimate": 2 * raw - draws.mean(0),
        "lo95": 2 * raw - q[1],
        "hi95": 2 * raw - q[0],
        "bootstrap_bias": draws.mean(0) - raw,
    }


def permutations(n, draws, rng):
    if math.factorial(n) <= draws or n <= 8 and draws >= 19999:
        return np.array(list(itertools.permutations(range(n)))), True
    return np.array([rng.permutation(n) for _ in range(draws)]), False


def tails(values, observed, exact=False):
    values = np.asarray(values)
    scale = np.maximum(np.max(abs(values), axis=0), np.abs(observed))
    tol = 100 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
    shift = 0 if exact else 1
    low = (shift + np.sum(values <= observed + tol, axis=0)) / (len(values) + shift)
    high = (shift + np.sum(values >= observed - tol, axis=0)) / (len(values) + shift)
    return low, high, np.minimum(1, 2 * np.minimum(low, high))


def assignment(cells, p0, p1, cs, draws=19999):
    n = len(cs)
    perm, exact = permutations(n, draws, np.random.default_rng(seed("assignment", p0, n)))
    values = np.empty((len(perm), 3))
    obs = []
    for d in range(3):
        x0 = np.array([cells[c, p0]["mean"][d] for c in cs])
        x1 = np.array([cells[c, p1]["mean"][d] for c in cs])
        a = x0 - x0.mean(0)
        e = x1 - x0
        e -= e.mean(0)
        h = np.mean(np.sum(e * e, axis=1))
        cross = 2 * a @ e.T / n
        values[:, d] = h + cross[np.arange(n)[None, :], perm].sum(1)
        obs.append(h + np.trace(cross))
    values = joint(values)
    obs = joint(obs)
    low, high, two = tails(values, obs, exact)
    q = np.quantile(values, [0.025, 0.975], axis=0, method="inverted_cdf")
    return [
        dict(
            dimension=d,
            observed=float(obs[j]),
            reference_mean=float(values[:, j].mean()),
            reference_lo95=float(q[0, j]),
            reference_hi95=float(q[1, j]),
            reference_lower=float(low[j]),
            reference_upper=float(high[j]),
            reference_two_sided=float(two[j]),
            draws=len(perm),
            exact=exact,
        )
        for j, d in enumerate(DIMS)
    ]


def split_check(cells, p0, p1, cs, mode, draws=499):
    weights = {}
    for c in cs:
        n = cells[c, p0]["n"]
        rng = np.random.default_rng(seed("split", p0, c, mode))
        wa = np.zeros((draws, n))
        wb = wa.copy()
        for r in range(draws):
            if mode == "random_half":
                ix = rng.permutation(n)
                aa = ix[: n // 2]
                bb = ix[n // 2 :]
            else:
                chunks = [np.arange(i, min(i + 3, n)) for i in range(0, n, 3)]
                # Merge a one-work tail so both halves contain at least two works.
                if len(chunks[-1]) == 1:
                    tail = chunks.pop()
                    chunks[-1] = np.r_[chunks[-1], tail]
                ix = rng.permutation(len(chunks))
                cut = len(chunks) // 2
                aa = np.concatenate([chunks[i] for i in ix[:cut]])
                bb = np.concatenate([chunks[i] for i in ix[cut:]])
            wa[r, aa] = 1 / len(aa)
            wb[r, bb] = 1 / len(bb)
        weights[c] = (wa, wb)
    vals = np.empty((draws, 3))
    for d in range(3):
        x1 = np.array([cells[c, p1]["mean"][d] for c in cs])
        x1 -= x1.mean(0)
        for lo in range(0, draws, 64):
            hi = min(lo + 64, draws)
            a = np.stack([weights[c][0][lo:hi] @ cells[c, p0]["x"][d] for c in cs], axis=1)
            b = np.stack([weights[c][1][lo:hi] @ cells[c, p0]["x"][d] for c in cs], axis=1)
            a -= a.mean(1, keepdims=True)
            b -= b.mean(1, keepdims=True)
            vals[lo:hi, d] = (
                np.einsum("bid,id->b", a + b, x1) - 2 * np.einsum("bid,bid->b", a, b)
            ) / len(cs)
    vals = joint(vals)
    q = np.quantile(vals, [0.025, 0.975], axis=0)
    return [
        dict(
            dimension=d,
            split_mode=mode,
            mean=float(vals[:, j].mean()),
            lo95_splits=float(q[0, j]),
            hi95_splits=float(q[1, j]),
            negative_fraction=float((vals[:, j] < 0).mean()),
            draws=draws,
        )
        for j, d in enumerate(DIMS)
    ]


def local_stationary(cells, p0, p1, cs, length, draws=499, nperm=999):
    n = len(cs)
    plan = {}
    for c in cs:
        a, b = cells[c, p0], cells[c, p1]
        xs = [np.vstack([a["x"][d], b["x"][d]]) for d in range(3)]
        ix = circular_indices(
            a["n"] + b["n"],
            a["n"] + b["n"],
            draws,
            length,
            np.random.default_rng(seed("local_null", p0, c, length)),
        )
        plan[c] = (xs, ix[:, : a["n"]], ix[:, a["n"] :])
    perm, exact = permutations(n, nperm, np.random.default_rng(seed("local_permutation", p0, n)))
    cc = np.empty((draws, 3))
    hh = cc.copy()
    rates = np.empty((draws, 4))
    for lo in range(0, draws, 32):
        hi = min(lo + 32, draws)
        nb = hi - lo
        reference = np.empty((nb, len(perm), 3))
        for d in range(3):
            x0 = np.stack([weighted_means(plan[c][0][d], plan[c][1][lo:hi]) for c in cs], axis=1)
            x1 = np.stack([weighted_means(plan[c][0][d], plan[c][2][lo:hi]) for c in cs], axis=1)
            a = x0 - x0.mean(1, keepdims=True)
            e = x1 - x0
            e -= e.mean(1, keepdims=True)
            cv = 2 * np.einsum("bid,bid->b", a, e) / n
            hv = np.einsum("bid,bid->b", e, e) / n
            cc[lo:hi, d] = cv
            hh[lo:hi, d] = hv
            cross = 2 * np.matmul(a, e.transpose(0, 2, 1)) / n
            reference[:, :, d] = hv[:, None] + sum(cross[:, i, perm[:, i]] for i in range(n))
        reference = joint(reference)
        obs = joint(cc[lo:hi] + hh[lo:hi])
        tol = 100 * np.finfo(float).eps * np.maximum(np.max(abs(reference), axis=1), abs(obs))
        shift = 0 if exact else 1
        rates[lo:hi] = (shift + (reference <= obs[:, None, :] + tol[:, None, :]).sum(1)) / (
            len(perm) + shift
        )
    c = joint(cc)
    dv = joint(cc + hh)
    observed = decomp(cells, p0, p1, cs)
    ld, ud, td = tails(dv, observed[:, 2])
    lc, uc, tc = tails(c, observed[:, 3])
    qd = np.quantile(dv, [0.025, 0.975], axis=0)
    qc = np.quantile(c, [0.025, 0.975], axis=0)
    return [
        dict(
            dimension=d,
            block_length=length,
            draws=draws,
            permutations=len(perm),
            null_delta_mean=float(dv[:, j].mean()),
            null_delta_lo95=float(qd[0, j]),
            null_delta_hi95=float(qd[1, j]),
            observed_delta=float(observed[j, 2]),
            delta_reference_two_sided=float(td[j]),
            null_C_mean=float(c[:, j].mean()),
            null_C_lo95=float(qc[0, j]),
            null_C_hi95=float(qc[1, j]),
            observed_C=float(observed[j, 3]),
            C_reference_two_sided=float(tc[j]),
            null_C_negative_rate=float((c[:, j] < 0).mean()),
            null_reassignment_lower025_rate=float((rates[:, j] <= 0.025).mean()),
        )
        for j, d in enumerate(DIMS)
    ]
