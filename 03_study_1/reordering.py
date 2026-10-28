"""Sequence reordering reference for Supplementary Section S2.4."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from common import load_module

study = load_module("study_one_runner", Path(__file__).with_name("run.py"))
from vector_model import sufficient, estimate, BLOCKS, PAIRS, RPAIRS
from common import dump, output_directory

ORDER_DRAWS = 999
MODELS = ["change_first_FE", "split_first_FE", "raw_first_FE"]


def calibration(meta, output, d, arrays, ids, index, stats):
    lookup = {cid: i for i, cid in enumerate(ids)}
    meta = meta.loc[meta.content_id.isin(lookup)].drop_duplicates("content_id")
    assert len(meta) == len(ids)
    G = len(stats["change_first_FE"]["groups"])
    Hnull = {m: np.zeros((ORDER_DRAWS, G, 3)) for m in MODELS}
    Nnull = {m: np.zeros((ORDER_DRAWS, G, 3)) for m in MODELS}
    max_gram_error = 0.0
    rng = np.random.default_rng(202610233)

    def unpack(values):
        h, n, v, rawh, rawn = values.T
        return {
            "change_first_FE": (h, n),
            "split_first_FE": (h + v, n - v),
            "raw_first_FE": (rawh, rawn),
        }

    for gi, creator in enumerate(stats["change_first_FE"]["groups"]):
        selected = np.flatnonzero(d.creator_handle.eq(creator).to_numpy())
        global_pool = np.array(
            [
                lookup[cid]
                for cid in meta.loc[meta.creator_handle.eq(creator)]
                .sort_values(["created_at", "content_id"])
                .content_id
            ]
        )
        local_lookup = {global_i: local_i for local_i, global_i in enumerate(global_pool)}
        local_windows = np.array([[local_lookup[j] for j in w] for w in index[selected]])
        unique, inverse = np.unique(local_windows, return_inverse=True)
        win = inverse.reshape(local_windows.shape)
        n, u = len(win), len(unique)
        ref, focal, next1 = [np.zeros((n, u)) for _ in range(3)]
        for e in range(n):
            ref[e, win[e, :6]] = 1 / 6
            focal[e, win[e, 6]] = 1
            next1[e, win[e, 7]] = 1
        X, Y1 = focal - ref, next1 - ref
        xc, y1c = X - X.mean(axis=0), Y1 - Y1.mean(axis=0)
        fc, nc = focal - focal.mean(axis=0), next1 - next1.mean(axis=0)
        Qv = (np.diag(ref.sum(axis=0) * 6) - 6 * (ref.T @ ref)) / 30
        Q = np.array([xc.T @ xc, xc.T @ y1c, Qv, fc.T @ fc, fc.T @ nc])
        permutations = np.array([rng.permutation(len(global_pool)) for _ in range(ORDER_DRAWS)])[
            :, unique
        ]
        for bi, b in enumerate(BLOCKS):
            x = arrays[b][global_pool].astype(float)
            K = x @ x.T
            real = np.einsum("uv,quv->q", K[unique[:, None], unique[None, :]], Q)[None, :]
            for m, (h, numer) in unpack(real).items():
                max_gram_error = max(
                    max_gram_error,
                    abs(h[0] - stats[m]["H"][gi, bi]),
                    abs(numer[0] - stats[m]["N"][gi, bi]),
                )
            values = []
            for start in range(0, ORDER_DRAWS, 64):
                p = permutations[start : start + 64]
                grams = K[p[:, :, None], p[:, None, :]]
                values.append(np.einsum("buv,quv->bq", grams, Q, optimize=True))
            values = np.concatenate(values)
            for m, (h, numer) in unpack(values).items():
                Hnull[m][:, gi, bi], Nnull[m][:, gi, bi] = h, numer
        if (gi + 1) % 9 == 0:
            print(f"Sequence reordering: {gi + 1}/{G} creators", flush=True)
    assert max_gram_error < 1e-7, max_gram_error
    summaries, draw_rows = [], []
    for model in ["change_first_FE", "split_first_FE", "raw_first_FE"]:
        beta_null = Nnull[model].sum(axis=1) / Hnull[model].sum(axis=1)
        beta_obs = estimate(stats[model])["beta"]
        assert np.isfinite(beta_null).all()
        for i, beta in enumerate(beta_null):
            draw_rows.append(dict(model=model, draw=i + 1, **dict(zip(BLOCKS, beta))))
        for label, observed, reference in [
            (b, beta_obs[j], beta_null[:, j]) for j, b in enumerate(BLOCKS)
        ] + [(name, float(r @ beta_obs), beta_null @ r) for name, r in zip(PAIRS, RPAIRS)]:
            summaries.append(
                dict(
                    model=model,
                    statistic=label,
                    observed=float(observed),
                    reorder_mean=float(reference.mean()),
                    reorder_lo95=float(np.quantile(reference, 0.025)),
                    reorder_hi95=float(np.quantile(reference, 0.975)),
                    observed_percentile=float(np.mean(reference <= observed)),
                    draws=ORDER_DRAWS,
                )
            )
    pd.DataFrame(summaries).to_csv(output / "order_calibration.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(output / "order_calibration_draws.csv", index=False)
    return dict(
        draws=ORDER_DRAWS, max_direct_vs_Gram_moment_error=float(max_gram_error), seed=202610233
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    a = p.parse_args()
    inp, out = output_directory(a.inputs, a.outputs)
    d, blocks, arrays, ids, index, meta = study.historical(inp, out)
    stats = {m: sufficient(d, blocks, m) for m in MODELS}
    dump(out / "reordering_verification.json", calibration(meta, out, d, arrays, ids, index, stats))
