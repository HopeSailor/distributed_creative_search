"""Scalar vector regressions with creator intercepts and cluster inference."""

import numpy as np
from scipy.stats import t, f

BLOCKS = ["conceptual", "audiovisual", "embodied"]
PAIRS = ["conceptual-audiovisual", "embodied-audiovisual", "conceptual-embodied"]
RPAIRS = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0], [1.0, 0.0, -1.0]])
BOOT_DRAWS, WILD_DRAWS = 9999, 4999


def holm(values):
    p = np.asarray(values, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    adj = np.maximum.accumulate([(len(order) - i) * p[j] for i, j in enumerate(order)])
    result[order] = np.minimum(adj, 1)
    return result


def vectors(block, model, selected):
    if model.startswith("raw"):
        return block["rawX"][selected], block["rawY"][selected]
    return block["X"][selected], block["Y1"][selected]


def sufficient(d, blocks, model, selected=None, balanced=False):
    if selected is None:
        selected = np.arange(len(d))
    sub = d.iloc[selected]
    groups = sorted(sub.creator_handle.unique())
    H, N, Q = np.zeros((len(groups), 3)), np.zeros((len(groups), 3)), np.zeros((len(groups), 3))
    counts = np.array([sub.creator_handle.eq(g).sum() for g in groups])
    for gi, creator in enumerate(groups):
        idx = selected[sub.creator_handle.eq(creator).to_numpy()]
        for bi, b in enumerate(BLOCKS):
            X, Y = vectors(blocks[b], model, idx)
            if model.endswith("_FE"):
                X, Y = X - X.mean(axis=0), Y - Y.mean(axis=0)
            h, n, q = np.sum(X * X), np.sum(X * Y), np.sum(Y * Y)
            if model.startswith("split"):
                v = blocks[b]["V"][idx].sum()
                h, n, q = h + v, n - v, q + v
            scale = 1 / len(idx) if balanced else 1.0
            H[gi, bi], N[gi, bi], Q[gi, bi] = h * scale, n * scale, q * scale
    assert np.all(H.sum(axis=0) > 1e-10)
    return dict(H=H, N=N, Q=Q, groups=groups, counts=counts)


def estimate(S):
    H, N = S["H"], S["N"]
    total = H.sum(axis=0)
    beta = N.sum(axis=0) / total
    G = len(H)
    influence = (N - H * beta) / total
    cov = G / (G - 1) * (influence.T @ influence)
    return dict(
        beta=beta, cov=cov, G=G, H=total, effective_creators=1 / ((H / total) ** 2).sum(axis=0)
    )


def boot(S, seed=202610231):
    G = len(S["H"])
    counts = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=BOOT_DRAWS)
    return (counts @ S["N"]) / (counts @ S["H"])


def wald(beta, cov, R, G):
    theta, cv = R @ beta, R @ cov @ R.T
    assert np.linalg.matrix_rank(cv) == len(R)
    value = float(theta @ np.linalg.solve(cv, theta) / len(R))
    return dict(F=value, df_num=len(R), df_den=G - 1, p_cluster=float(f.sf(value, len(R), G - 1)))


def wild(S, R, seed=202610232):
    fitted = estimate(S)
    H, N, G = S["H"], S["N"], fitted["G"]
    bread = np.diag(1 / fitted["H"])
    beta0 = fitted["beta"] - bread @ R.T @ np.linalg.solve(R @ bread @ R.T, R @ fitted["beta"])
    assert abs(R @ beta0).max() < 1e-10
    U = N - H * beta0
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(WILD_DRAWS, G))
    delta = (signs @ U) / fitted["H"]
    influence = (signs[:, :, None] * U[None, :, :] - H[None, :, :] * delta[:, None, :]) / fitted[
        "H"
    ]
    projected = influence @ R.T
    cov = G / (G - 1) * np.einsum("bgq,bgr->bqr", projected, projected)
    theta = delta @ R.T
    values = np.einsum("bq,bqr,br->b", theta, np.linalg.pinv(cov, hermitian=True), theta) / len(R)
    observed = wald(fitted["beta"], fitted["cov"], R, G)
    result = dict(
        **observed,
        p_wild=float((1 + (values >= observed["F"] - 1e-10).sum()) / (WILD_DRAWS + 1)),
        wild_draws=WILD_DRAWS,
    )
    return result, dict(beta0=beta0, signs=signs[:4], values=values[:4])


def output_model(model, S, do_wild=True):
    fitted, draws = estimate(S), boot(S)
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    rows, contrasts = [], []
    for bi, b in enumerate(BLOCKS):
        se = np.sqrt(fitted["cov"][bi, bi])
        critical = t.ppf(0.975, fitted["G"] - 1)
        rows.append(
            dict(
                model=model,
                dimension=b,
                beta=fitted["beta"][bi],
                lo95_boot=ci[0, bi],
                hi95_boot=ci[1, bi],
                se_cluster=se,
                lo95_cluster=fitted["beta"][bi] - critical * se,
                hi95_cluster=fitted["beta"][bi] + critical * se,
                events=int(S["counts"].sum()),
                creators=fitted["G"],
                effective_creators=fitted["effective_creators"][bi],
                within_R2=(S["N"][:, bi].sum() ** 2) / (S["H"][:, bi].sum() * S["Q"][:, bi].sum()),
            )
        )
    for label, r in zip(PAIRS, RPAIRS):
        interval = np.quantile(draws @ r, [0.025, 0.975])
        result = dict(
            model=model,
            contrast=label,
            estimate=float(r @ fitted["beta"]),
            lo95_boot=interval[0],
            hi95_boot=interval[1],
        )
        if do_wild:
            result.update(wild(S, r[None, :])[0])
        contrasts.append(result)
    if do_wild:
        for r, adj in zip(contrasts, holm([r["p_wild"] for r in contrasts])):
            r["p_holm_wild"] = float(adj)
    joint = dict(model=model, **wild(S, RPAIRS[:2])[0]) if do_wild else {}
    return rows, contrasts, joint, draws
