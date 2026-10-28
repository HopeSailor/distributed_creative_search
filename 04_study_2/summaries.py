from core import *


def save_draws(out, stem, draws):
    mapping = []
    arrays = {}
    for i, ((p, cs), v) in enumerate(draws.items()):
        k = f"g{i:03d}"
        arrays[k] = v
        mapping.append({"key": k, "period": p, "creators": list(cs)})
    dump(out / (stem + "_keys.json"), mapping)
    np.savez_compressed(out / (stem + ".npz"), **arrays)


def summarize_bootstrap(cells, specs, levels, boots, out, length):
    pairs = []
    triples = []
    fixedrows = []
    for s in specs:
        ps, cs = s["periods"], s["creators"]
        n = len(cs)
        if s["kind"] == "pair":
            raw = levels[ps[1], cs]["V"] - levels[ps[0], cs]["V"]
            bs = boots[ps[1], cs] - boots[ps[0], cs]
            ci = basic_interval(raw, bs)
            dec = decomp(cells, *ps, cs)
            for j, d in enumerate(DIMS):
                pairs.append(
                    dict(
                        start=ps[0],
                        end=ps[1],
                        n=n,
                        creators="|".join(cs),
                        dimension=d,
                        block_length=length,
                        **dict(zip(DECOMP_COLS, dec[j])),
                        **{k: float(v[j]) for k, v in ci.items()},
                    )
                )
        elif s["kind"] == "triple":
            v0, v1, v2 = [levels[p, cs]["V"] for p in ps]
            b0, b1, b2 = [boots[p, cs] for p in ps]
            ca = basic_interval(v1 - v0, b1 - b0)
            cb = basic_interval(v2 - v0, b2 - b0)
            for j, d in enumerate(DIMS):
                row = dict(
                    start=ps[0],
                    middle=ps[1],
                    end=ps[2],
                    n=n,
                    creators="|".join(cs),
                    dimension=d,
                    block_length=length,
                    raw01=float(v1[j] - v0[j]),
                    raw02=float(v2[j] - v0[j]),
                )
                row.update({k + "_01": float(v[j]) for k, v in ca.items()})
                row.update({k + "_02": float(v[j]) for k, v in cb.items()})
                row["raw_same_side"] = bool(row["raw01"] * row["raw02"] > 0)
                row["adjusted_same_side"] = bool(row["estimate_01"] * row["estimate_02"] > 0)
                # Both marginal intervals have the same sign; do not call this a joint 95% test.
                row["both_marginal_intervals_same_side"] = bool(
                    (row["lo95_01"] > 0 and row["lo95_02"] > 0)
                    or (row["hi95_01"] < 0 and row["hi95_02"] < 0)
                )
                triples.append(row)
        else:
            for p in ps:
                raw = levels[p, cs]["V"]
                ci = basic_interval(raw, boots[p, cs])
                for j, d in enumerate(DIMS):
                    fixedrows.append(
                        dict(
                            period=p,
                            n=n,
                            dimension=d,
                            block_length=length,
                            raw=float(raw[j]),
                            iid_corrected=float(levels[p, cs]["iid_corrected"][j]),
                            **{k: float(v[j]) for k, v in ci.items()},
                        )
                    )
    for name, rows in [
        ("transitions", pairs),
        ("retention", triples),
        ("fixed_trajectory", fixedrows),
    ]:
        pd.DataFrame(rows).to_csv(out / f"{name}_L{length}.csv", index=False)


def global_diagnostics(cells, specs, levels, draws, length, out):
    records = []
    counts = np.zeros((999, 4), dtype=int)
    observed_count = np.zeros(4, dtype=int)
    for s in specs:
        if s["kind"] == "fixed":
            continue
        ps, cs = s["periods"], s["creators"]
        v = [levels[p, cs]["V"] for p in ps]
        b = [draws[p, cs] for p in ps]
        raw = v[1] - v[0]
        null = b[1] - b[0]
        if s["kind"] == "pair":
            lo, up, two = tails(null, raw)
            q = np.quantile(null, [0.025, 0.975], axis=0)
            for j, d in enumerate(DIMS):
                records.append(
                    dict(
                        start=ps[0],
                        end=ps[1],
                        dimension=d,
                        block_length=length,
                        observed_delta=float(raw[j]),
                        null_mean=float(null[:, j].mean()),
                        null_lo95=float(q[0, j]),
                        null_hi95=float(q[1, j]),
                        reference_two_sided=float(two[j]),
                    )
                )
        else:
            follow = v[2] - v[0]
            nf = b[2] - b[0]
            counts += null * nf > 0
            observed_count += raw * follow > 0
    q = np.quantile(counts, [0.025, 0.975], axis=0, method="inverted_cdf")
    rows = []
    ntriples = sum(s["kind"] == "triple" for s in specs)
    for j, d in enumerate(DIMS):
        rows.append(
            dict(
                dimension=d,
                block_length=length,
                triples=ntriples,
                observed_same_side=int(observed_count[j]),
                null_mean=float(counts[:, j].mean()),
                null_lo95=int(q[0, j]),
                null_hi95=int(q[1, j]),
                upper_reference_area=float(
                    (1 + np.sum(counts[:, j] >= observed_count[j])) / (len(counts) + 1)
                ),
                draws=len(counts),
            )
        )
    pd.DataFrame(rows).to_csv(out / f"global_retention_reference_L{length}.csv", index=False)
    pd.DataFrame(records).to_csv(out / f"global_transition_reference_L{length}.csv", index=False)
    pd.DataFrame(counts, columns=DIMS).to_csv(
        out / f"global_retention_count_draws_L{length}.csv", index=False
    )
