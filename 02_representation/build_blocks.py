"""Build the final conceptual, audiovisual, embodied and equally weighted joint vectors."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json
import numpy as np
import pandas as pd
from common import BLOCKS, metadata, dump


def unit(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    if not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12:
        raise ValueError("Invalid or zero channel.")
    return v / np.linalg.norm(v)


def run(inventory, channels, conceptual, output, scaler=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = metadata(inventory)
    raw = {}
    for cid in m.content_id:
        p = Path(channels) / cid / "embeddings.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                raw[cid] = {k: z[k].astype(float) for k in z.files}
        else:
            raw[cid] = {}
    if scaler:
        scale = json.loads(Path(scaler).read_text(encoding="utf-8"))
        mu, sd = np.array(scale["mean"]), np.array(scale["scale"])
    else:
        early = m[m.created_at <= pd.Timestamp("2021-12-31T23:59:59Z")]
        e = np.stack(
            [
                raw[c]["editing"]
                for c in early.content_id
                if "editing" in raw[c] and np.isfinite(raw[c]["editing"]).all()
            ]
        )
        mu, sd = e.mean(0), e.std(0, ddof=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        scale = dict(fit_end="2021-12-31T23:59:59Z", n=len(e), mean=mu, scale=sd)
    dump(output / "editing_scaler.json", scale)
    with np.load(conceptual, allow_pickle=False) as z:
        con = {str(cid): v for cid, v in zip(z["content_id"], z["conceptual"])}
    ids = []
    parts = {b: [] for b in BLOCKS}
    missing = []
    for cid in m.content_id:
        try:
            f = raw[cid]
            c = unit(con[cid])
            a = np.concatenate(
                [unit(f["visual"]), unit(f["audio_clap"]), unit((f["editing"] - mu) / sd)]
            ) / np.sqrt(3)
            e = unit(f["performance"])
            if (len(c), len(a), len(e)) != (768, 1046, 1452):
                raise ValueError("Unexpected channel dimensions.")
        except (KeyError, ValueError) as exc:
            missing.append(dict(content_id=cid, reason=str(exc)))
            continue
        ids.append(cid)
        for b, v in zip(BLOCKS, [c, a, e]):
            parts[b].append(v.astype(np.float32))
    arrays = {b: np.stack(parts[b]) for b in BLOCKS}
    np.savez_compressed(
        output / "coordinates_three_block.npz",
        content_id=np.asarray(ids),
        **{"block_" + b: v for b, v in arrays.items()},
        joint=np.concatenate([arrays[b] / np.sqrt(3) for b in BLOCKS], axis=1).astype(np.float32),
    )
    m.to_csv(output / "media_subset.csv", index=False)
    pd.DataFrame(missing).to_csv(output / "missing_representations.csv", index=False)
    dump(
        output / "representation_summary.json",
        dict(
            inventory=len(m),
            complete=len(ids),
            creators=m[m.content_id.isin(ids)].creator_handle.nunique(),
            missing=len(missing),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["inventory", "channels", "conceptual", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--editing-scaler", type=Path, help="Use the frozen scaler for exact reproduction."
    )
    a = p.parse_args()
    run(a.inventory, a.channels, a.conceptual, a.output, a.editing_scaler)
