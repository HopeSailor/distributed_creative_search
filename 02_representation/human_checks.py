"""Independent coder agreement and correspondence with the final representations (S1.3)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import load_vectors, BLOCKS


def run(inputs, ratings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(ratings).read_text(encoding="utf-8"))
    ids, _, blocks = load_vectors(inputs)
    ix = {v: i for i, v in enumerate(ids)}
    agreement = []
    correspondence = []
    for j, b in enumerate(BLOCKS):
        a = np.array([r["coder1"][j] for r in rows])
        c = np.array([r["coder2"][j] for r in rows])
        expected = sum(
            abs(x - y) * np.mean(a == x) * np.mean(c == y) for x in range(1, 6) for y in range(1, 6)
        )
        agreement.append(
            dict(
                dimension=b,
                pairs=len(rows),
                linear_weighted_kappa=1 - np.mean(abs(a - c)) / expected,
                exact_agreement=float(np.mean(a == c)),
                within_one_point=float(np.mean(abs(a - c) <= 1)),
            )
        )
        for group in ["within", "between"]:
            kept = [r for r in rows if r["distance_eligible"] and r["stratum"].startswith(group)]
            dist = np.array(
                [
                    np.sum((blocks[b][ix[r["video_a"]]] - blocks[b][ix[r["video_b"]]]) ** 2)
                    for r in kept
                ]
            )
            for label in ["consensus", "coder1", "coder2"]:
                scores = np.array([r[label][j] for r in kept])
                rho = float(spearmanr(scores, dist).statistic)
                creators = sorted(
                    set(r["creator_a"] for r in kept) | set(r["creator_b"] for r in kept)
                )
                loo = []
                for creator in creators:
                    retain = np.array(
                        [creator not in [r["creator_a"], r["creator_b"]] for r in kept]
                    )
                    loo.append(float(spearmanr(scores[retain], dist[retain]).statistic))
                correspondence.append(
                    dict(
                        group=group,
                        dimension=b,
                        human_score=label,
                        pairs=len(kept),
                        rho=rho,
                        membership_min=min(loo),
                        membership_max=max(loo),
                    )
                )
    pd.DataFrame(agreement).to_csv(output / "coder_agreement.csv", index=False)
    pd.DataFrame(correspondence).to_csv(output / "human_correspondence.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "ratings", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.ratings, a.output)
