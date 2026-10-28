"""Prepare the supplementary coverage check using the same final representations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, shutil
import numpy as np
from common import metadata, output_directory


def run(inputs, output):
    inputs, output = output_directory(inputs, output)
    if inputs == output:
        raise ValueError("Use a different directory for the coverage subset.")
    m = metadata(inputs / "media_subset.csv")
    if "caption_available" in m:
        available = m.caption_available.astype(str).str.lower().isin(["true", "1"])
    else:
        available = m.description.fillna("").str.strip().ne("")
    with np.load(inputs / "coordinates_three_block.npz", allow_pickle=False) as z:
        keep = np.isin(z["content_id"].astype(str), m.loc[available, "content_id"])
        np.savez_compressed(
            output / "coordinates_three_block.npz", **{k: z[k][keep] for k in z.files}
        )
    # Retain the original acquisition inventory and publication positions.
    m.to_csv(output / "media_subset.csv", index=False)
    for name in ["window_members.csv", "publication_inventory.csv"]:
        if (inputs / name).exists():
            shutil.copy2(inputs / name, output / name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.inputs, a.output)
