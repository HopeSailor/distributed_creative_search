"""Validate the archived inventory and reconstruct historical window membership."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from common import metadata, dump
from windows import build_windows


def run(source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m = metadata(source)
    # Original publication indices are retained; filtering must not create new neighbors.
    m.to_csv(output / "media_subset.csv", index=False)
    roles = [f"study1_q{q}_role" for q in (20, 40, 60, 80)]
    if set(roles + ["selection_reasons"]) <= set(m):
        w = build_windows(m)
        w.to_csv(output / "window_members.csv", index=False)
        windows = int(w.window_id.nunique())
    else:
        windows = None
    dump(
        output / "preprocessing_summary.json",
        dict(
            inventory_works=len(m),
            creators=m.creator_handle.nunique(),
            windows=windows,
            publication_indices="retained from original publication inventory",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.inventory, a.output)
