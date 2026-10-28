"""Study 2: changes in dispersion, subsequent visibility and movement decomposition."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from common import output_directory
from core import *
from summaries import summarize_bootstrap, save_draws, global_diagnostics


def run(inputs, output, diagnostics=False, coverage_inputs=None):
    inputs, output = output_directory(inputs, output)
    cells, membership, inventory = load(inputs)
    # The same fixed membership is used for the principal and coverage specifications.
    coverage = load(coverage_inputs)[0] if coverage_inputs else load(inputs, caption_only=True)[0]
    fixed, candidates = fixed_cohort(cells, coverage)
    candidates.to_csv(output / "fixed_cohort_candidates.csv", index=False)
    dump(output / "fixed_cohort.json", dict(periods=fixed[0], creators=fixed[1]))
    membership.to_csv(output / "cell_membership.csv", index=False)
    dump(output / "inventory.json", inventory)
    specs, groups = comparisons(cells, fixed)
    levels = point_levels(cells, groups)
    for length, draws in [(3, 1999), (1, 999), (5, 999)] if diagnostics else [(3, 1999)]:
        boots = resampled_levels(cells, groups, draws, length, "bootstrap")
        summarize_bootstrap(cells, specs, levels, boots, output, length)
        save_draws(output, f"bootstrap_L{length}", boots)
    if not diagnostics:
        return
    assignments = []
    splits = []
    for s in specs:
        if s["kind"] != "pair":
            continue
        p0, p1 = s["periods"]
        cs = s["creators"]
        head = dict(start=p0, end=p1, n=len(cs))
        assignments.extend(dict(**head, **r) for r in assignment(cells, p0, p1, cs))
        for mode in ["random_half", "block_split"]:
            splits.extend(dict(**head, **r) for r in split_check(cells, p0, p1, cs, mode))
    pd.DataFrame(assignments).to_csv(output / "reassignment_reference.csv", index=False)
    pd.DataFrame(splits).to_csv(output / "split_starting_positions.csv", index=False)
    strong, _, _ = load(inputs, strong=True)
    rows = []
    for p0, p1 in zip(PERIODS[:-1], PERIODS[1:]):
        cs = tuple(sorted(c for c, p in strong if p == p0 and (c, p1) in strong))
        if len(cs) < 3:
            continue
        dec = decomp(strong, p0, p1, cs)
        rows.extend(
            dict(start=p0, end=p1, n=len(cs), dimension=d, **dict(zip(DECOMP_COLS, dec[j])))
            for j, d in enumerate(DIMS)
        )
    pd.DataFrame(rows).to_csv(output / "strong_coverage.csv", index=False)
    rows = []
    for length in [3, 1]:
        samples = resampled_levels(cells, groups, 999, length, "global_null")
        global_diagnostics(cells, specs, levels, samples, length, output)
        save_draws(output, f"global_stationary_L{length}", samples)
        for s in specs:
            if s["kind"] != "pair":
                continue
            p0, p1 = s["periods"]
            cs = s["creators"]
            rows.extend(
                dict(start=p0, end=p1, n=len(cs), **r)
                for r in local_stationary(cells, p0, p1, cs, length)
            )
    pd.DataFrame(rows).to_csv(output / "local_stationary_reference.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run the supporting analyses in Supplementary Section S3.",
    )
    p.add_argument(
        "--coverage-inputs",
        type=Path,
        help="Optional coverage subset for common fixed-cohort selection.",
    )
    a = p.parse_args()
    run(a.inputs, a.outputs, a.diagnostics, a.coverage_inputs)
