# Study 2 Collective differentiation

This study asks how differences among creators expand, contract and remain visible over time.

Each half-year comparison retains the same creators at its endpoints. Dispersion is the mean squared distance of creator positions from their group mean. Circular block resampling estimates finite-video bias and uncertainty. Three-period comparisons assess whether the earlier shift remains visible relative to the initial period.

The decomposition expresses observed dispersion change as movement variation plus alignment between starting positions and movements. The supplementary diagnostics include reassigned movements, disjoint starting-position estimates, stronger cell-coverage rules and stationary-distribution references.

Run `python run.py study2 --inputs INPUTS --outputs RESULTS` from the repository root. Add `--diagnostics` for the supplementary checks. The default fixed-cohort selection uses coverage shared with the caption-available specification, as in the frozen analysis; all primary estimates use the 4,635-work representations.

`transitions_L3.csv` contains dimensional estimates and decomposition terms. `retention_L3.csv` contains subsequent visibility. `fixed_trajectory_L3.csv` contains the 19-creator trajectory. `plot.py` produces Figure 2 with panels (a), (b) and (c).
