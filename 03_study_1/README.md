# Study 1 Individual change and continuity

This study asks where creative change occurs and which dimensions carry related features into the next work.

- **Half-year change:** dimension-specific squared displacements, creator-balanced shares and transitions in the leading dimension.
- **Consecutive works:** one scalar vector-regression coefficient per dimension, with creator intercepts, creator-bootstrap intervals and wild-cluster coefficient comparisons.
- **Personal history:** the same model using shared six-work references and complementary three-work references.

Run `python run.py study1 --inputs INPUTS --outputs RESULTS` from the repository root. This also runs equal-creator weighting, the 30-day interval check and leave-one-creator-out estimates. Add `--diagnostics` for the supplementary sequence-reordering reference.

Outputs include `period_changes.csv`, `transition_counts.csv`, `all_adjacent_coefficients.csv`, `all_adjacent_differences.csv`, `change_first_FE_coefficients.csv` and `split_first_FE_coefficients.csv`. The bootstrap draws are saved for the figure. `plot.py` produces the four-panel Figure 1 from these outputs.
