# Study 3 Personal departures and collective history

This study asks when a personal departure also lacks a close collective precedent, and what happens afterward.

Eligibility requires six consecutive prior works and an adequate strictly earlier collective bank. Future works are not required for inclusion. The same creator-balanced historical nearest-neighbor cutoff is applied to personal and collective distances. The historical bank and cutoff are fixed before the focal work when evaluating follow-ups.

Run `python run.py study3 --inputs INPUTS --outputs RESULTS` from the repository root. Pairwise distances are cached under the output directory; the cache is checked against its source vectors before reuse.

Outputs include eligibility, work-level metrics, cutoff sensitivity, dimensional departure counts, personal-change quartiles and return destinations. `earliest_observed_returns.csv` counts each returning focal case once and distinguishes a return within personal history from a peer precedent only. Missing observations are retained as missing. `plot.py` produces the four-panel Figure 3.
