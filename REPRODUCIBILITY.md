# Reproducibility checks

The code accompanies the three-study manuscript using 29 creators.

| Analysis | Expected result |
| --- | --- |
| Complete representations | 4,635 works; 29 creators |
| Eligible half-year cells | 302 cells; 4,495 works |
| Study 1 adjacent periods | 254 comparisons |
| Largest dimension contributes at least 50% | 195 comparisons |
| Leading-dimension sequences | 218 sequences; 59 switches |
| Consecutive-work regression | 2,853 pairs; 3,389 distinct works; 29 creators |
| Historical-reference diagnostic | 261 windows; 260 focal works; 27 creators |
| Study 2 adjacent periods | 12 common-member comparisons |
| Earlier shifts still visible | 9 of 11 three-period comparisons |
| Fixed-cohort trajectory | 19 creators; 2023H1–2026H1 |
| Study 3 eligible foci | 946 works; 27 focal creators; historical bank from all 29 |
| Joint personal departures | 62 works |
| Personal departures covered by earlier peer work | 31 works |
| Joint collective departures | 31 works, or 3.3% of eligible foci |
| Collective departures by dimension | Conceptual 50; audiovisual 44; embodied 17 |
| Follow-ups establishing a return | 22 focal cases; nine have no observed follow-up |
| Earliest observed return | 18 within personal history; four with a peer precedent only |

The consecutive-work coefficients are approximately 0.0464, 0.1564 and 0.0630. The shared-history coefficients are 0.1678, 0.1580 and 0.1578; separate-history coefficients are 0.0090, −0.0005 and −0.0047. Dimensions are ordered conceptual, audiovisual and embodied.

## Checks performed for this release

- Executed the three study runners against the frozen processed inputs.
- Compared regression coefficients, intervals and direct contrasts with the frozen results.
- Compared all 48 Study 2 dimensional estimates, 44 dimensional three-period comparisons and 28 fixed-cohort levels.
- Recomputed historical distance matrices and checked all 946 personal, collective and follow-up classifications.
- Verified the 18/4 classification of earliest observed returns using the same pre-focal bank and cutoff.
- Ran synthetic tests for equal block weighting, decomposition, creator intercepts, separate-reference moments, missing follow-ups, weighted cutoffs and OCR deduplication.

The optional neural extraction pipeline is included with its model choices and sampling settings. Full neural inference was not repeated during code packaging. Analysis reproduction uses the frozen representations; reproducing raw media acquisition requires the archived publication inventory and locally available videos described in `INPUTS.md`.

The primary random seeds, bootstrap sizes and reference procedures are explicit in the scripts. Raw media, model weights, embeddings and research rows are excluded from the Git repository.
