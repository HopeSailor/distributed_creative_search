# Private input specification

Place these seven files in the directory supplied to `--inputs`. They are **not included in this repository**. CSVs use UTF-8 and a header row. Video IDs must be read as strings, not floating-point numbers. Blank numerical fields denote missing values, never zero-filled observations.

| File | Stage | Unit |
| --- | --- | --- |
| `videos.csv` | Study 2 and diagnostics | One row per focal video, 1,690 rows |
| `coordinates_three_block.npz` | Study 2 and diagnostics | Complete three-block video representations |
| `coordinates_two_block.npz` | Study 2 and diagnostics | Complete concept and audiovisual representations |
| `events.csv` | Study 1 | Frozen local event, 106 three-block rows |
| `pairs.csv` | Study 1 | Adjacent creator-period comparison, 95 three-block rows |
| `triples.csv` | Study 1 | Three-period creator sequence, 83 three-block rows |
| `human_ratings.json` | Human validation | One record per completed video pair, 150 records |

## Video inventory and coordinates

`videos.csv` requires `content_id`, `creator_handle` and `created_at` (UTC timestamp). Creator names must match the fixed ten-account list in `common.py`. Duplicate IDs are rejected.

Each NPZ file must have:

- `content_id`: a one-dimensional Unicode array of video IDs, aligned with every matrix row;
- `joint`: a finite two-dimensional floating-point matrix of the final joint representations;
- `block_conceptual` and `block_audiovisual`: final block representations;
- `block_embodied`: required in the three-block file, absent in the two-block file.

The joint matrix is the concatenation of the included blocks weighted by the inverse square root of their number. Blocks already contain the within-block channel weighting and normalization used in the study. Do not restandardize features on the final ten creators. Earlier fitted preprocessing parameters belong to the upstream measurement specification.

The focal overlap must contain 1,632 three-block and 1,681 two-block videos. A larger private coordinate store is accepted, but only IDs present in the focal video inventory enter Study 2. The matched two-block specification uses exactly the three-block video IDs. The primary period range is 2020H1 through 2026H1; primary coverage requires five videos, three distinct publication dates and a 30-day timestamp span. Minimal and strong thresholds are specified in `study2.py`.

## Local event measurements

`events.csv` requires:

| Column | Meaning |
| --- | --- |
| `space` | `three_block` for the primary representation |
| `window_family` | `career` or `temporal` |
| `creator_handle` | Focal creator |
| `M` | Magnitude of the local departure |
| `S` | Frozen concentration measure of change across blocks |
| `P` | Continuation of the departure in follow-up works |
| `gap_days` | Event follow-up time control, in days |
| `peer_bank_valid` | Boolean validity indicator for the historical peer comparison |
| `D_start_priorbank` | Start position's distance from the frozen prior peer bank |
| `D_event_priorbank` | Event position's distance from that bank |
| `D_follow_priorbank` | Follow-up position's distance from that bank |

Optional identifiers such as `event_id` and `event_content_id` may remain in the private table. The primary models use creator fixed effects and CR1 creator-clustered standard errors; their Student-t reference uses the number of creator clusters minus one degree of freedom.

The historical peer bank uses the broader reference pool, not only the ten focal creators. For each event it is frozen at the event date, excludes the focal creator and uses eligible prior peer outputs. These distances are upstream input measurements. The published code fits the models from those measurements; it does not claim to rebuild them from the focal inventory alone.

## Individual-period measurements

Both tables require `space_variant`, `creator_handle`, `period_t` and `period_t1`. They may contain additional representation specifications; only `three_block` is analyzed here.

`pairs.csv` also requires, for each prefix `conceptual`, `audiovisual`, `embodied`:

- `<prefix>_distance`: nonnegative Euclidean movement distance;
- `<prefix>_signal_ratio`: frozen signal-to-uncertainty comparison;
- `<prefix>_relative_shift`: frozen relative movement measure;
- `pair_evidence_state`: the preserved upstream evidence classification, including `position_change_identified` where applicable.

`triples.csv` requires `period_t2`, `retention_beta`, `stable_pool_retention_mean` and `triple_evidence_state`. The last column includes `persistent_position_change_supported` where applicable. The stable-pool reference values and evidence labels are supplied, not inferred from thresholds chosen after inspecting the ten-account results. Their original multiplicity adjustments must remain unchanged.

## Human ratings

`human_ratings.json` is an array of records with these fields:

| Field | Type and definition |
| --- | --- |
| `pair_id` | Unique string |
| `stratum` | `within_near`, `within_far`, `between_same_halfyear`, `diagnostic_embodied_missing` or `diagnostic_caption_missing` |
| `creator_a`, `creator_b` | Endpoint creator names |
| `coder1`, `coder2` | Separate three-element integer arrays, each score from 1 to 5 |
| `consensus` | Three-element array of agreed scores; `null` for unavailable diagnostic consensus scores |
| `distances` | Three-element array of squared model distances, with unavailable channels represented by `null` |

Array order is **conceptual, audiovisual, embodied**. Higher human scores indicate greater difference. Coder 1 and coder 2 are independent human ratings; consensus is their subsequent agreed assessment. AI-generated auxiliary ratings must not be substituted for either human coder.

There are 91 within-creator and 41 between-creator complete pairs. Eighteen diagnostic pairs retain independent human scores but do not supply complete model distances. The validation code uses all 150 for human agreement and the 132 complete pairs for model correspondence. It uses creator-cluster bootstrap intervals for within-creator correspondence and leave-one-creator-out sensitivity ranges, not confidence intervals, for cross-creator comparisons.

## Local outputs

Every output is written below the explicit `--outputs` directory. These generated files are not part of this code-only release. Keep original private input files unchanged so that re-running the same version can be audited.
