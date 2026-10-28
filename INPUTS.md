# Input files

Keep research inputs outside the repository. The main analysis directory has this layout:

```text
analysis_inputs/
    media_subset.csv
    coordinates_three_block.npz
    window_members.csv
    publication_inventory.csv          # optional coverage diagnostic
```

## Publication metadata

`media_subset.csv` contains the 4,940-work acquisition inventory. The vector file identifies the 4,635 works with complete representations.

| Column | Meaning |
| --- | --- |
| `content_id` | Unique work identifier, read as a string. |
| `creator_handle` | Stable creator identifier, used consistently across files. |
| `created_at` | Original publication timestamp in ISO 8601 format. |
| `creator_sequence_index` | Integer position in the original creator publication inventory, before filtering. |
| `caption_available` or `description` | Caption availability for the supplementary coverage check and common fixed-cohort selection. If `caption_available` is supplied, use `True` or `False`. |

For upstream text preparation, `description` supplies the platform caption unless a `platform_metadata.json` file supplies it. For reconstruction of historical windows, retain `study1_q20_role`, `study1_q40_role`, `study1_q60_role`, `study1_q80_role` and `selection_reasons` from the acquisition manifest. These are recorded sampling roles; the code does not infer them from the results.

`publication_inventory.csv`, when provided, contains `content_id`, `creator_handle` and `created_at` for the larger publication inventory. It is used only to calculate the fraction of prior publications represented in the observed bank. It does not change eligibility, cutoffs or departure classifications.

## Representations

`coordinates_three_block.npz` is a NumPy archive with aligned rows:

| Array | Shape in the final sample |
| --- | --- |
| `content_id` | `(4635,)` |
| `block_conceptual` | `(4635, 768)` |
| `block_audiovisual` | `(4635, 1046)` |
| `block_embodied` | `(4635, 1452)` |
| `joint` | `(4635, 3266)` |

Each dimension vector has unit length. `joint` concatenates the three dimension vectors after multiplying each by `1/sqrt(3)`. Finite, nonzero channels are required; missing channels are not replaced with zeros. Creator-period means are not renormalized.

## Historical windows

`window_members.csv` supplies the recorded career and calendar windows for the Study 1 historical-reference analysis. It has `window_id`, `window_family`, `creator_handle`, `position`, `content_id`, `created_at`, `creator_sequence_index` and `consecutive_window` columns. Within each eligible window, positions 1–6 are history, position 7 is the focal work, and positions 8–9 are follow-ups. The final diagnostic uses 261 window records representing 260 distinct focal works from 27 creators.

`01_preprocessing/prepare.py` reconstructs this file from the sampling roles in the acquisition manifest. If only the basic metadata columns are supplied, provide the original window file separately. Study 3 constructs its own eligible focal works from strictly prior information and does not require these window labels or future availability.

## Optional raw-media inputs

```text
raw_archive/
    media_subset.csv
    media/<content_id>.mp4
    multimodal/<content_id>/
        platform_metadata.json
        asr.json
        ocr.json
        embeddings.npz
```

The extraction script produces the `multimodal` files from locally available media. `asr.json` supplies `transcript`; `ocr.json` supplies observations with `text` and `confidence`. The embedding archive includes `visual`, `audio_clap`, `editing` and `performance` channels. The extraction script also saves intermediate channel outputs for auditing.

For exact reconstruction of the representations, supply the frozen `editing_scaler.json` and the original model outputs. The scaler stores the means and scales fitted on the 862 observations dated through 31 December 2021. Running neural inference again can produce small differences across model revisions, devices and libraries.

## Human ratings

`02_representation/human_checks.py` takes the processed `joined_pairs.json` rating file. Each record contains `pair_id`, `video_a`, `video_b`, `creator_a`, `creator_b`, `stratum`, `distance_eligible`, and the three-item arrays `coder1`, `coder2` and `consensus`, ordered conceptual, audiovisual, embodied. Consensus ratings are required for distance-eligible records. The final check uses 150 pairs for coder agreement and 132 complete pairs for correspondence; this rated subset is distinct from the full analytical sample.
