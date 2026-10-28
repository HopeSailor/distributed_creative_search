# Distributed Creative Search on Digital Platforms

Code for the final manuscript analysis of **4,635 videos from 29 food and cooking creators on TikTok**. The acquisition inventory contains 4,940 videos. The three studies distinguish conceptual content, audiovisual production and embodied enactment.

## Repository guide

| Stage | Folder | Purpose |
| --- | --- | --- |
| 1. Preprocessing | [01_preprocessing](01_preprocessing) | Validate publication order, reconstruct historical windows, and combine captions, speech transcripts and deduplicated OCR. |
| 2. Representation | [02_representation](02_representation) | Extract multimodal channels, encode conceptual content, build the three dimension vectors, and check correspondence with human ratings. |
| 3. Study 1 | [03_study_1](03_study_1) | Measure dimensional change and continuity between successive works. |
| 4. Study 2 | [04_study_2](04_study_2) | Trace expansion and contraction in differences among creators. |
| 5. Study 3 | [05_study_3](05_study_3) | Compare personal departures with collective history and follow subsequent works. |

Each study includes its analysis and final figure script. Supporting specifications belong to the same final analysis and are described in the Supplementary Material.

## Install

Use Python 3.11 or later. The numerical analyses require only the packages in the root requirements file.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Raw-media extraction requires FFmpeg, FFprobe and the additional dependencies in [02_representation/requirements.txt](02_representation/requirements.txt). GPU inference is optional; prepared vectors can be analyzed without installing the extraction models.

## Reproduce the analyses

Prepare the files specified in [INPUTS.md](INPUTS.md) in a local directory outside this repository. Then run:

```bash
python run.py all --inputs /path/to/analysis_inputs --outputs /path/to/results
```

The study-specific commands are:

```bash
python run.py study1 --inputs /path/to/analysis_inputs --outputs /path/to/results
python run.py study2 --inputs /path/to/analysis_inputs --outputs /path/to/results
python run.py study3 --inputs /path/to/analysis_inputs --outputs /path/to/results
```

Use quoted paths on Windows, for example `--inputs "D:/research/analysis_inputs"`. Outputs are written under `study1/`, `study2/` and `study3/`. The scripts use the original publication indices, so removing a video never makes its neighbors artificially consecutive.

For the sequence-reordering reference in Supplementary Section S2.4 and the additional resampling, reassignment and reference checks in Section S3:

```bash
python run.py all --inputs /path/to/analysis_inputs --outputs /path/to/results --diagnostics
```

The coverage check in the supplement retains the same final representations and changes which videos are included. Prepare it with `01_preprocessing/coverage_subset.py`, then run the studies on that input directory. Study 2 selects its fixed cohort from coverage common to both specifications; the final cohort has 19 creators across 2023H1–2026H1.

## Reproduce the figures

```bash
python 03_study_1/plot.py --results /path/to/results/study1 --output /path/to/figures
python 04_study_2/plot.py --results /path/to/results/study2 --output /path/to/figures
python 05_study_3/plot.py --results /path/to/results/study3 --output /path/to/figures
```

These commands export PDF, SVG and PNG files with the final panel structure. Text is black, except the grey “No” labels in Figure 2. Arial is used when installed; font substitution can change the layout on another system.

## Verification

The packaged analyses were checked against the frozen manuscript results. The main regression estimates and collective comparisons agree to numerical precision. Recomputed float32 historical distances differ by at most 1.2 × 10⁻⁷; all personal-departure, collective-departure and follow-up classifications agree. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the expected counts and validation scope.

## Data availability

This repository contains code, documentation and synthetic tests. Processed analysis data are available from the corresponding authors upon request for scholarly verification. Original TikTok videos, transcripts, individual embeddings and human-rating records are not redistributed here. Model weights are obtained from their respective providers when the optional extraction steps are run.
