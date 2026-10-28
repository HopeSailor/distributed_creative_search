# Preprocessing

Publication order is preserved from the acquisition inventory. This stage validates IDs and timestamps, reconstructs the recorded historical windows, and prepares conceptual text.

```bash
python 01_preprocessing/prepare.py --inventory /path/to/raw_archive/media_subset.csv --output /path/to/analysis_inputs
python 01_preprocessing/text_fusion.py --metadata /path/to/raw_archive/media_subset.csv --channels /path/to/raw_archive/multimodal --output /path/to/text_inputs
```

Text is assembled in the order caption, speech transcript, OCR. OCR observations below 0.35 confidence are removed; normalized duplicates are removed while retaining first occurrence order. Any nonempty source can contribute.

For the supplementary coverage check:

```bash
python 01_preprocessing/coverage_subset.py --inputs /path/to/analysis_inputs --output /path/to/coverage_inputs
```
