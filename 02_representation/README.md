# Representation

Conceptual content uses captions, speech transcripts and deduplicated OCR. Audiovisual production combines visual, audio and editing channels. Embodied enactment combines face and pose sequence summaries. Each dimension receives equal weight in joint comparisons.

Install the additional extraction dependencies only when starting from raw videos:

```bash
python -m pip install -r 02_representation/requirements.txt
python 02_representation/extract_channels.py --run-dir /path/to/raw_archive --ffmpeg /path/to/ffmpeg --ffprobe /path/to/ffprobe --device cpu
```

Extraction uses 32 visual frames, 12 OCR frames and approximately five frames per second for temporal descriptors. Models are Whisper large-v3-turbo, all-mpnet-base-v2, CLIP ViT-B-32 with laion2b_s34b_b79k weights, CLAP htsat-fused, EasyOCR and MediaPipe.

After running text fusion:

```bash
python 02_representation/encode_conceptual.py --texts /path/to/text_inputs/fused_texts.jsonl --output /path/to/text_vectors
python 02_representation/build_blocks.py --inventory /path/to/raw_archive/media_subset.csv --channels /path/to/raw_archive/multimodal --conceptual /path/to/text_vectors/fused_conceptual.npz --editing-scaler /path/to/editing_scaler.json --output /path/to/analysis_inputs
```

The conceptual encoder pins revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Text is split into 382-token payloads, encoded with special tokens, averaged with token-count weights and normalized. Omit `--editing-scaler` to estimate it from the designated pre-2022 observations; use the frozen scaler for manuscript reproduction.

Human measurement checks:

```bash
python 02_representation/human_checks.py --inputs /path/to/analysis_inputs --ratings /path/to/joined_pairs.json --output /path/to/measurement_checks
```
