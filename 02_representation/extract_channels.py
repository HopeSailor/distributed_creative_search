"""Extract the multimodal measurement channels from local TikTok videos.

This is a resumable multimodal measurement pipeline. It writes one directory
per immutable content_id, never treats a missing modality as zero, and records
model/version/sampling choices needed to reproduce every representation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "0.3.0"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
FACE_POINTS = [
    10,
    21,
    33,
    46,
    52,
    53,
    55,
    61,
    63,
    65,
    66,
    70,
    78,
    81,
    82,
    87,
    91,
    95,
    103,
    105,
    107,
    133,
    144,
    145,
    152,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    178,
    181,
    185,
    191,
    234,
    246,
    263,
    276,
    282,
    283,
    285,
    291,
    293,
    295,
    296,
    300,
    308,
    311,
    312,
    317,
    321,
    324,
    332,
    334,
    336,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    386,
    387,
    388,
    390,
    398,
    402,
    405,
    409,
    415,
    454,
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    def encode_unknown(item: Any) -> Any:
        if hasattr(item, "item"):
            return item.item()
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=encode_unknown),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def flatten_stats(prefix: str, values: Any) -> dict[str, float | None]:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {}
    return {
        f"{prefix}_mean": finite(np.nanmean(array)),
        f"{prefix}_std": finite(np.nanstd(array)),
        f"{prefix}_median": finite(np.nanmedian(array)),
    }


def l2_normalize(vector: Any) -> Any:
    import numpy as np

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def uniformly_spaced_indices(total: int, count: int, trim: int = 0) -> list[int]:
    import numpy as np

    if total <= 0 or count <= 0:
        return []
    lo = min(max(0, trim), total - 1)
    hi = max(lo, total - 1 - trim)
    realized = min(count, hi - lo + 1)
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi, realized)))


def clean_ocr_text(chunks: Iterable[str]) -> str:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in chunks:
        text = re.sub(r"\s+", " ", raw).strip()
        key = re.sub(r"\W+", "", text).lower()
        if len(key) < 2 or key in seen:
            continue
        if key in {"tiktok", "bdylanhollis"}:
            continue
        seen.add(key)
        cleaned.append(text)
    return "\n".join(cleaned)


def probe_media(ffprobe: Path, media_path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    return float(value)


def resolve_media(run_dir: Path, row: dict[str, str]) -> Path | None:
    declared = (row.get("local_media_path") or "").strip()
    if declared and Path(declared).is_file():
        return Path(declared).resolve()
    media_dir = run_dir / "media"
    for extension in VIDEO_EXTENSIONS:
        candidate = media_dir / f"{row['content_id']}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    return None


@dataclass
class Settings:
    ffmpeg: Path
    ffprobe: Path
    device: str
    visual_frames: int
    ocr_frames: int
    temporal_fps: float
    whisper_model: str
    sentence_model: str
    clip_model: str
    clip_pretrained: str
    clap_model: str


class ModelBundle:
    def __init__(self, settings: Settings) -> None:
        import torch
        from faster_whisper import WhisperModel
        from sentence_transformers import SentenceTransformer
        import open_clip
        import easyocr
        import mediapipe as mp
        from transformers import AutoProcessor, ClapModel

        self.torch = torch
        # A 10 GB consumer GPU cannot reliably hold Whisper, MPNet, CLIP,
        # CLAP, and EasyOCR at the same time.  Keep the latency-sensitive ASR
        # model on CUDA, while the smaller torch feature encoders run on CPU.
        # This preserves the models and representations while avoiding a CUDA
        # OOM that otherwise poisons every subsequent item in a long run.
        self.whisper_device = settings.device
        self.device = "cpu" if settings.device == "cuda" else settings.device
        compute_type = "float16" if settings.device == "cuda" else "int8"
        self.whisper = WhisperModel(
            settings.whisper_model, device=settings.device, compute_type=compute_type
        )
        self.sentence = SentenceTransformer(settings.sentence_model, device=self.device)
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained, device=self.device
        )
        self.clip.eval()
        self.clap_processor = AutoProcessor.from_pretrained(settings.clap_model)
        self.clap = ClapModel.from_pretrained(settings.clap_model).to(self.device).eval()
        self.ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self.face_mesh.close()
        self.pose.close()


def read_sampled_frames(
    path: Path, visual_count: int, ocr_count: int
) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]], float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0:
        capture.release()
        raise RuntimeError("container did not report a usable frame count")
    if fps <= 0:
        fps = 30.0
    trim = min(max(0, int(round(fps * 0.25))), max(0, total // 4))
    visual_indices = uniformly_spaced_indices(total, visual_count, trim=trim)
    ocr_indices = uniformly_spaced_indices(total, ocr_count, trim=trim)
    requested = set(visual_indices) | set(ocr_indices)
    sampled: dict[int, Any] = {}
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in requested:
            sampled[frame_index] = frame.copy()
        if frame_index >= max(requested, default=-1):
            break
        frame_index += 1
    capture.release()
    missing = sorted(requested - sampled.keys())
    if missing:
        raise RuntimeError(f"could not decode {len(missing)} requested frames")
    visual = [(index, sampled[index]) for index in visual_indices]
    ocr = [(index, sampled[index]) for index in ocr_indices]
    return visual, ocr, fps, total, width, height


def extract_audio(settings: Settings, media_path: Path, wav_path: Path) -> None:
    command = [
        str(settings.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def transcribe(
    models: ModelBundle, wav_path: Path
) -> tuple[dict[str, Any], Any | None, dict[str, Any]]:
    import numpy as np

    segments_iter, info = models.whisper.transcribe(
        str(wav_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segments = []
    for segment in segments_iter:
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": finite(segment.avg_logprob),
                "no_speech_prob": finite(segment.no_speech_prob),
                "words": [
                    {
                        "start": finite(word.start),
                        "end": finite(word.end),
                        "word": word.word,
                        "probability": finite(word.probability),
                    }
                    for word in (segment.words or [])
                ],
            }
        )
    transcript = " ".join(s["text"] for s in segments if s["text"]).strip()
    embedding = None
    semantic_stats: dict[str, Any] = {
        "asr_segment_count": len(segments),
        "asr_character_count": len(transcript),
    }
    if transcript:
        embedding = l2_normalize(models.sentence.encode(transcript, normalize_embeddings=True))
        segment_texts = [s["text"] for s in segments if s["text"]]
        if len(segment_texts) >= 2:
            segment_embeddings = np.asarray(
                models.sentence.encode(segment_texts, normalize_embeddings=True)
            )
            adjacent = 1.0 - np.sum(segment_embeddings[1:] * segment_embeddings[:-1], axis=1)
            centroid = l2_normalize(segment_embeddings.mean(axis=0))
            dispersion = 1.0 - segment_embeddings @ centroid
            semantic_stats.update(flatten_stats("narrative_adjacent_distance", adjacent))
            semantic_stats.update(flatten_stats("narrative_centroid_distance", dispersion))
    payload = {
        "language": getattr(info, "language", None),
        "language_probability": finite(getattr(info, "language_probability", float("nan"))),
        "duration": finite(getattr(info, "duration", float("nan"))),
        "duration_after_vad": finite(getattr(info, "duration_after_vad", float("nan"))),
        "transcript": transcript,
        "segments": segments,
    }
    return payload, embedding, semantic_stats


def encode_text(models: ModelBundle, text: str) -> Any | None:
    if not text.strip():
        return None
    return l2_normalize(models.sentence.encode(text.strip(), normalize_embeddings=True))


def visual_and_ocr(
    models: ModelBundle,
    visual_frames: list[tuple[int, Any]],
    ocr_frames: list[tuple[int, Any]],
) -> tuple[Any, dict[str, Any], dict[str, Any], Any | None]:
    import cv2
    import numpy as np
    from PIL import Image

    tensors = []
    visual_indices = [index for index, _frame in visual_frames]
    for _index, frame in visual_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(models.clip_preprocess(Image.fromarray(rgb)))
    batch = models.torch.stack(tensors).to(models.device)
    clip_vectors: list[Any] = []
    with models.torch.inference_mode():
        for start in range(0, len(batch), 8):
            encoded = models.clip.encode_image(batch[start : start + 8])
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            clip_vectors.append(encoded.detach().cpu().numpy())
    per_frame = np.concatenate(clip_vectors, axis=0)
    visual_embedding = l2_normalize(per_frame.mean(axis=0))
    adjacent = (
        1.0 - np.sum(per_frame[1:] * per_frame[:-1], axis=1)
        if len(per_frame) > 1
        else np.asarray([])
    )
    centroid_distance = 1.0 - per_frame @ visual_embedding
    visual_stats: dict[str, Any] = {
        "visual_frame_count": len(visual_indices),
        **flatten_stats("visual_adjacent_distance", adjacent),
        **flatten_stats("visual_centroid_distance", centroid_distance),
    }

    ocr_indices = [index for index, _frame in ocr_frames]
    observations: list[dict[str, Any]] = []
    all_text: list[str] = []
    for index, frame in ocr_frames:
        results = models.ocr.readtext(frame, detail=1, paragraph=False)
        frame_text = []
        for box, text, confidence in results:
            if float(confidence) < 0.35:
                continue
            frame_text.append(text)
            all_text.append(text)
            observations.append(
                {"frame_index": index, "text": text, "confidence": float(confidence), "box": box}
            )
    ocr_text = clean_ocr_text(all_text)
    ocr_payload = {
        "sampled_frame_indices": ocr_indices,
        "text": ocr_text,
        "observations": observations,
    }
    return visual_embedding, visual_stats, ocr_payload, encode_text(models, ocr_text)


def acoustic_and_clap(models: ModelBundle, wav_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        raise RuntimeError("decoded audio is empty")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    spectral_functions = {
        "centroid": librosa.feature.spectral_centroid,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "rolloff": librosa.feature.spectral_rolloff,
        "flatness": librosa.feature.spectral_flatness,
        "rms": librosa.feature.rms,
        "zcr": librosa.feature.zero_crossing_rate,
    }
    scalar: dict[str, Any] = {"audio_duration_s": float(len(y) / sr)}
    for name, function in spectral_functions.items():
        values = function(y=y, sr=sr) if name not in {"flatness", "rms", "zcr"} else function(y=y)
        features.extend([values.mean(axis=1), values.std(axis=1)])
        scalar.update(flatten_stats(f"audio_{name}", values))
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    scalar["audio_tempo_bpm"] = float(np.asarray(tempo).reshape(-1)[0])
    acoustic_embedding = l2_normalize(np.concatenate([np.asarray(v).reshape(-1) for v in features]))

    clap_sr = 48000
    clap_audio = librosa.resample(y, orig_sr=sr, target_sr=clap_sr)
    window_samples = clap_sr * 10
    starts = uniformly_spaced_indices(
        max(1, len(clap_audio) - window_samples + 1),
        min(8, max(1, math.ceil(len(clap_audio) / window_samples))),
    )
    clap_vectors = []
    with models.torch.inference_mode():
        for start in starts:
            chunk = clap_audio[start : start + window_samples]
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))
            inputs = models.clap_processor(audio=chunk, sampling_rate=clap_sr, return_tensors="pt")
            inputs = {key: value.to(models.device) for key, value in inputs.items()}
            clap_output = models.clap.get_audio_features(**inputs)
            encoded = (
                clap_output.pooler_output if hasattr(clap_output, "pooler_output") else clap_output
            )
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            clap_vectors.append(encoded[0].detach().cpu().numpy())
    clap_embedding = l2_normalize(np.mean(clap_vectors, axis=0))
    scalar["clap_window_count"] = len(clap_vectors)
    return acoustic_embedding, clap_embedding, scalar


def normalize_face(landmarks: list[Any]) -> Any | None:
    import numpy as np

    points = np.asarray(
        [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in FACE_POINTS], dtype=np.float32
    )
    left = np.asarray([landmarks[33].x, landmarks[33].y, landmarks[33].z], dtype=np.float32)
    right = np.asarray([landmarks[263].x, landmarks[263].y, landmarks[263].z], dtype=np.float32)
    scale = float(np.linalg.norm(left[:2] - right[:2]))
    if scale < 1e-5:
        return None
    return ((points - (left + right) / 2.0) / scale).reshape(-1)


def normalize_pose(landmarks: list[Any]) -> Any | None:
    import numpy as np

    points = np.asarray([[p.x, p.y, p.z, p.visibility] for p in landmarks], dtype=np.float32)
    left, right = points[11, :3], points[12, :3]
    scale = float(np.linalg.norm(left[:2] - right[:2]))
    if scale < 1e-5:
        return None
    center = (left + right) / 2.0
    points[:, :3] = (points[:, :3] - center) / scale
    return points.reshape(-1)


def sequence_embedding(sequences: list[Any]) -> tuple[Any | None, dict[str, Any]]:
    import numpy as np

    if not sequences:
        return None, {"frames": 0}
    array = np.asarray(sequences, dtype=np.float32)
    velocity = np.diff(array, axis=0)
    acceleration = np.diff(velocity, axis=0)
    pieces = [array.mean(axis=0), array.std(axis=0)]
    if len(velocity):
        pieces.extend([np.mean(np.abs(velocity), axis=0), np.std(velocity, axis=0)])
    else:
        pieces.extend([np.zeros(array.shape[1]), np.zeros(array.shape[1])])
    embedding = l2_normalize(np.concatenate(pieces))
    diagnostics = {
        "frames": int(len(array)),
        "velocity_energy": finite(np.mean(np.linalg.norm(velocity, axis=1)))
        if len(velocity)
        else 0.0,
        "acceleration_energy": finite(np.mean(np.linalg.norm(acceleration, axis=1)))
        if len(acceleration)
        else 0.0,
    }
    return embedding, diagnostics


def performance_and_editing(
    models: ModelBundle,
    media_path: Path,
    fps: float,
    total_frames: int,
    temporal_fps: float,
) -> tuple[Any | None, dict[str, Any], Any, dict[str, Any]]:
    import cv2
    import numpy as np

    stride = max(1, int(round(fps / temporal_fps)))
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot stream {media_path}")
    indices: list[int] = []
    face_vectors, pose_vectors = [], []
    histograms, grays, brightness, saturation = [], [], [], []
    decoded_frames_seen = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames_seen += 1
        if frame_index % stride != 0:
            frame_index += 1
            continue
        indices.append(frame_index)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = models.face_mesh.process(rgb)
        if face.multi_face_landmarks:
            normalized = normalize_face(face.multi_face_landmarks[0].landmark)
            if normalized is not None:
                face_vectors.append(normalized)
        pose = models.pose.process(rgb)
        if pose.pose_landmarks:
            normalized = normalize_pose(pose.pose_landmarks.landmark)
            if normalized is not None:
                pose_vectors.append(normalized)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist)
        grays.append(gray)
        brightness.append(float(hsv[..., 2].mean() / 255.0))
        saturation.append(float(hsv[..., 1].mean() / 255.0))
        frame_index += 1
    capture.release()
    if not indices:
        raise RuntimeError("streaming decoder returned no temporal samples")

    face_embedding, face_stats = sequence_embedding(face_vectors)
    pose_embedding, pose_stats = sequence_embedding(pose_vectors)
    # The representation must have the same meaning and dimensionality for
    # every video. A face-only or pose-only vector is therefore marked missing
    # rather than silently substituted for the joint performance channel.
    performance_embedding = (
        l2_normalize(np.concatenate([face_embedding, pose_embedding]))
        if face_embedding is not None and pose_embedding is not None
        else None
    )
    performance_stats: dict[str, Any] = {
        "performance_sampled_frames": len(indices),
        "source_decoded_frames_seen": decoded_frames_seen,
        "face_detection_coverage": len(face_vectors) / len(indices) if indices else 0.0,
        "pose_detection_coverage": len(pose_vectors) / len(indices) if indices else 0.0,
        **{f"face_{k}": v for k, v in face_stats.items()},
        **{f"pose_{k}": v for k, v in pose_stats.items()},
    }

    hist_distances, pixel_differences, motion, freezes = [], [], [], []
    for previous, current, prev_gray, gray in zip(histograms, histograms[1:], grays, grays[1:]):
        hist_distances.append(float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA)))
        pixel_differences.append(float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0))
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion.append(float(np.median(magnitude)))
        freezes.append(float(np.mean(cv2.absdiff(prev_gray, gray)) < 1.0))
    hist_array = np.asarray(hist_distances)
    cut_flags = hist_array > 0.55
    sample_interval = stride / fps
    duration = total_frames / fps
    cut_times = np.asarray(indices[1:])[cut_flags] / fps if len(hist_array) else np.asarray([])
    boundaries = np.concatenate([[0.0], cut_times, [duration]])
    shot_lengths = np.diff(boundaries)
    editing_stats: dict[str, Any] = {
        "duration_s": duration,
        "source_fps": fps,
        "editing_sample_fps": 1.0 / sample_interval,
        "editing_sampled_frames": len(indices),
        "cut_count": int(cut_flags.sum()),
        "cuts_per_second": float(cut_flags.sum() / duration) if duration else 0.0,
        "freeze_transition_share": float(np.mean(freezes)) if freezes else None,
        **flatten_stats("shot_length_s", shot_lengths),
        **flatten_stats("histogram_transition", hist_distances),
        **flatten_stats("pixel_transition", pixel_differences),
        **flatten_stats("optical_flow", motion),
        **flatten_stats("brightness", brightness),
        **flatten_stats("saturation", saturation),
    }
    # Exclude codec and sampling diagnostics (source FPS, realized sample count)
    # from the creative editing representation. They remain in scalars for QA.
    editing_names = sorted(
        name
        for name in editing_stats
        if name
        in {
            "duration_s",
            "cut_count",
            "cuts_per_second",
            "freeze_transition_share",
            "shot_length_s_mean",
            "shot_length_s_std",
            "shot_length_s_median",
            "histogram_transition_mean",
            "histogram_transition_std",
            "histogram_transition_median",
            "pixel_transition_mean",
            "pixel_transition_std",
            "pixel_transition_median",
            "optical_flow_mean",
            "optical_flow_std",
            "optical_flow_median",
            "brightness_mean",
            "brightness_std",
            "brightness_median",
            "saturation_mean",
            "saturation_std",
            "saturation_median",
        }
        and editing_stats[name] is not None
    )
    editing_embedding = np.asarray(
        [float(editing_stats[name]) for name in editing_names], dtype=np.float32
    )
    editing_embedding = l2_normalize(editing_embedding)
    editing_stats["editing_feature_order"] = editing_names
    return performance_embedding, performance_stats, editing_embedding, editing_stats


def save_npz(path: Path, embeddings: dict[str, Any | None]) -> None:
    import numpy as np

    present = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in embeddings.items()
        if value is not None
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **present)
    os.replace(tmp, path)


def process_video(
    row: dict[str, str], media_path: Path, output_dir: Path, settings: Settings, models: ModelBundle
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio_16k_mono.wav"
    probe = probe_media(settings.ffprobe, media_path)
    visual_frames, ocr_frames, fps, declared_frames, width, height = read_sampled_frames(
        media_path, settings.visual_frames, settings.ocr_frames
    )
    extract_audio(settings, media_path, wav_path)

    asr, semantic_embedding, semantic_stats = transcribe(models, wav_path)
    json_dump(output_dir / "asr.json", asr)
    caption_embedding = encode_text(models, row.get("description", ""))
    json_dump(
        output_dir / "platform_metadata.json",
        {
            "content_id": row["content_id"],
            "caption": row.get("description", ""),
            "music_id": row.get("music_id", ""),
            "music_title": row.get("music_title", ""),
            "music_author": row.get("music_author", ""),
        },
    )

    visual_embedding, visual_stats, ocr, ocr_embedding = visual_and_ocr(
        models, visual_frames, ocr_frames
    )
    json_dump(output_dir / "ocr.json", ocr)

    acoustic_embedding, clap_embedding, audio_stats = acoustic_and_clap(models, wav_path)
    performance_embedding, performance_stats, editing_embedding, editing_stats = (
        performance_and_editing(models, media_path, fps, declared_frames, settings.temporal_fps)
    )

    embeddings = {
        "semantic_transcript": semantic_embedding,
        "semantic_caption": caption_embedding,
        "onscreen_text": ocr_embedding,
        "visual": visual_embedding,
        "audio_clap": clap_embedding,
        "audio_acoustic": acoustic_embedding,
        "performance": performance_embedding,
        "editing": editing_embedding,
    }
    save_npz(output_dir / "embeddings.npz", embeddings)
    scalars = {
        "content_id": row["content_id"],
        "created_at": row.get("created_at"),
        "music_id": row.get("music_id", ""),
        "music_title": row.get("music_title", ""),
        "music_author": row.get("music_author", ""),
        **semantic_stats,
        **visual_stats,
        **audio_stats,
        **performance_stats,
        **editing_stats,
    }
    json_dump(output_dir / "scalars.json", scalars)
    qa = {
        "pipeline_version": PIPELINE_VERSION,
        "completed_at_utc": utc_now(),
        "media_path": str(media_path),
        "media_sha256": sha256_file(media_path),
        "media_bytes": media_path.stat().st_size,
        "probe": probe,
        "decoded_frame_count": performance_stats.get("source_decoded_frames_seen"),
        "retained_visual_frame_count": len(visual_frames),
        "retained_ocr_frame_count": len(ocr_frames),
        "container_declared_frame_count": declared_frames,
        "decoded_fps": fps,
        "width": width,
        "height": height,
        "channels_present": sorted(
            embeddings.keys() - {key for key, value in embeddings.items() if value is None}
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    json_dump(output_dir / "qa.json", qa)
    wav_path.unlink(missing_ok=True)
    return {"status": "complete", "content_id": row["content_id"], **qa}


def write_status(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "content_id",
        "status",
        "media_path",
        "error",
        "completed_at_utc",
        "elapsed_seconds",
        "channels_present",
    ]
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("content_id"):
                    merged[row["content_id"]] = row
    for record in records:
        merged[str(record["content_id"])] = record
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in merged.values():
            row = dict(record)
            if isinstance(row.get("channels_present"), list):
                row["channels_present"] = "|".join(row["channels_present"])
            writer.writerow(row)
    os.replace(tmp, path)


def audit_media(run_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    found, missing = [], []
    for row in rows:
        path = resolve_media(run_dir, row)
        (found if path else missing).append(row["content_id"])
    return {
        "frozen_rows": len(rows),
        "media_found": len(found),
        "media_missing": len(missing),
        "found_content_ids": found,
        "missing_content_ids": missing,
        "expected_directory": str((run_dir / "media").resolve()),
        "required_filename_pattern": "<content_id>.<mp4|mov|mkv|webm|avi|m4v>",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--ffprobe", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--visual-frames", type=int, default=32)
    parser.add_argument("--ocr-frames", type=int, default=12)
    parser.add_argument("--temporal-fps", type=float, default=5.0)
    parser.add_argument(
        "--content-id-file",
        type=Path,
        help="Optional UTF-8 text file containing one content ID per line; process only those rows",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    with (run_dir / "media_subset.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    posts_path = run_dir / "posts.csv"
    if posts_path.exists():
        with posts_path.open("r", encoding="utf-8-sig", newline="") as handle:
            posts = {str(post["content_id"]): post for post in csv.DictReader(handle)}
        for row in rows:
            source = posts.get(str(row["content_id"]), {})
            for field in ("music_id", "music_title", "music_author"):
                row[field] = source.get(field, "")
    if args.content_id_file:
        wanted = {
            line.strip()
            for line in args.content_id_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row["content_id"]) in wanted]
    total_rows = len(rows)
    start_index = max(0, args.start_index)
    rows = rows[start_index:]
    if args.limit > 0:
        rows = rows[: args.limit]
    output_root = run_dir / "multimodal"
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_media(run_dir, rows)
    if args.audit_only and start_index == 0 and (args.limit <= 0 or args.limit >= total_rows):
        audit_path = output_root / "media_audit.json"
    else:
        audit_path = (
            output_root / "batch_audits" / f"media_audit_{start_index:06d}_{len(rows):04d}.json"
        )
    json_dump(audit_path, audit)
    print(
        json.dumps(
            {
                k: audit[k]
                for k in ["frozen_rows", "media_found", "media_missing", "expected_directory"]
            }
        ),
        flush=True,
    )
    if args.audit_only or audit["media_found"] == 0:
        return 2 if audit["media_found"] == 0 else 0

    settings = Settings(
        ffmpeg=args.ffmpeg.resolve(),
        ffprobe=args.ffprobe.resolve(),
        device=args.device,
        visual_frames=args.visual_frames,
        ocr_frames=args.ocr_frames,
        temporal_fps=args.temporal_fps,
        whisper_model="large-v3-turbo",
        sentence_model="sentence-transformers/all-mpnet-base-v2",
        clip_model="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        clap_model="laion/clap-htsat-fused",
    )
    json_dump(
        output_root / "model_manifest.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "created_at_utc": utc_now(),
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in settings.__dict__.items()
            },
            "execution_profile": {
                "whisper_device": settings.device,
                "torch_feature_device": "cpu" if settings.device == "cuda" else settings.device,
                "easyocr_device": "cpu",
            },
            "python": sys.version,
            "inferential_boundary": "channel extraction; sample eligibility is evaluated downstream",
            "package_versions": {
                package: importlib.metadata.version(package)
                for package in [
                    "torch",
                    "torchvision",
                    "faster-whisper",
                    "sentence-transformers",
                    "open-clip-torch",
                    "transformers",
                    "librosa",
                    "mediapipe",
                    "easyocr",
                    "opencv-python-headless",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "scipy",
                ]
            },
        },
    )
    records: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, str], Path]] = []
    for position, row in enumerate(rows, 1):
        content_id = row["content_id"]
        media_path = resolve_media(run_dir, row)
        qa_path = output_root / content_id / "qa.json"
        if media_path is None:
            records.append(
                {
                    "content_id": content_id,
                    "status": "missing_media",
                    "error": "local video not found",
                }
            )
            continue
        if qa_path.exists() and not args.force:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            if qa.get("pipeline_version") == PIPELINE_VERSION and qa.get(
                "media_sha256"
            ) == sha256_file(media_path):
                records.append({"content_id": content_id, "status": "skipped_verified", **qa})
                print(f"[{position}/{len(rows)}] skip verified {content_id}", flush=True)
                continue
        pending.append((position, row, media_path))

    write_status(output_root / "extraction_status.csv", records)
    if not pending:
        completed = sum(record["status"] in {"complete", "skipped_verified"} for record in records)
        print(json.dumps({"completed": completed, "failed": 0, "total": len(rows)}), flush=True)
        return 0

    models = ModelBundle(settings)
    try:
        for position, row, media_path in pending:
            content_id = row["content_id"]
            per_video = output_root / content_id
            print(f"[{position}/{len(rows)}] process {content_id}", flush=True)
            try:
                records.append(process_video(row, media_path, per_video, settings, models))
            except Exception as exc:
                records.append(
                    {
                        "content_id": content_id,
                        "status": "failed",
                        "media_path": str(media_path),
                        "error": f"{type(exc).__name__}: {exc}",
                        "completed_at_utc": utc_now(),
                    }
                )
                print(
                    f"FAILED {content_id}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True
                )
            finally:
                # Release per-video frame/audio arrays before the next item.
                # Several native libraries keep cyclic Python references long
                # enough to exhaust Windows' commit limit in long batches.
                gc.collect()
            write_status(output_root / "extraction_status.csv", records)
    finally:
        models.close()
    write_status(output_root / "extraction_status.csv", records)
    failures = sum(record["status"] == "failed" for record in records)
    completed = sum(record["status"] in {"complete", "skipped_verified"} for record in records)
    print(json.dumps({"completed": completed, "failed": failures, "total": len(rows)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
