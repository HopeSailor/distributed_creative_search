"""Shared input validation and deterministic numerical utilities."""

from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import pandas as pd

BLOCKS = ["conceptual", "audiovisual", "embodied"]
PERIODS = [f"{y}H{h}" for y in range(2020, 2027) for h in (1, 2) if (y, h) <= (2026, 1)]


def dump(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metadata(path):
    m = pd.read_csv(path, dtype={"content_id": str})
    required = ["content_id", "creator_handle", "created_at", "creator_sequence_index"]
    if not set(required) <= set(m):
        raise ValueError("Required metadata columns: " + ", ".join(required))
    if m[required].isna().any().any() or m.content_id.duplicated().any():
        raise ValueError("Metadata must have unique IDs and complete ordering fields.")
    m["created_at"] = pd.to_datetime(m.created_at, utc=True, format="mixed")
    seq = pd.to_numeric(m.creator_sequence_index, errors="raise")
    if not (seq % 1 == 0).all():
        raise ValueError("Publication indices must be integers.")
    m["creator_sequence_index"] = seq.astype(int)
    if m.duplicated(["creator_handle", "creator_sequence_index"]).any():
        raise ValueError("A publication index must identify one work within its creator.")
    return m


def load_vectors(inputs, include_joint=False):
    inputs = Path(inputs)
    with np.load(inputs / "coordinates_three_block.npz", allow_pickle=False) as z:
        ids = z["content_id"].astype(str)
        arrays = {b: z["block_" + b].astype(float) for b in BLOCKS}
        if include_joint:
            arrays["joint"] = z["joint"].astype(float)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate vector IDs.")
    m = metadata(inputs / "media_subset.csv").set_index("content_id").loc[ids].reset_index()
    for b, n in zip(BLOCKS, [768, 1046, 1452]):
        a = arrays[b]
        if a.shape != (len(ids), n) or not np.isfinite(a).all():
            raise ValueError("Invalid " + b + " vectors.")
        if np.max(abs(np.linalg.norm(a, axis=1) - 1)) > 1e-5:
            raise ValueError("Nonunit " + b + " vectors.")
    if include_joint:
        expected = np.concatenate([arrays[b] / np.sqrt(3) for b in BLOCKS], axis=1)
        if not np.allclose(arrays["joint"], expected, atol=1e-7, rtol=0):
            raise ValueError("Joint vectors must give each block equal weight.")
    return ids, m, arrays


def distance_cache(arrays, cache):
    """Create float32 Euclidean matrices, checked against their source arrays."""
    from scipy.spatial.distance import cdist

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for b, x in arrays.items():
        path = cache / (b + "_distances.npy")
        stamp = path.with_suffix(".json")
        signature = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        valid = (
            path.exists()
            and stamp.exists()
            and json.loads(stamp.read_text())["source_sha256"] == signature
        )
        if not valid:
            a = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(x), len(x)))
            for lo in range(0, len(x), 128):
                a[lo : lo + 128] = cdist(x[lo : lo + 128], x)
            np.fill_diagonal(a, 0)
            a.flush()
            del a
            dump(stamp, dict(source_sha256=signature, rows=len(x), metric="euclidean"))
        result[b] = np.load(path, mmap_mode="r")
    return result


def output_directory(inputs, outputs):
    inputs = Path(inputs).resolve()
    outputs = Path(outputs).resolve()
    if outputs == inputs or outputs in inputs.parents:
        raise ValueError("Use a separate output directory, not the input directory or its parent.")
    outputs.mkdir(parents=True, exist_ok=True)
    return inputs, outputs
