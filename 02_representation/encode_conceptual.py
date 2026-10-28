import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import json, time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dump as write_json


def run(texts, output, model_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    revision = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_path:
        model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    else:
        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", revision=revision, device=device
        )
    model.eval()
    tokenizer = model.tokenizer
    max_len = int(model.max_seq_length)
    capacity = max_len - tokenizer.num_special_tokens_to_add(pair=False)
    special = tokenizer("", add_special_tokens=True)["input_ids"]
    assert len(special) == 2
    probe = "A short sentence about cooking."
    assert tokenizer(probe, add_special_tokens=True)["input_ids"] == [
        special[0]
    ] + tokenizer.encode(probe, add_special_tokens=False) + [special[1]]
    records = [json.loads(line) for line in Path(texts).read_text(encoding="utf-8").splitlines()]
    chunks = []
    chunk_lookup = {}
    documents = []
    audit = []
    for rec in records:
        for variant, key in [("fused", "combined_text")]:
            tokens = tokenizer.encode(rec[key], add_special_tokens=False, truncation=False)
            refs = []
            weights = []
            for start in range(0, len(tokens), capacity):
                part = tokens[start : start + capacity]
                token_key = tuple(part)
                if token_key not in chunk_lookup:
                    chunk_lookup[token_key] = len(chunks)
                    chunks.append(part)
                refs.append(chunk_lookup[token_key])
                weights.append(len(part))
            assert sum(weights) == len(tokens)
            documents.append((rec["content_id"], variant, refs, weights))
            audit.append(
                {
                    "content_id": rec["content_id"],
                    "variant": variant,
                    "tokens": len(tokens),
                    "chunks": len(refs),
                    "retained_tokens": sum(weights),
                }
            )
    print(
        json.dumps(
            {
                "device": device,
                "unique_chunks": len(chunks),
                "max_model_tokens": max_len,
                "chunk_payload": capacity,
            }
        ),
        flush=True,
    )
    values = np.empty((len(chunks), 768), dtype=np.float32)
    batch_size = 24
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            prepared = [[special[0]] + x + [special[1]] for x in chunks[start : start + batch_size]]
            width = max(map(len, prepared))
            input_ids = torch.full((len(prepared), width), tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for j, ids in enumerate(prepared):
                input_ids[j, : len(ids)] = torch.tensor(ids)
                attention[j, : len(ids)] = 1
            inputs = {"input_ids": input_ids, "attention_mask": attention}
            assert inputs["input_ids"].shape[1] <= max_len
            inputs = {k: v.to(device) for k, v in inputs.items()}
            v = model(inputs)["sentence_embedding"]
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            values[start : start + len(prepared)] = v.cpu().numpy()
            if start % 480 == 0:
                print(
                    f"Encoded {start + len(prepared)}/{len(chunks)} chunks; elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
    for variant in ["fused"]:
        ids = []
        vectors = []
        for cid, v, refs, w in documents:
            if v != variant or not refs:
                continue
            a = np.average(values[refs].astype(float), axis=0, weights=w)
            a = a / np.linalg.norm(a)
            ids.append(cid)
            vectors.append(a.astype(np.float32))
        np.savez_compressed(
            output / (variant + "_conceptual.npz"),
            content_id=np.array(ids),
            conceptual=np.stack(vectors),
        )
    import pandas as pd

    pd.DataFrame(audit).to_csv(output / "token_coverage.csv", index=False)
    write_json(
        output / "encoder_manifest.json",
        {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "revision": revision,
            "device": device,
            "max_seq_length": max_len,
            "payload_tokens_per_chunk": capacity,
            "pooling": "L2-normalized chunks, token-count-weighted mean, final L2",
            "unique_chunks": len(chunks),
            "seconds": time.time() - started,
            "torch_version": torch.__version__,
        },
    )
    print("Text encoding complete", flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Encode all fused text with token-weighted MPNet chunks."
    )
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model-path", type=Path, help="Optional local snapshot of the pinned model revision."
    )
    a = p.parse_args()
    run(a.texts, a.output, a.model_path)
