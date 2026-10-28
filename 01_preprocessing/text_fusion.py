"""Combine captions, transcripts and deduplicated OCR in that order."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, re, unicodedata
import pandas as pd
from common import dump


def clean(value):
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def dedup_ocr(payload):
    observations = payload.get("observations", [])
    if observations:
        texts = [o.get("text", "") for o in observations if float(o.get("confidence", 0)) >= 0.35]
    else:
        texts = str(payload.get("text", "")).splitlines()
    seen = set()
    kept = []
    duplicates = 0
    for value in texts:
        text = clean(value)
        key = unicodedata.normalize("NFKC", text).casefold()
        if not text:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(text)
    return " ".join(kept), duplicates


def run(metadata, channel_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(metadata, dtype={"content_id": str}).fillna("")
    rows = []
    coverage = []
    for r in meta.itertuples():
        folder = Path(channel_root) / r.content_id
        payload = {}
        for name in ["platform_metadata", "asr", "ocr"]:
            p = folder / (name + ".json")
            payload[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        caption = clean(payload["platform_metadata"].get("caption", getattr(r, "description", "")))
        speech = clean(payload["asr"].get("transcript", ""))
        ocr, duplicates = dedup_ocr(payload["ocr"])
        combined = "\n".join(v for v in [caption, speech, ocr] if v)
        rows.append(
            dict(
                content_id=r.content_id,
                caption=caption,
                transcript=speech,
                ocr=ocr,
                combined_text=combined,
            )
        )
        coverage.append(
            dict(
                content_id=r.content_id,
                caption_chars=len(caption),
                transcript_chars=len(speech),
                ocr_chars=len(ocr),
                combined_chars=len(combined),
                ocr_duplicates_removed=duplicates,
            )
        )
    with (output / "fused_texts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(coverage).to_csv(output / "text_coverage.csv", index=False)
    dump(
        output / "text_summary.json",
        dict(videos=len(rows), nonempty_combined=sum(bool(r["combined_text"]) for r in rows)),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--channels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.metadata, a.channels, a.output)
