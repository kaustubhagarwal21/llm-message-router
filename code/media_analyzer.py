"""Multimodal pre-processing.

- Images: vision-model analysis -> description, OCR text, category, risk signals.
- Voice notes: local Whisper (faster-whisper) transcription.

Both are cached as JSON under code/cache/ so the routing stage is deterministic
and re-runnable offline.
"""

import json
import os
import sys

from llm_client import complete, extract_json

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE_DIR)
DATASET_DIR = os.path.join(ROOT, "dataset")
CACHE_DIR = os.path.join(CODE_DIR, "cache")
IMAGE_CACHE = os.path.join(CACHE_DIR, "image_analysis.json")
AUDIO_CACHE = os.path.join(CACHE_DIR, "voice_transcripts.json")

IMAGE_PROMPT = """You are analyzing a WhatsApp image attachment for a message-routing system.

Return ONLY a JSON object with these keys:
- "description": 1-2 sentences describing what the image shows
- "ocr_text": all readable text in the image, verbatim (empty string if none)
- "category_guess": one of poster/screenshot/document/photo/promo/qr_payment/other
- "risk_signals": list of strings; include entries like "qr_code_payment", "otp_request",
  "lookalike_domain", "prize_lure", "pressure_deadline", "loan_lure", "investment_lure"
  if visible, else empty list. Judge only what is visible in the image.

JSON only, no other text."""


def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(path, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _ensure_decodable(image_id, abs_path):
    """Some dataset images are AVIF files with a .jpg extension, which vision
    backends reject. Convert those to real JPEG via ffmpeg (cached)."""
    try:
        with open(abs_path, "rb") as f:
            magic = f.read(24)
    except OSError:
        return abs_path
    if b"ftypavif" not in magic:
        return abs_path
    conv_dir = os.path.join(CACHE_DIR, "converted")
    os.makedirs(conv_dir, exist_ok=True)
    out = os.path.join(conv_dir, f"{image_id}.jpg")
    if not os.path.exists(out):
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-y", "-i", abs_path, out],
                           capture_output=True, timeout=120, check=True)
            print(f"[images] {image_id}: converted AVIF -> JPEG", flush=True)
        except Exception as e:
            print(f"[images] {image_id}: AVIF conversion failed ({e}); using original", flush=True)
            return abs_path
    return out


def analyze_images(image_map):
    """image_map: {image_id: relative file path}. Returns {image_id: analysis dict}."""
    cache = _load(IMAGE_CACHE)
    for image_id, rel_path in sorted(image_map.items()):
        if image_id in cache and not str(cache[image_id].get("description", "")).startswith("analysis failed"):
            continue
        abs_path = os.path.join(DATASET_DIR, rel_path)
        if not os.path.exists(abs_path):
            cache[image_id] = {"description": "file missing", "ocr_text": "",
                               "category_guess": "other", "risk_signals": []}
            _save(IMAGE_CACHE, cache)
            continue
        print(f"[images] analyzing {image_id} ...", flush=True)
        try:
            raw = complete(IMAGE_PROMPT, image_paths=[_ensure_decodable(image_id, abs_path)],
                           tag=f"image:{image_id}")
            cache[image_id] = extract_json(raw)
        except Exception as e:  # keep going; one bad image must not kill the run
            print(f"[images] ERROR on {image_id}: {e}", flush=True)
            cache[image_id] = {"description": f"analysis failed: {e}", "ocr_text": "",
                               "category_guess": "other", "risk_signals": []}
        _save(IMAGE_CACHE, cache)
    return cache


def transcribe_voice_notes(voice_map):
    """voice_map: {voice_note_id: relative file path}. Returns {id: transcript dict}."""
    cache = _load(AUDIO_CACHE)
    missing = {k: v for k, v in voice_map.items() if k not in cache}
    if not missing:
        return cache

    from faster_whisper import WhisperModel
    print("[audio] loading Whisper model (first run downloads it)...", flush=True)
    model = WhisperModel("small", device="cpu", compute_type="int8")

    for vn_id, rel_path in sorted(missing.items()):
        abs_path = os.path.join(DATASET_DIR, rel_path)
        if not os.path.exists(abs_path):
            cache[vn_id] = {"transcript": "", "language": "", "note": "file missing"}
            _save(AUDIO_CACHE, cache)
            continue
        print(f"[audio] transcribing {vn_id} ...", flush=True)
        segments, info = model.transcribe(abs_path, beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        cache[vn_id] = {"transcript": text, "language": info.language,
                        "language_probability": round(info.language_probability, 3)}
        _save(AUDIO_CACHE, cache)
    return cache


def media_note_for(msg, image_cache, audio_cache):
    """Build the media context string for one message, or '' if no media."""
    media_type = msg.get("media_type") or ""
    media_id = msg.get("media_id") or ""
    if media_type == "image" and media_id in image_cache:
        a = image_cache[media_id]
        parts = [a.get("description", "")]
        if a.get("ocr_text"):
            parts.append(f"Text in image: \"{a['ocr_text'][:400]}\"")
        if a.get("risk_signals"):
            parts.append(f"Visible risk signals: {', '.join(a['risk_signals'])}")
        return " ".join(p for p in parts if p)
    if media_type == "voice" and media_id in audio_cache:
        t = audio_cache[media_id]
        if t.get("transcript"):
            return f"Voice note transcript ({t.get('language','?')}): \"{t['transcript']}\""
        return "Voice note could not be transcribed."
    if media_type:
        return f"{media_type} attachment {media_id} (content unavailable)."
    return ""


if __name__ == "__main__":
    # Standalone pre-processing: python media_analyzer.py [images|audio|all]
    import csv

    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    with open(os.path.join(DATASET_DIR, "images.csv"), encoding="utf-8-sig") as f:
        image_map = {r["image_id"]: r["file_path"] for r in csv.DictReader(f)}
    with open(os.path.join(DATASET_DIR, "voice_notes.csv"), encoding="utf-8-sig") as f:
        voice_map = {r["voice_note_id"]: r["file_path"] for r in csv.DictReader(f)}

    if what in ("audio", "all"):
        transcribe_voice_notes(voice_map)
        print("[audio] done")
    if what in ("images", "all"):
        analyze_images(image_map)
        print("[images] done")
