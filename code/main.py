"""Entry point: route every message in dataset/messages.csv and write output.csv.

Usage:
    python code/main.py            # full run (uses caches; calls LLM for anything uncached)
    python code/main.py --limit 5  # smoke test on the first 5 template rows

Requires one LLM backend on the first run (ANTHROPIC_API_KEY or a logged-in
Claude Code CLI); afterwards everything is served from code/cache/ and the run
is fully offline and deterministic.
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_store import DataStore
from media_analyzer import analyze_images, transcribe_voice_notes
from router import build_few_shots, route_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT, "output.csv")
BATCH_SIZE = 5
COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="route only the first N template rows")
    ap.add_argument("--out", default=OUTPUT_PATH, help="output CSV path")
    args = ap.parse_args()

    t0 = time.time()
    store = DataStore()
    print(f"[data] {len(store.messages)} messages, {len(store.history)} history rows, "
          f"{len(store.samples)} solved samples")

    image_cache = analyze_images(store.images)
    audio_cache = transcribe_voice_notes(store.voice_notes)
    print(f"[media] {len(image_cache)} images analyzed, {len(audio_cache)} voice notes transcribed")

    few_shots_text = build_few_shots(store, image_cache, audio_cache)

    # Route in the exact row order of the output.csv template.
    order = [r["message_id"] for r in store.output_template]
    by_id = {m["message_id"]: m for m in store.messages}
    todo = [by_id[mid] for mid in order if mid in by_id]
    missing = [mid for mid in order if mid not in by_id]
    if missing:
        print(f"[warn] {len(missing)} template ids not in messages.csv: {missing[:5]}")
    if args.limit:
        todo = todo[: args.limit]

    results = {}
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        ids = [m["message_id"] for m in batch]
        print(f"[route] batch {i // BATCH_SIZE + 1}/{-(-len(todo) // BATCH_SIZE)}: {ids}", flush=True)
        results.update(route_batch(batch, store, image_cache, audio_cache, few_shots_text))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for mid in order:
            if mid in results:
                w.writerow(results[mid])
            elif not args.limit:
                print(f"[warn] no result for {mid}")

    done = sum(1 for mid in order if mid in results)
    print(f"[done] wrote {done} predictions to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
