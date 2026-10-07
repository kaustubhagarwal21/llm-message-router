"""Evaluation workflow.

Two jobs:

1. Sample benchmark (default):
   Re-routes every row of dataset/sample_messages.csv as if it were unlabeled
   (few-shot examples never include the row being evaluated) and scores the
   predictions against the provided labels:
     - action accuracy, message_type accuracy
     - confusion matrix for action
     - evidence overlap with the labeled evidence ids
     - mean absolute confidence gap

2. Output validation (--check-output):
   Validates output.csv against the submission contract: exact columns, one row
   per messages.csv id, allowed enum values, confidence in [0, 1].

Usage:
    python code/evaluation/main.py
    python code/evaluation/main.py --check-output
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import router  # noqa: E402
from data_store import DataStore  # noqa: E402
from media_analyzer import analyze_images, transcribe_voice_notes  # noqa: E402
from router import build_few_shots, route_batch, ACTIONS, TYPES  # noqa: E402

ROOT = os.path.dirname(CODE_DIR)
BATCH_SIZE = 5
COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def check_output(store, path):
    ok = True
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            print(f"FAIL columns: {reader.fieldnames}")
            return False
        rows = list(reader)
    expected = [r["message_id"] for r in store.output_template]
    got = [r["message_id"] for r in rows]
    if got != expected:
        print(f"FAIL row ids/order mismatch (got {len(got)}, expected {len(expected)})")
        ok = False
    for r in rows:
        if r["action"] not in ACTIONS:
            print(f"FAIL {r['message_id']}: bad action {r['action']!r}"); ok = False
        if r["message_type"] not in TYPES:
            print(f"FAIL {r['message_id']}: bad message_type {r['message_type']!r}"); ok = False
        try:
            c = float(r["confidence"])
            if not 0 <= c <= 1:
                raise ValueError
        except ValueError:
            print(f"FAIL {r['message_id']}: bad confidence {r['confidence']!r}"); ok = False
        if not r["reason"].strip():
            print(f"FAIL {r['message_id']}: empty reason"); ok = False
        ev = r["evidence_message_ids"]
        if ev != "none":
            for e in ev.split(";"):
                if e not in store.history_by_id:
                    print(f"FAIL {r['message_id']}: unknown evidence id {e}"); ok = False
    print("output.csv:", "PASS — schema, order, enums, confidence, evidence all valid" if ok else "FAILED checks above")
    return ok


def run_benchmark(store, image_cache, audio_cache):
    samples = store.samples
    print(f"[eval] benchmarking on {len(samples)} solved samples (leave-self-out few-shots)")

    preds = {}
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i : i + BATCH_SIZE]
        batch_ids = {s["message_id"] for s in batch}
        saved = router.FEW_SHOT_IDS
        router.FEW_SHOT_IDS = [x for x in saved if x not in batch_ids]
        try:
            fs = build_few_shots(store, image_cache, audio_cache)
            preds.update(route_batch(batch, store, image_cache, audio_cache, fs))
        finally:
            router.FEW_SHOT_IDS = saved
        print(f"[eval] {min(i + BATCH_SIZE, len(samples))}/{len(samples)} done", flush=True)

    act_ok = typ_ok = ev_hit = ev_total = 0
    conf_gap = []
    confusion = defaultdict(Counter)
    for s in samples:
        p = preds[s["message_id"]]
        gold_a, gold_t = s["action"], s["message_type"]
        if p["action"] == gold_a:
            act_ok += 1
        if p["message_type"] == gold_t:
            typ_ok += 1
        confusion[gold_a][p["action"]] += 1
        conf_gap.append(abs(float(s["confidence"]) - float(p["confidence"])))
        gold_ev = {e for e in s["evidence_message_ids"].split(";") if e and e != "none"}
        pred_ev = {e for e in p["evidence_message_ids"].split(";") if e and e != "none"}
        if gold_ev:
            ev_total += 1
            if gold_ev & pred_ev:
                ev_hit += 1
        if p["action"] != gold_a or p["message_type"] != gold_t:
            print(f"  MISS {s['message_id']}: gold={gold_a}/{gold_t} pred={p['action']}/{p['message_type']}")
            print(f"       pred reason: {p['reason']}")

    n = len(samples)
    print(f"\naction accuracy      : {act_ok}/{n} = {act_ok / n:.1%}")
    print(f"message_type accuracy: {typ_ok}/{n} = {typ_ok / n:.1%}")
    print(f"evidence overlap     : {ev_hit}/{ev_total} rows share >=1 gold evidence id")
    print(f"mean |conf gap|      : {sum(conf_gap) / n:.3f}")
    print("\nconfusion (gold rows x predicted cols):")
    acts = ["notify", "digest", "mute"]
    print("        " + "".join(f"{a:>8}" for a in acts))
    for g in acts:
        print(f"{g:>8}" + "".join(f"{confusion[g][a]:>8}" for a in acts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-output", action="store_true", help="validate output.csv instead of benchmarking")
    ap.add_argument("--out", default=os.path.join(ROOT, "output.csv"))
    args = ap.parse_args()

    store = DataStore()
    if args.check_output:
        sys.exit(0 if check_output(store, args.out) else 1)

    image_cache = analyze_images(store.images)
    audio_cache = transcribe_voice_notes(store.voice_notes)
    run_benchmark(store, image_cache, audio_cache)


if __name__ == "__main__":
    main()
