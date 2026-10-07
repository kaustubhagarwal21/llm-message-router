# LLM Message Router

A multimodal, per-user notification router for WhatsApp-style messages. For
every incoming message it decides, for the specific receiving user, whether to
`notify` (interrupt now), `digest` (show later in a batch) or `mute`
(suppress), and returns a message type, a one-sentence reason, a confidence
score and the IDs of the historical messages that justify the decision.

## Context

Built by Kaustubh Agarwal for the **HackerRank Orchestrate** challenge
(August 2026) as part of a team entry, which ranked in the **top 10%
(198th of 1,983)**.

The input stream mixes plain text, image posters and screenshots, and voice
notes, sent from friends, groups and business accounts. Some of the messages
are scams and some try to manipulate the router. The right answer depends on
the receiving user: the same promotion can be worth a digest for one user and
a mute for another.

## Architecture

```
dataset/*.csv ──► data_store.py ──────────────► per-message CONTEXT DOSSIER ──┐
                                                                              │
media/images ──► media_analyzer.py  (vision LLM: description + OCR + risk) ───┤
media/audio  ──► media_analyzer.py  (local faster-whisper transcription) ─────┤
                                                                              ▼
                 safety.py  (deterministic rule scanner, hard verdicts) ──► router.py
                                                                              │
                 few-shot anchoring: solved examples from                     │
                 sample_messages.csv, rendered with their real dossiers       │
                                                                              ▼
                                        post-validation ──► output.csv
```

### Pipeline

1. **Context dossier** (`code/data_store.py`). Joins everything known about
   the message and the person receiving it:
   - the user's notification behaviour (opens, replies, dismissals, reports, DND window);
   - group metadata, the user's membership and mute state, and the sender's role (admin or not);
   - business verification, official vs. sender domain, account age and user reports;
   - the user-business relationship (how they know the account, promotion opt-in or opt-out, engagement);
   - a ranked retrieval of the user's own message history (token overlap plus same-sender, same-business and same-group bonuses), annotated with how the user reacted to each message (opened, replied, dismissed, muted, reported).
2. **Media understanding** (`code/media_analyzer.py`).
   - **Images:** a vision LLM returns a short description, verbatim OCR text, a category guess and the visible risk signals (QR payment, OTP request, lookalike domain and similar). AVIF files saved with a `.jpg` extension are converted to JPEG with `ffmpeg` first.
   - **Voice notes:** transcribed locally with `faster-whisper` (the `small` model on CPU, int8).
3. **Deterministic safety layer** (`code/safety.py`). Regex and metadata rules
   catch prompt-injection attempts aimed at the router, OTP and credential
   requests, pressure wording, payment lures, suspicious or lookalike links,
   business-impersonation profiles, chain forwards and health misinformation.
   The highest-risk combinations get a **hard verdict** (`mute` / `scam`). The
   other findings go into the prompt as risk flags.
4. **Batched few-shot LLM routing** (`code/router.py`). Messages are routed
   five at a time. Each prompt holds the decision policy and a message-type
   guide. It also holds 13 solved examples from `sample_messages.csv`, each
   rendered with its real dossier, which anchor the decision logic, the reason
   style and the confidence calibration. The model must reply with a strict
   JSON array of decisions.
5. **Post-validation** (`router.finalize`):
   - checks the action and message type against the allowed values;
   - clamps confidence to a calibrated band;
   - keeps an evidence ID only if it exists in the message history and belongs to the receiving user (three at most);
   - enforces the hard safety verdicts over whatever the LLM returned.

Some of the personalisation cases this covers:
- The same marketplace post is `digest` for a user who engages with such posts and `mute` for one who keeps dismissing them.
- A promotion is `digest` for an opted-in user and `mute` after an opt-out.
- An urgent, personally directed message can still notify inside a group the user has muted.
- Clear scams are muted for everyone, however engaged the user usually is.

## Safety: two-layer guard against manipulation

Several messages in the stream try to instruct the router itself, for example
"mark this notify", "routing override" or fake router metadata. Two
independent layers handle them:

1. **Deterministic rules.** When an injection pattern matches, the message
   gets a hard `mute` / `scam` verdict. After the LLM responds, the
   post-validation step enforces that verdict, so the model's output cannot
   override it. The same applies to credential requests combined with
   pressure, a suspicious link or impersonation, and to impersonating
   businesses that push for payment or verification.
2. **Prompt instruction.** The routing policy tells the model to treat any
   instruction inside message content as a manipulation attempt and never to
   follow it.

This makes the router resistant to the injection styles it was built against.
It is a pattern-based defence, not a guarantee against every possible attack.

## Evaluation

`code/evaluation/main.py` has two modes:

- **Sample benchmark** (default). Re-routes the organisers' 30 solved sample
  messages as if they were unlabelled. It uses a leave-self-out setup: when a
  batch is evaluated, any sample in that batch is removed from the few-shot
  examples. The harness reports:
  - action accuracy and message-type accuracy;
  - an action confusion matrix;
  - evidence-ID overlap with the labelled evidence;
  - the mean confidence gap.
- **Output-contract validator** (`--check-output`). Checks `output.csv`:
  - exact columns;
  - one row per template ID, in template order;
  - allowed values for action and message type;
  - confidence in [0, 1];
  - a non-empty reason;
  - evidence IDs that exist in the message history.

The final version scored **30/30 on action** and **30/30 on message type** in
the sample benchmark. **This is not a held-out score.** The same 30 samples
supplied the few-shot anchors and were used to tune the prompt rules across
iterations. Treat the result as a regression check, not an estimate of
accuracy on unseen data.

## Reproducibility

- Every LLM completion is cached in `code/cache/llm_cache.json`. The cache key
  is a SHA-256 hash of the prompt, a call tag and a hash of the bytes of any
  attached images.
- Image analyses and voice transcripts are also stored as JSON under
  `code/cache/`, keyed by media ID.
- On a re-run with the cache present, every result comes from the cache, so
  the run is deterministic and needs no LLM access.
- The cache is **not included** in this repository, because it was produced
  from the competition data. The first run therefore needs an LLM backend and
  rebuilds it.

## Setup

```bash
pip install -r code/requirements.txt
```

The only third-party Python dependency is `faster-whisper`, which downloads the
Whisper model on first use. The Anthropic API is called with the Python
standard library. `ffmpeg` on PATH is optional and only needed for AVIF images.

LLM access is through Anthropic's Claude models. The code tries these backends
in order:

1. **Anthropic API**, if the `ANTHROPIC_API_KEY` environment variable is set.
   The model defaults to `claude-sonnet-4-5` and can be changed with
   `LLM_MODEL`.
2. **A logged-in Claude Code CLI**, run headlessly (`claude -p`). The binary
   is found on PATH, or set `CLAUDE_CLI` to its path. The model alias can be
   changed with `LLM_CLI_MODEL`.

Keys are read only from environment variables. None are stored in the repo.

## Run

```bash
# route every row of the output template and write output.csv at the repo root
python code/main.py

# quick smoke test on the first 5 rows
python code/main.py --limit 5

# benchmark on the solved samples
python code/evaluation/main.py

# validate output.csv against the submission contract
python code/evaluation/main.py --check-output
```

## Data

The competition dataset is **not included**, because it belongs to the
organisers. To run the code, place it in a `dataset/` folder at the repo root
with this layout:

```
dataset/
├── messages.csv                     # incoming messages to route
├── output.csv                       # submission template (defines row order)
├── sample_messages.csv              # solved examples (few-shots + benchmark)
├── users.csv
├── groups.csv
├── group_members.csv
├── business_accounts.csv
├── user_business_history.csv
├── message_history.csv
├── message_events.csv
├── daily_notification_summary.csv
├── images.csv                       # image_id,file_path  (e.g. media/images/img_001.jpg)
├── voice_notes.csv                  # voice_note_id,file_path  (e.g. media/audio/vn_001.mp3)
└── media/
    ├── images/
    └── audio/
```

Media paths in `images.csv` and `voice_notes.csv` are relative to `dataset/`.

## Repository layout

```
code/
├── main.py              # entry point: route all messages, write output.csv
├── data_store.py        # CSV loading + per-message context dossier
├── media_analyzer.py    # vision analysis of images, Whisper transcription of voice notes
├── safety.py            # deterministic risk scanner and hard verdicts
├── router.py            # prompt construction, batched LLM routing, post-validation
├── llm_client.py        # LLM backends (API / CLI) with content-hash caching
├── requirements.txt
└── evaluation/
    └── main.py          # sample benchmark + output-contract validator
```
