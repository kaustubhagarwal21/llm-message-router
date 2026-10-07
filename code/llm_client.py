"""LLM access layer with caching.

Two backends, tried in order:
  1. Anthropic API  — if ANTHROPIC_API_KEY is set (no extra dependencies).
  2. Claude Code CLI — headless `claude -p`, using the local logged-in session.
     Override the binary with the CLAUDE_CLI env var if it is not on PATH.

Every response is cached in code/cache/llm_cache.json keyed by a content hash,
so re-runs are deterministic and work fully offline once the cache exists.
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(CODE_DIR, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.json")

API_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
CLI_MODEL_ALIAS = os.environ.get("LLM_CLI_MODEL", "sonnet")


def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, CACHE_FILE)


_cache = _load_cache()


def _find_cli():
    if os.environ.get("CLAUDE_CLI") and os.path.exists(os.environ["CLAUDE_CLI"]):
        return os.environ["CLAUDE_CLI"]
    found = shutil.which("claude")
    if found:
        return found
    # Known VS Code extension bundle location (Windows)
    ext_root = os.path.join(os.environ.get("USERPROFILE", ""), ".vscode", "extensions")
    if os.path.isdir(ext_root):
        for d in sorted(os.listdir(ext_root), reverse=True):
            if d.startswith("anthropic.claude-code"):
                cand = os.path.join(ext_root, d, "resources", "native-binary", "claude.exe")
                if os.path.exists(cand):
                    return cand
    return None


def _call_api(prompt, image_paths, max_tokens):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    content = []
    for p in image_paths or []:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        media_type = "image/jpeg" if p.lower().endswith((".jpg", ".jpeg")) else "image/png"
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
    content.append({"type": "text", "text": prompt})
    body = json.dumps({
        "model": API_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return "".join(b.get("text", "") for b in out.get("content", []))


def _call_cli(prompt, image_paths, max_tokens):
    cli = _find_cli()
    if not cli:
        return None
    full_prompt = prompt
    args = [cli, "-p", "--model", CLI_MODEL_ALIAS, "--output-format", "text"]
    if image_paths:
        refs = "\n".join(f"- {os.path.abspath(p)}" for p in image_paths)
        full_prompt = (
            "First use the Read tool to view each of these image files:\n"
            f"{refs}\n\n" + prompt
        )
        args += ["--allowedTools", "Read"]
    res = subprocess.run(
        args, input=full_prompt.encode("utf-8"),
        capture_output=True, timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(f"CLI failed: {res.stderr.decode('utf-8', 'replace')[:500]}")
    return res.stdout.decode("utf-8", "replace").strip()


def complete(prompt, image_paths=None, max_tokens=4000, tag=""):
    """Cached LLM completion. Returns response text."""
    def _img_key(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read())
        return f"{os.path.basename(p)}:{h.hexdigest()[:16]}"

    key_src = json.dumps({
        "tag": tag, "prompt": prompt,
        "images": [_img_key(p) for p in image_paths or []],
    }, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()
    if key in _cache:
        return _cache[key]

    text = None
    last_err = None
    for attempt in range(3):
        try:
            text = _call_api(prompt, image_paths, max_tokens)
            if text is None:
                text = _call_cli(prompt, image_paths, max_tokens)
            if text:
                break
        except Exception as e:  # transient network/CLI hiccups: back off and retry
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"[llm] attempt {attempt + 1} failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    if text is None:
        raise RuntimeError(
            "No LLM backend available or all retries failed. Set ANTHROPIC_API_KEY, or "
            "install/login the Claude Code CLI (or point CLAUDE_CLI at the binary). "
            f"Cached runs need no backend. Last error: {last_err}"
        )
    _cache[key] = text
    _save_cache(_cache)
    return text


def extract_json(text):
    """Pull the first JSON object/array out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start_obj = text.find("{"); start_arr = text.find("[")
    starts = [s for s in (start_obj, start_arr) if s != -1]
    if not starts:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"Unbalanced JSON in response: {text[:200]}")
