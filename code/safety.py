"""Deterministic safety layer.

Runs BEFORE the LLM and resists prompt-injection attempts: matched patterns get a
hard verdict the LLM cannot override. Produces hard verdicts for the highest-risk
classes and risk annotations that are passed into the LLM prompt for everything else.
"""

import re

# Text that tries to give instructions TO the routing system itself.
INJECTION_PATTERNS = [
    r"notification router", r"routing (override|rule)", r"routing decision",
    r"system note for", r"internal router", r"router metadata",
    r"assistant instruction", r"ignore (all )?(previous|prior|sender)",
    r"mark (this )?(message )?as notify", r"set action\s*=", r"action\s*=\s*notify",
    r"classify (this )?as", r"confidence\s*=\s*1", r"mark notify",
]

# Requests for credentials/OTP or payment under pressure — classic scam mechanics.
CREDENTIAL_PATTERNS = [
    r"\botp\b", r"one[- ]time password", r"verification code", r"login code",
    r"\bpin\b", r"password", r"cvv", r"card number", r"bank details",
    r"account number", r"credentials", r"aadhaar", r"\bpan\b",
]

PRESSURE_PATTERNS = [
    r"account.{0,30}(block|lock|clos|restrict|suspend|hold|expire)",
    r"(block|lock|clos|restrict|suspend)ed?.{0,30}account",
    r"profile.{0,20}(band|block|restrict|lock)", r"immediately", r"within \d+ (min|hour)",
    r"\bjaldi\b", r"\babhi\b", r"warna", r"final (reminder|warning)", r"last chance",
    r"today only", r"before (midnight|tonight|evening|\d)", r"expires? (today|soon|tonight)",
    r"right now", r"in \d+ min", r"time kam",
]

PAYMENT_LURE_PATTERNS = [
    r"scan.{0,20}qr", r"pay.{0,30}(fee|charge|token|amount|clearance)",
    r"(processing|reattempt|clearance|reactivation|registration) (fee|charge|amount)",
    r"token (payment|booking|amount)", r"send screenshot.{0,30}(payment|after)",
    r"refund.{0,40}(verify|wallet|card|details|link)", r"claim.{0,30}(reward|prize|voucher|benefit)",
    r"selected for (a )?(reward|prize)", r"lottery", r"loan approved",
    r"(release|approved).{0,25}(amount|payout|refund)", r"assured appreciation",
    r"registry papers.{0,30}after (payment|token)",
]

SUSPICIOUS_LINK_PATTERNS = [
    r"bit\.ly", r"tinyurl", r"shorturl", r"weurl", r"vl\.gl", r"wame\.pro",
    r"[a-z0-9-]+-(secure|alert|verify|refund|kyc|reward|helpdesk|delivery|payouts?|simkyc|billpay|gift)[a-z0-9-]*\.(com|in|net|co)",
    r"(account-login|account-help|pay-check-secure|verify-quick)\.[a-z]+",
]

CHAIN_FORWARD_PATTERNS = [
    r"forward (this )?to \d+", r"share (this |it )?(with|to|in) (all|family|\d+|ten|everyone)",
    r"do( not|n't) (ignore|break the chain)", r"good luck", r"blessing", r"positive energy",
    r"send (this )?to (all|ten|10|everyone)", r"bhagwan", r"share kar", r"failao",
    r"forwarding because", r"fwd as received", r"share in (all )?family groups",
]

HEALTH_MISINFO_PATTERNS = [
    r"stop (all )?(tablets|medicines|medication)", r"avoid (all )?tablets",
    r"herbal mix", r"ancient remedy", r"doctors don'?t", r"one habit.{0,20}(fix|cure)",
    r"drink warm water every hour", r"miracle (cure|remedy)",
]


def _any(patterns, text):
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def analyze(msg, business=None, media_text=""):
    """Return a dict of deterministic risk findings for one message."""
    text = " ".join(filter(None, [msg.get("message_text", ""), media_text]))
    low = text.lower()

    findings = {
        "injection": _any(INJECTION_PATTERNS, low),
        "credential_request": _any(CREDENTIAL_PATTERNS, low),
        "pressure": _any(PRESSURE_PATTERNS, low),
        "payment_lure": _any(PAYMENT_LURE_PATTERNS, low),
        "suspicious_link": _any(SUSPICIOUS_LINK_PATTERNS, low),
        "chain_forward": _any(CHAIN_FORWARD_PATTERNS, low),
        "health_misinfo": _any(HEALTH_MISINFO_PATTERNS, low),
        "impersonating_business": False,
        "hard_verdict": None,   # (action, message_type) when deterministic
        "notes": [],
    }

    if business is not None:
        verified = business.get("verified") == "1"
        official = business.get("official_domain") or ""
        used = business.get("domain_used_by_sender") or ""
        young = int(business.get("account_age_days") or "0") < 60
        reports = int(business.get("user_reports_30d") or "0")
        domain_mismatch = bool(official) and used != official
        if not verified and (young or domain_mismatch or not official) and reports >= 10:
            findings["impersonating_business"] = True
            findings["notes"].append(
                f"Business is unverified, {business.get('account_age_days')}d old, uses "
                f"'{used or 'no domain'}' instead of '{official or 'any official domain'}', "
                f"{reports} reports in 30d — impersonation profile."
            )

    # --- hard verdicts (deterministic; the LLM cannot override them) ---
    if findings["injection"]:
        findings["hard_verdict"] = ("mute", "scam")
        findings["notes"].append("Message attempts to instruct the notification router itself.")
    elif findings["credential_request"] and (findings["pressure"] or findings["suspicious_link"] or findings["impersonating_business"]):
        findings["hard_verdict"] = ("mute", "scam")
        findings["notes"].append("Credential/OTP request combined with pressure, suspicious link, or impersonation.")
    elif findings["impersonating_business"] and (findings["payment_lure"] or findings["pressure"] or findings["suspicious_link"] or findings["credential_request"]):
        findings["hard_verdict"] = ("mute", "scam")
        findings["notes"].append("Impersonating business pushing payment/verification.")

    findings["risky"] = any([
        findings["injection"], findings["credential_request"], findings["payment_lure"],
        findings["suspicious_link"], findings["impersonating_business"],
    ])
    return findings


def summary_for_prompt(findings):
    """Compact, human-readable risk block for the LLM prompt."""
    out = []
    for key, label in [
        ("injection", "router-instruction injection"),
        ("credential_request", "asks for OTP/credentials"),
        ("pressure", "urgency/blocking pressure"),
        ("payment_lure", "payment/fee/prize lure"),
        ("suspicious_link", "suspicious or lookalike link"),
        ("chain_forward", "chain/forward broadcast wording"),
        ("health_misinfo", "health misinformation"),
    ]:
        if findings.get(key):
            out.append(label)
    if findings.get("impersonating_business"):
        out.append("business impersonation profile")
    for n in findings.get("notes", []):
        out.append(n)
    return "; ".join(out) if out else "no deterministic risk flags"
