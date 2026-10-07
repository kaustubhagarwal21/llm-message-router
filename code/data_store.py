"""Data layer: loads all dataset CSVs and builds a per-message context dossier.

Every routing decision is personalized. The dossier for a message joins:
  - the receiving user's notification behavior
  - group metadata + the user's relationship with that group
  - business metadata (verification, domain trust, reports) + user-business history
  - the user's historical messages from the same sender/business/group
  - how the user reacted to those historical messages (events)
  - retrieval of similar historical messages for evidence
"""

import csv
import os
import re
from collections import defaultdict

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "at", "for", "and", "or", "if", "it", "this", "that", "your", "you",
    "we", "our", "us", "i", "me", "my", "with", "from", "by", "as", "so", "do",
    "not", "no", "will", "can", "may", "please", "pls", "hi", "dear", "team",
    "customer", "tap", "below", "view", "details", "reply", "stop", "opt", "out",
    "unsubscribe", "now", "today", "before", "after", "have", "has", "had",
}


def _read_csv(name):
    path = os.path.join(DATASET_DIR, name)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


class DataStore:
    def __init__(self):
        self.messages = _read_csv("messages.csv")
        self.samples = _read_csv("sample_messages.csv")
        self.users = {r["user_id"]: r for r in _read_csv("users.csv")}
        self.groups = {r["group_id"]: r for r in _read_csv("groups.csv")}
        self.businesses = {r["business_id"]: r for r in _read_csv("business_accounts.csv")}
        self.images = {r["image_id"]: r["file_path"] for r in _read_csv("images.csv")}
        self.voice_notes = {r["voice_note_id"]: r["file_path"] for r in _read_csv("voice_notes.csv")}
        self.output_template = _read_csv("output.csv")

        self.group_members = {}
        for r in _read_csv("group_members.csv"):
            self.group_members[(r["group_id"], r["user_id"])] = r

        self.user_business = {}
        for r in _read_csv("user_business_history.csv"):
            self.user_business[(r["user_id"], r["business_id"])] = r

        self.history = _read_csv("message_history.csv")
        self.history_by_id = {r["message_id"]: r for r in self.history}
        self.history_by_user = defaultdict(list)
        for r in self.history:
            self.history_by_user[r["user_id"]].append(r)
        for rows in self.history_by_user.values():
            rows.sort(key=lambda r: r["created_at"])

        self.events = {}
        for r in _read_csv("message_events.csv"):
            self.events[r["message_id"]] = r

        self.daily_load = defaultdict(list)
        for r in _read_csv("daily_notification_summary.csv"):
            self.daily_load[r["user_id"]].append(r)

        self._token_cache = {
            r["message_id"]: tokenize(r["message_text"]) for r in self.history
        }

    # ---------- helpers ----------

    def event_summary(self, message_id):
        ev = self.events.get(message_id)
        if not ev:
            return "no reaction data"
        parts = []
        if ev["message_opened"] == "1":
            parts.append("opened")
        if ev["message_replied"] == "1":
            parts.append("replied")
        if ev.get("reaction_time_minutes"):
            parts.append(f"reaction {ev['reaction_time_minutes']}min")
        if ev["notification_dismissed"] == "1":
            parts.append("dismissed")
        if ev["muted_after_message"] == "1":
            parts.append("muted-after")
        if ev["message_reported"] == "1":
            parts.append("REPORTED")
        return ", ".join(parts) if parts else "ignored (no open)"

    def similar_history(self, user_id, text, sender_user_id="", business_id="", group_id="", top_k=6):
        """Rank this user's historical messages by relevance to the incoming one.

        Same-sender/business/group messages get a bonus so evidence points at the
        pattern that actually justifies the decision.
        """
        q = tokenize(text)
        scored = []
        for r in self.history_by_user.get(user_id, []):
            overlap = 0.0
            toks = self._token_cache.get(r["message_id"], set())
            if q and toks:
                overlap = len(q & toks) / max(len(q | toks), 1)
            bonus = 0.0
            if sender_user_id and r.get("sender_user_id") == sender_user_id:
                bonus += 0.35
            if business_id and r.get("business_id") == business_id:
                bonus += 0.45
            if group_id and r.get("group_id") == group_id:
                bonus += 0.15
            if r.get("media_id") and r["media_id"] in (text or ""):
                bonus += 0.2
            score = overlap + bonus
            if score > 0.05:
                scored.append((score, r))
        scored.sort(key=lambda t: (-t[0], t[1]["message_id"]))
        return [r for _, r in scored[:top_k]]

    def build_dossier(self, msg, media_notes=None):
        """Assemble the full personalized context block for one incoming message."""
        user_id = msg["user_id"]
        lines = []

        u = self.users.get(user_id)
        if u:
            lines.append(
                f"RECEIVING USER {user_id}: opened {u['messages_opened_30d']} msgs/30d, "
                f"replied {u['messages_replied_30d']}, dismissed {u['notifications_dismissed_30d']}, "
                f"reported {u['messages_reported_30d']}. DND window {u['do_not_disturb_window']}."
            )
            opened = int(u["messages_opened_30d"]); dismissed = int(u["notifications_dismissed_30d"])
            if dismissed > opened:
                lines.append("  -> This user dismisses more notifications than they open (low tolerance).")

        gid = msg.get("group_id", "")
        if gid and gid in self.groups:
            g = self.groups[gid]
            lines.append(
                f"GROUP {gid} '{g['group_name']}' (type={g['group_type']}, {g['member_count']} members, "
                f"{g['admin_count']} admins, {g['messages_30d']} msgs/30d)."
            )
            gm = self.group_members.get((gid, user_id))
            if gm:
                lines.append(
                    f"  User in group: role={gm['role']}, read {gm['messages_read_30d']}/30d, "
                    f"replied {gm['replies_sent_30d']}, dismissed {gm['notifications_dismissed_30d']}, "
                    f"group_muted_by_user={'YES' if gm['group_muted_by_user']=='1' else 'no'}."
                )
            sender = msg.get("sender_user_id", "")
            sm = self.group_members.get((gid, sender)) if sender else None
            if sm:
                lines.append(
                    f"  Sender {sender} in group: role={sm['role']}"
                    f"{' (GROUP ADMIN)' if sm['role']=='admin' else ''}, "
                    f"sent {sm['messages_sent_30d']} msgs/30d, joined {sm['joined_at']}."
                )

        bid = msg.get("business_id", "")
        if bid and bid in self.businesses:
            b = self.businesses[bid]
            dom_match = (b["official_domain"] and b["official_domain"] == b["domain_used_by_sender"])
            lines.append(
                f"BUSINESS {bid} '{b['display_name']}' (brand={b['brand_name']}, category={b['category']}): "
                f"verified={'YES' if b['verified']=='1' else 'NO'}, "
                f"official_domain={b['official_domain'] or 'none'}, "
                f"sender_domain={b['domain_used_by_sender'] or 'none'} "
                f"({'MATCHES official' if dom_match else 'DOES NOT match official domain'}), "
                f"account_age={b['account_age_days']}d, reports_30d={b['user_reports_30d']}."
            )
            if b["verified"] != "1" and int(b["account_age_days"]) < 60:
                lines.append(
                    "  -> UNVERIFIED business, very new account"
                    + (", lookalike/mismatched domain" if not dom_match else "")
                    + f", {b['user_reports_30d']} user reports: strong impersonation-scam signal."
                )
            ub = self.user_business.get((user_id, bid))
            if ub:
                lines.append(
                    f"  User-business history: knows account via '{ub['why_user_knows_account']}', "
                    f"last_activity={ub['last_activity_at'] or 'none'}, "
                    f"allows_promotions={'YES' if ub['allows_promotions']=='1' else 'NO'}"
                    + (f", OPTED OUT of promotions at {ub['promotions_opted_out_at']}" if ub["promotions_opted_out_at"] else "")
                    + f", opened {ub['messages_opened_30d']}/30d, dismissed {ub['messages_dismissed_30d']}/30d, "
                    f"replied {ub['messages_replied_30d']}/30d."
                )
            else:
                lines.append("  User-business history: NONE — user has no known relationship with this business.")

        sender = msg.get("sender_user_id", "")
        if msg.get("conversation_type") == "personal" and sender:
            past_from_sender = [
                r for r in self.history_by_user.get(user_id, []) if r.get("sender_user_id") == sender
            ]
            if past_from_sender:
                lines.append(f"SENDER {sender}: {len(past_from_sender)} past personal/group messages to this user (see history below).")
            else:
                lines.append(f"SENDER {sender}: FIRST CONTACT — no message history with this user.")

        sim = self.similar_history(
            user_id, msg.get("message_text", ""),
            sender_user_id=sender, business_id=bid, group_id=gid,
        )
        if media_notes:
            lines.append(f"MEDIA CONTENT ({msg.get('media_type')}): {media_notes}")
        if sim:
            lines.append("RELEVANT MESSAGE HISTORY for this user (id | when | from | text | user reaction):")
            for r in sim:
                src = r.get("business_id") or r.get("sender_user_id") or "?"
                grp = f" in {r['group_id']}" if r.get("group_id") else ""
                text = (r.get("message_text") or f"[{r.get('media_type')} {r.get('media_id')}]").replace("\n", " ")
                if len(text) > 160:
                    text = text[:157] + "..."
                lines.append(f"  {r['message_id']} | {r['created_at']} | {src}{grp} | {text} | [{self.event_summary(r['message_id'])}]")
        else:
            lines.append("RELEVANT MESSAGE HISTORY: none found for this user.")

        fwd = msg.get("forwarded_count", "0")
        if fwd and int(fwd) >= 5:
            lines.append(f"FORWARD SIGNAL: forwarded_count={fwd} (heavily forwarded chain/broadcast content).")

        return "\n".join(lines)
