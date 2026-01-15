# src/utils.py
from __future__ import annotations

import re
import unicodedata
from typing import List


# -----------------------------
# Regex helpers (compiled once)
# -----------------------------

# Sentence boundary split (robust enough for most backstories)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Break long sentences further on discourse/conjunction markers
# Keep this conservative to avoid destroying meaning.
_SPLIT_MARKERS = re.compile(
    r"\b("
    r"and|but|while|because|although|however|though|yet|whereas|since|unless|"
    r"therefore|thus|hence"
    r")\b",
    re.IGNORECASE,
)

# Remove bullets / list symbols
_BULLET_PREFIX = re.compile(r"^\s*[-•*]+\s*")

# Normalize whitespace
_WS = re.compile(r"\s+")

# If the line is obviously a heading or too short
_TOO_SHORT = re.compile(r"^\W*$")

# Leading pronouns to ground
_LEADING_PRONOUN = re.compile(
    r"^(he|she|they|his|her|their|him|them)\b", re.IGNORECASE
)


# -----------------------------
# Public utilities
# -----------------------------

def slugify(text: str) -> str:
    """
    Convert a string into a stable, folder-friendly slug.
    Example: "The Count of Monte Cristo" -> "the_count_of_monte_cristo"
    """
    text = unicodedata.normalize("NFKD", (text or "").strip())
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ground_claim(claim: str, char_name: str) -> str:
    """
    Replace leading ambiguous pronouns with a character name.
    Helps retrieval + NLI stay focused on the right entity.

    Examples:
      "He was born in Paris." + "Edmond Dantès" -> "Edmond Dantès was born in Paris."
      "His father died early." -> "Edmond Dantès father died early."
    """
    claim = (claim or "").strip()
    if not claim or not char_name:
        return claim

    # Replace only at the start (do NOT replace pronouns mid-sentence)
    grounded = _LEADING_PRONOUN.sub(char_name, claim, count=1)

    # Light grammar fix: if we replaced "His" -> "Edmond", we often want "Edmond's".
    # But forcing possessive can hurt sometimes, so we keep it simple.
    # If you want possessive, uncomment:
    # grounded = re.sub(rf"^{re.escape(char_name)}\s+(father|mother|sister|brother|wife|husband)\b",
    #                   rf"{char_name}'s \1", grounded, flags=re.IGNORECASE)

    return grounded


def split_into_claims(
    text: str,
    max_words: int = 28,
    min_words: int = 4,
    max_claims: int | None = None,
) -> List[str]:
    """
    Convert a long backstory into more "atomic" claims suitable for NLI.

    Strategy:
    1) Split into sentences.
    2) If a sentence is long, split further on safe discourse markers.
    3) Clean bullets / whitespace.
    4) Drop fragments that are too short.
    5) De-duplicate while preserving order.

    Notes:
    - This is a heuristic splitter (not perfect).
    - It is intentionally conservative to avoid breaking meaning.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Normalize newlines and bullets
    lines = [ln.strip() for ln in text.splitlines() if not _TOO_SHORT.match(ln)]
    cleaned_lines = []
    for ln in lines:
        ln = _BULLET_PREFIX.sub("", ln)
        ln = _WS.sub(" ", ln).strip()
        if ln:
            cleaned_lines.append(ln)

    normalized = " ".join(cleaned_lines).strip()
    if not normalized:
        return []

    # 1) Sentence split
    sents = _SENT_SPLIT.split(normalized)

    # 2) Further split long sentences
    raw_claims: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue

        words = s.split()
        if len(words) <= max_words:
            raw_claims.append(s)
            continue

        # Split on markers but keep only meaningful parts (drop marker tokens themselves)
        parts = _SPLIT_MARKERS.split(s)

        # parts alternates: [chunk, marker, chunk, marker, chunk...]
        # we only want the chunks; markers are in odd indices
        chunks = [p.strip(" ,;:") for idx, p in enumerate(parts) if idx % 2 == 0]

        # If splitting produced junk, fallback to the original
        if len(chunks) <= 1:
            raw_claims.append(s)
        else:
            for c in chunks:
                c = c.strip()
                if c:
                    raw_claims.append(c)

    # 3) Filter + final cleanup
    filtered: List[str] = []
    for c in raw_claims:
        c = _WS.sub(" ", c).strip(" \t\n-•*")
        if not c:
            continue
        if len(c.split()) < min_words:
            continue
        filtered.append(c)

    # 4) De-duplicate preserving order
    seen = set()
    out: List[str] = []
    for c in filtered:
        key = _WS.sub(" ", c.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

        if max_claims is not None and len(out) >= max_claims:
            break

    return out
