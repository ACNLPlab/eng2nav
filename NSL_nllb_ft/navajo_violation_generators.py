"""
navajo_violation_generators.py
================================
Violation generators for NSL-MT training.

Works on ALL data — both morphosyntactically tagged (gloss file) and plain
parallel data (Bible / user sentences).  Generators use surface heuristics
that are conservative enough to fire reliably without morpheme tags, and
upgrade to tag-confirmed detection when gloss data is available.

VIOLATION TYPES (6 total, 2 per category)
------------------------------------------
MORPHOLOGICAL
  1. da_removal         — strip da- DISTR distributive prefix
                          surface : word starts with [Dd]a + ≥3 remaining chars,
                                    not a blocklisted closed-class word
                          gloss   : confirm morph='da-' tag contains 'DISTR'
                          severity: 0.7

  2. possessive_swap    — swap bi- (3.BI.POSS "his/her/its") → shi- ("my")
                          Breaks referential/possessor agreement throughout.
                          surface : word starts with [Bb]i, length ≥ 4,
                                    not in hard blocklist
                          gloss   : confirm tag contains 'BI.POSS' or '3.BI'
                                    on a morph starting with 'bi'
                          severity: 0.8

SYNTACTIC
  3. go_removal         — strip =go subordinator clitic
                          surface : word ends in 'go', total length ≥ 4
                          gloss   : confirm morph in ('=go','go'), tag 'SUB'
                          severity: 0.9

  4. daa_removal        — strip =dą́ą́' past temporal suffix
                          Removes temporal anchor from time expressions.
                          surface : word ends with exact Unicode string dą́ą́'
                                    and prefix is ≥ 2 chars
                          gloss   : confirm morph '=dą́ą́'' tag 'PST'
                          severity: 0.8

LEXICAL
  5. negation_break     — remove closing 'da' of doo...da negation frame
                          surface : 'doo' present, last standalone 'da'
                                    after it is removed
                          gloss   : prefer NEG-tagged 'da' word after 'doo'
                          severity: 0.9

  6. tense_removal      — remove standalone ńt'éé' / ńt'ée' past-tense word
                          surface : exact token match (closed vocabulary)
                          gloss   : confirm single morph tagged PST
                          severity: 0.8

WHY THESE SIX (NOT SVO)
-----------------------
SVO reordering was excluded: detecting Navajo finite verbs on untagged surface
text is unreliable — verb-final forms have dozens of different endings, so the
generator would miss most verbs or make systematic false fires on nouns.
The possessive swap replaces it: bi- words are very common in Bible/narrative
text, the surface rule is high-precision, and possessor agreement violation is
linguistically distinct from all other five types.

PUBLIC API
----------
  parse_gloss_file(path)              → list[dict]   tagged gloss records
  make_untagged_record(eng, nav)      → dict         plain pair as record
  SEVERITY                            dict[str,float]
  _ALL_GENERATORS                     list of callables
  generate_all_violations(record)     → list[dict]   for inspection/testing
"""

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Severity weights
# ---------------------------------------------------------------------------
SEVERITY = {
    "da_removal":      0.7,
    "possessive_swap": 0.8,
    "go_removal":      0.9,
    "daa_removal":     0.8,
    "negation_break":  0.9,
    "tense_removal":   0.8,
}

# ---------------------------------------------------------------------------
# Closed-vocabulary constants
# ---------------------------------------------------------------------------

PST_TOKENS  = {"ńt'éé'", "ńt'ée'"}
DAA_SUFFIX  = "dą́ą́'"

# Blocklisted words for da_removal: standalone closed-class words starting
# with 'da' that are NOT distributive-prefix forms
DA_BLOCKLIST = frozenset({
    "da", "dah", "daʼ", "doo", "daashin",
})

# Words starting with 'bi' that are directional postpositions, not
# bi-POSS + stem — swap would be meaningless or very low precision
BI_HARD_BLOCKLIST = frozenset({
    "biih",   # "into it" — directional postposition
})

DA_MIN_REMAINDER = 3   # chars remaining after stripping 'da'
BI_MIN_LENGTH    = 4   # minimum total length of bi- word to swap
GO_MIN_LENGTH    = 4   # minimum total length of =go word to strip
DAA_MIN_PREFIX   = 2   # minimum chars before =dą́ą́' suffix


# ---------------------------------------------------------------------------
# GLOSS FILE PARSER
# ---------------------------------------------------------------------------

def parse_gloss_file(filepath: str) -> list:
    """
    Parse a morphosyntactic gloss file into sentence records.

    Blocks separated by lines of ={10,} characters.
    Returns list of dicts:
      {
        "sentence_nv": str,
        "sentence_en": str,
        "words": [{"word_nv": str, "gloss_en": str,
                   "morphs": [{"morph": str, "tag": str}, ...]}, ...],
        "_has_gloss": True
      }
    Blocks lacking both Navajo and English strings are skipped.
    """
    text   = Path(filepath).read_text(encoding="utf-8")
    blocks = re.split(r"={10,}", text)
    records = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m_nv = re.search(r"Sentence \(nv\):\s*(.+)", block)
        m_en = re.search(r"Sentence gloss \(en\):\s*(.+)", block)
        if not m_nv or not m_en:
            continue

        record = {
            "sentence_nv": m_nv.group(1).strip(),
            "sentence_en": m_en.group(1).strip(),
            "words":       [],
            "_has_gloss":  True,
        }

        for ws in re.split(r"\n(?=Word \(nv\):)", block):
            wm = re.search(r"Word \(nv\):\s*(.+)", ws)
            if not wm:
                continue
            gm = re.search(r"Gloss \(en\):\s*(.+)", ws)
            entry = {
                "word_nv":  wm.group(1).strip(),
                "gloss_en": gm.group(1).strip() if gm else "",
                "morphs":   [],
            }
            for ml in re.finditer(r"Morph:\s*(\S+)\s*→\s*(.+)", ws):
                entry["morphs"].append({
                    "morph": ml.group(1).strip(),
                    "tag":   ml.group(2).strip(),
                })
            record["words"].append(entry)

        records.append(record)

    return records


def make_untagged_record(sentence_en: str, sentence_nv: str) -> dict:
    """
    Wrap a plain (eng, nav) pair in the unified record structure.
    Generators use surface heuristics only (_has_gloss=False).
    """
    return {
        "sentence_en": sentence_en,
        "sentence_nv": sentence_nv,
        "words":       [
            {"word_nv": w, "gloss_en": "", "morphs": []}
            for w in sentence_nv.split()
        ],
        "_has_gloss":  False,
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _tag_contains(word_entry: dict, substr: str) -> bool:
    up = substr.upper()
    return any(up in m["tag"].upper() for m in word_entry["morphs"])


def _any_morph_equals(word_entry: dict, morph: str, tag_substr: str) -> bool:
    return any(
        m["morph"] == morph and tag_substr.upper() in m["tag"].upper()
        for m in word_entry["morphs"]
    )


# ---------------------------------------------------------------------------
# GENERATORS
# ---------------------------------------------------------------------------

def _da_removal(record: dict) -> Optional[dict]:
    """Strip da- (DISTR) distributive prefix from qualifying words."""
    has_gloss = record["_has_gloss"]
    modified  = [w["word_nv"] for w in record["words"]]
    changed   = False

    for i, we in enumerate(record["words"]):
        w = we["word_nv"]
        if has_gloss:
            if not _any_morph_equals(we, "da-", "DISTR"):
                continue
        else:
            if not re.match(r"^[Dd]a", w):
                continue
            if w.lower() in DA_BLOCKLIST:
                continue
            if len(w[2:]) < DA_MIN_REMAINDER:
                continue

        stripped = w[2:]
        if stripped:
            modified[i] = stripped
            changed = True

    if not changed:
        return None
    violated = " ".join(modified)
    if violated == record["sentence_nv"]:
        return None
    return {
        "type":     "da_removal",
        "sentence": violated,
        "severity": SEVERITY["da_removal"],
        "note":     "da- (DISTR) stripped — plural/distributive collapsed to singular",
    }


def _possessive_swap(record: dict) -> Optional[dict]:
    """
    Swap bi- (3rd person possessive) → shi- (1st person possessive).

    All bi- words in the sentence are swapped for consistency — a partial
    swap would be an internally inconsistent violation with weaker signal.
    """
    has_gloss = record["_has_gloss"]
    modified  = [w["word_nv"] for w in record["words"]]
    changed   = False

    for i, we in enumerate(record["words"]):
        w = we["word_nv"]

        if has_gloss:
            # Require a morph with BI.POSS or 3.BI tag AND word starts with bi
            is_bi_poss = (
                re.match(r"^[Bb]i", w) and
                any(
                    ("BI.POSS" in m["tag"].upper()) or
                    ("3.BI" in m["tag"].upper() and m["morph"].lower() in ("bi-", "b-"))
                    for m in we["morphs"]
                )
            )
            if not is_bi_poss:
                continue
        else:
            if not re.match(r"^[Bb]i", w):
                continue
            if w.lower() in BI_HARD_BLOCKLIST:
                continue
            if len(w) < BI_MIN_LENGTH:
                continue

        # bi → shi (first 2 chars replaced)
        modified[i] = "shi" + w[2:]
        changed = True

    if not changed:
        return None
    violated = " ".join(modified)
    if violated == record["sentence_nv"]:
        return None
    return {
        "type":     "possessive_swap",
        "sentence": violated,
        "severity": SEVERITY["possessive_swap"],
        "note":     "bi- (3.BI.POSS) → shi- (1SG.POSS) — possessor agreement broken",
    }


def _go_removal(record: dict) -> Optional[dict]:
    """Strip =go subordinator clitic, breaking clause-chaining."""
    has_gloss = record["_has_gloss"]
    modified  = [w["word_nv"] for w in record["words"]]
    changed   = False

    for i, we in enumerate(record["words"]):
        w = we["word_nv"]
        if has_gloss:
            has_sub = any(
                m["morph"] in ("=go", "go") and "SUB" in m["tag"].upper()
                for m in we["morphs"]
            )
            if not has_sub:
                continue
        else:
            if not (w.endswith("go") and len(w) >= GO_MIN_LENGTH):
                continue

        modified[i] = w[:-2]
        changed = True

    if not changed:
        return None
    violated = re.sub(r"  +", " ", " ".join(modified)).strip()
    if violated == record["sentence_nv"]:
        return None
    return {
        "type":     "go_removal",
        "sentence": violated,
        "severity": SEVERITY["go_removal"],
        "note":     "=go (SUB) stripped — subordinate clause-chaining broken",
    }


def _daa_removal(record: dict) -> Optional[dict]:
    """
    Strip =dą́ą́' past temporal suffix from time-reference words.

    Examples:
      Tł'éédą́ą́' (that night) → Tł'éé'
      'Íídą́ą́'   (back then)  → 'Íí
    Removes the past temporal anchor, stripping discourse-level
    time grounding from the clause.
    """
    has_gloss = record["_has_gloss"]
    modified  = [w["word_nv"] for w in record["words"]]
    changed   = False

    for i, we in enumerate(record["words"]):
        w = we["word_nv"]

        if has_gloss:
            has_daa = any(
                m["morph"] == "=dą́ą́'" and "PST" in m["tag"].upper()
                for m in we["morphs"]
            )
            if not has_daa:
                continue
        else:
            if not w.endswith(DAA_SUFFIX):
                continue
            prefix = w[: -len(DAA_SUFFIX)]
            if len(prefix) < DAA_MIN_PREFIX:
                continue

        prefix = w[: -len(DAA_SUFFIX)]
        if prefix:
            modified[i] = prefix
            changed = True

    if not changed:
        return None
    violated = re.sub(r"  +", " ", " ".join(modified)).strip()
    if violated == record["sentence_nv"]:
        return None
    return {
        "type":     "daa_removal",
        "sentence": violated,
        "severity": SEVERITY["daa_removal"],
        "note":     "=dą́ą́' (PST) stripped — past temporal anchor removed",
    }


def _negation_break(record: dict) -> Optional[dict]:
    """
    Remove closing 'da' of the doo...da discontinuous negation frame.

    Inverts the polarity of the sentence — high-impact, high-precision.
    """
    sentence  = record["sentence_nv"]
    words     = record["words"]
    has_gloss = record["_has_gloss"]

    if not any(w["word_nv"].lower() == "doo" for w in words):
        return None

    if has_gloss:
        doo_seen = False
        neg_idx  = None
        for i, we in enumerate(words):
            if we["word_nv"].lower() == "doo":
                doo_seen = True
            if doo_seen and we["word_nv"].lower() == "da" and _tag_contains(we, "NEG"):
                neg_idx = i
        if neg_idx is not None:
            modified = [w["word_nv"] for j, w in enumerate(words) if j != neg_idx]
            violated = re.sub(r"  +", " ", " ".join(modified)).strip()
            if violated != sentence:
                return {
                    "type":     "negation_break",
                    "sentence": violated,
                    "severity": SEVERITY["negation_break"],
                    "note":     "Closing 'da' (NEG) removed — doo...da frame broken",
                }

    # Surface fallback (also used when _has_gloss=False)
    doo_pos = sentence.lower().find("doo")
    if doo_pos == -1:
        return None
    after_doo  = sentence[doo_pos:]
    da_matches = list(re.finditer(r"\bda\b", after_doo))
    if not da_matches:
        return None
    last  = da_matches[-1]
    abs_s = doo_pos + last.start()
    abs_e = doo_pos + last.end()
    violated = re.sub(r"  +", " ",
                      (sentence[:abs_s] + sentence[abs_e:]).strip())
    if violated == sentence:
        return None
    return {
        "type":     "negation_break",
        "sentence": violated,
        "severity": SEVERITY["negation_break"],
        "note":     "Closing 'da' removed (surface) — doo...da negation broken",
    }


def _tense_removal(record: dict) -> Optional[dict]:
    """Remove standalone ńt'éé' / ńt'ée' past-tense words."""
    has_gloss = record["_has_gloss"]
    modified  = []
    changed   = False

    for we in record["words"]:
        w = we["word_nv"]
        if w not in PST_TOKENS:
            modified.append(w)
            continue
        if has_gloss:
            is_pure_pst = (
                len(we["morphs"]) == 1
                and "PST" in we["morphs"][0]["tag"].upper()
            )
            if not is_pure_pst:
                modified.append(w)
                continue
        changed = True   # omit this word

    if not changed:
        return None
    violated = re.sub(r"  +", " ", " ".join(modified)).strip()
    if violated == record["sentence_nv"]:
        return None
    return {
        "type":     "tense_removal",
        "sentence": violated,
        "severity": SEVERITY["tense_removal"],
        "note":     "ńt'éé' (PST) removed — past tense marking stripped",
    }


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

_ALL_GENERATORS = [
    _da_removal,        # MORPH 1 — distributive prefix
    _possessive_swap,   # MORPH 2 — 3rd→1st person possessive
    _go_removal,        # SYN   1 — subordinator clitic
    _daa_removal,       # SYN   2 — past temporal suffix
    _negation_break,    # LEX   1 — doo...da negation frame
    _tense_removal,     # LEX   2 — standalone PST word
]


def generate_all_violations(record: dict) -> list:
    """Run all generators; return every applicable violation (for inspection)."""
    return [v for g in _ALL_GENERATORS if (v := g(record)) is not None]
