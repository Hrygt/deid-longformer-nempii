"""Structured-identifier recognizer seed for clinical-deid (hybrid evolution).

Curated from UCSF Philter pattern files (BSD 3-Clause, Regents of the
University of California; Norgeot et al., npj Digital Medicine 2020) and
adapted to clinical-deid's entity families, Epic note structure, and the
failure classes recorded in the 21-note ICU pilot (2026-07-05 evidence pulls).

STATUS: planning seed. Not wired into the service. Emits spans in the
service's {type, start, end, text} shape so a future branch can union these
with NER output ahead of the merge passes. SAFE spans are vetoes: they claim
text that no detector (regex or NER) may surrogate.

Design rules carried from Philter:
  1. Value-bounded fields, never lazy digit counts (a month is (0?[1-9]|1[0-2]),
     not \\d{1,2}).
  2. Safe patterns run first and win.
  3. One named pattern per real-world shape, with provenance in the comment.

Python 3.11+, stdlib re only. Self-checks: python structured_recognizers_seed.py
"""

from __future__ import annotations

import re
import sys

# ---------------------------------------------------------------------------
# Shared bounded vocabulary (adapted from philter dates/*_transformed.txt)
# ---------------------------------------------------------------------------
_MM = r"(?:0?[1-9]|1[0-2])"                    # bounded month
_DD = r"(?:[1-2][0-9]|3[0-1]|0?[1-9])"         # bounded day
_YYYY = r"(?:19|20)\d{2}"                      # bounded-century year
_YY = r"\d{2}"
_STATE = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|"
    "MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|"
    "WI|WY|DC"
)

# ---------------------------------------------------------------------------
# SAFE patterns (exclude=false in Philter terms). These veto detections.
# Each exists for a recorded false-positive class in OUR pilot.
# ---------------------------------------------------------------------------
SAFE_PATTERNS = {
    # Pilot sites 4/5 + note 3: HHMM clock times in vitals/procedure rows.
    # NER tagged the trailing 2 digits as DATE; claiming the whole token as
    # SAFE-TIME kills the class at detection. Bounded to valid clock values.
    "clock_hhmm": re.compile(r"(?<![\d/:-])(?:[01]\d|2[0-3])[0-5]\d(?![\d/:-])"),
    # Notes 11/12: RM06/RM08 room tokens mis-tagged POSTCODE.
    "room_token": re.compile(r"\bRM\d{2}\b"),
    # Philter bp_safe, adapted: BP readings so date/ratio patterns cannot eat
    # 120/80. Provenance: filters/regex/safe/bp_safe.txt (bounds kept).
    "bp_reading": re.compile(
        r"\b(?i:bp)\:?\s{0,2}(?:[6-9][0-9]|1[0-9][0-9])/(?:[2-9][0-9]|1[0-9][0-9])\b"
    ),
}

# ---------------------------------------------------------------------------
# PHI recognizer patterns (exclude=true). Family names match deid.py replace().
# ---------------------------------------------------------------------------
PHI_PATTERNS = {
    # Provenance: philter addresses/city_state_zip_transformed.txt, reduced to
    # the state-abbr core our Epic blocks use; full-name states dropped (not
    # observed in Epic address lines). Catches the 0-for-7 ZIP+4 class.
    "POSTCODE": re.compile(
        r"\b(?:" + _STATE + r")\s{1,4}(\d{5}(?:-\d{4})?)(?![-\d])"
    ),
    # Provenance: philter contact/xxx_xxx_xxxx; NANP-bounded area code.
    "PHONE_NUMBER": re.compile(
        r"(?<!\d)\(?[2-9]\d{2}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
    ),
    # SSN, strict hyphenated form only (unhyphenated 9-digit runs are owned by
    # other layers per the Branch 5 non-match ruling).
    "SSN": re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Rigid numeric dates, value-bounded (philter mm_dd_yyyy / mm_dd_yy).
    # Written-date forms stay NER-owned for now; this recognizer only claims
    # shapes where regex recall is effectively perfect.
    "DATE": re.compile(
        r"(?<![\d/])"
        rf"{_MM}[/-]{_DD}[/-](?:{_YYYY}|{_YY})"
        r"(?![\d/])"
    ),
}

# Maximal-token DATE grammar for Branch 1's edge trim: a DATE span's end must
# not extend past a match of this anchored grammar, and a DATE span containing
# no digits is dropped. Kept separate from PHI_PATTERNS because Branch 1 lives
# at the substitution layer, not detection.
DATE_MAXIMAL_TOKEN = re.compile(
    rf"^{_MM}[/-]{_DD}[/-](?:{_YYYY}|{_YY})"
    rf"(?:\s{{1,2}}(?:[01]\d|2[0-3]):?[0-5]\d)?"   # optional folded time
)


def find_spans(text: str) -> tuple[list[dict], list[dict]]:
    """Return (safe_spans, phi_spans) in the service span shape.

    Overlap policy (Philter order): SAFE claims first; a PHI span overlapping
    any SAFE span is discarded. Within PHI, earlier/longer match wins.
    """
    safe: list[dict] = []
    for name, pat in SAFE_PATTERNS.items():
        for m in pat.finditer(text):
            safe.append({"type": f"SAFE_{name.upper()}", "start": m.start(),
                         "end": m.end(), "text": m.group(0)})
    taken = [(s["start"], s["end"]) for s in safe]

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and b > s for s, e in taken)

    phi: list[dict] = []
    for fam, pat in PHI_PATTERNS.items():
        for m in pat.finditer(text):
            grp = 1 if (fam == "POSTCODE" and m.groups()) else 0
            a, b = m.start(grp), m.end(grp)
            if not overlaps(a, b):
                phi.append({"type": fam, "start": a, "end": b,
                            "text": m.group(grp)})
                taken.append((a, b))
    return safe, phi


# ---------------------------------------------------------------------------
# Self-checks: every case is a recorded pilot failure or negative.
# ---------------------------------------------------------------------------
_CASES = [
    # (text, expect_family_or_None_on_substring, substring)
    ("KETHLEY RD / SHAWNEE OK 74804-9625", "POSTCODE", "74804-9625"),  # site 8
    ("OKLAHOMA CITY OK 73118-7532", "POSTCODE", "73118-7532"),         # note 11
    ("DEL CITY OK 73115 Phone", "POSTCODE", "73115"),                  # note 10
    ("CATHETER x2 # 1 ARRIVE 0610", None, "0610"),        # SAFE clock, sites 4
    ("06/27/26 0322\t98.2", None, "0322"),                # SAFE clock, site 5
    ("Patient's location: RM08", None, "RM08"),           # SAFE room
    ("bp: 120/80 measured", None, "120/80"),              # SAFE bp
    ("Date of Service: 11/13/2025", "DATE", "11/13/2025"),
    ("seen 06/27/26 at clinic", "DATE", "06/27/26"),
    ("Phone: 405-395-5600", "PHONE_NUMBER", "405-395-5600"),
    ("SSN 449-38-2044 on file", "SSN", "449-38-2044"),
    ("accession 1234567 unchanged", None, "1234567"),     # bare digits: no claim
]


def _self_check() -> int:
    fails = 0
    for text, fam, sub in _CASES:
        safe, phi = find_spans(text)
        hit = next((p for p in phi if sub in p["text"] or p["text"] in sub), None)
        if fam is None:
            ok = hit is None
            note = "unclaimed or SAFE" if ok else f"WRONGLY claimed as {hit['type']}"
        else:
            ok = hit is not None and hit["type"] == fam
            note = f"claimed as {hit['type']}" if hit else "MISSED"
        print(f"[{'PASS' if ok else 'FAIL'}] {text!r:<45} -> {note}")
        fails += 0 if ok else 1
    # Branch 1 grammar spot checks
    assert DATE_MAXIMAL_TOKEN.match("11/13/2025\nCre").end() == 10, "edge trim"
    assert DATE_MAXIMAL_TOKEN.match("43") is None, "bounded fields reject 43"
    print("DATE_MAXIMAL_TOKEN edge checks passed")
    return fails


if __name__ == "__main__":
    sys.exit(_self_check())
