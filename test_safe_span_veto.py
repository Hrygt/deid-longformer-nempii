"""Detection-filter-layer tests for the SAFE-span veto (fix/safe-span-veto, Branch 6).

Guards two recorded NER false-positive classes that fed length-expanding surrogates:
  - DATE over the trailing 2 digits of an HHMM clock time (pilot sites 4/5, note 3:
    "ARRIVE 0610" -> DATE "10", "0322" -> DATE "22", "2043" -> DATE "43").
  - POSTCODE over a room token (notes 11/12: "RM06"/"RM08" in location context).
A SAFE match claims text that no detector may surrogate; a detected entity FULLY
CONTAINED in a firing SAFE match is vetoed before name resolution and substitution.

These tests exercise the DETECTION FILTER layer (api.filter_safe_spans) DIRECTLY:
hand-built entity sets, NO model / NO GPU (DEID_SKIP_MODEL_LOAD is set before import),
SEED unused (the veto is deterministic and surrogate-free). deid_score.py scores the
extraction path and does NOT call this filter, so it cannot see this change; the real
safety gate is the direct veto-versus-gold test (test_veto_vs_gold.py).

FAIL-SAFE DESIGN NOTE (risk direction is inverted: a veto on real PHI LEAKS).
The clock pattern's value bounds ([01]\\d|2[0-3] then [0-5]\\d) and edge lookarounds
that reject adjacency to [\\d/:.,-] mean it can never claim a slash date component
(07/19/2024), a colon time (06:10), a hyphenated id, a European date (01.06.2028, the
year is period-adjacent), or a decimal-adjacent value (1234.5). It CAN still match a
bare 4-digit year in clock range (1900-1959, 2000-2359) when that year is space-
separated in prose ("February 15, 2024" -> "2024"). The 2026-07-05 year-sweep found
114 such gold DATE/DOB overlaps. Two layers close that class:
  1. Context gate: the clock pattern only FIRES on a tabular line (a tab), an Epic
     tabular-row cell (a line with no letters, a bare value column), or immediately
     after a time cue. Prose date lines have letters and no tab and no cue, so the
     pattern does not fire there. Cue list derived from the 158 pilot matches: {AT}
     only (18 clean "at HHMM" matches); IN/TO/OF/AROUND excluded (precede years, e.g.
     "in 2019"); DING/STE/ALT were substring artifacts of ending/Suite.
  2. Containment rule: only an entity FULLY inside a 4-char clock match is vetoed, so
     a full written date span (longer than 4 chars) is never dropped even on overlap.
The three gold-year forms are explicit must-survive negatives below.

Run:  DEID_SKIP_MODEL_LOAD=1 python test_safe_span_veto.py
"""
import io
import os
import sys
from contextlib import redirect_stdout

os.environ["DEID_SKIP_MODEL_LOAD"] = "1"  # model-free import of api

# Import-guarded fix hook: on HEAD (before implementation) filter_safe_spans is absent,
# so the RED run fails for the RIGHT reason (the FP entity survives, not an ImportError).
try:
    from api import filter_safe_spans as _fss
except Exception:  # noqa: BLE001
    _fss = None


def _run(src, span, etype):
    """Return (vetoed_bool, captured_log, surviving_entities)."""
    i = src.find(span)
    assert i >= 0, f"span {span!r} not in {src!r}"
    ents = [{"type": etype, "start": i, "end": i + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = _fss(src, ents) if _fss is not None else list(ents)
    return (len(out) == 0), buf.getvalue(), out


CASES = []  # (name, src, span, etype, expect_vetoed, note)


def case(name, src, span, etype, expect_vetoed, note):
    CASES.append((name, src, span, etype, expect_vetoed, note))


# ---- RED seeds: the five recorded false positives (expect VETOED) ----
# Site 4: tabular ARRIVE row, DATE over the trailing "10" of clock 0610.
case("site4 DATE '10' in ARRIVE 0610", " \t N/A; CATHETER x2 # 1 ARRIVE 0610", "10", "DATE",
     True, "tabular line -> clock SAFE contains DATE tail")
# Site 5: tabular vitals row, DATE over "22" of clock 0322.
case("site5 DATE '22' in 0322 vitals", "06/27/26 0322\t98.2 F", "22", "DATE",
     True, "tab present -> clock SAFE contains DATE tail")
# Note 3: bare-value column cell, DATE over "43" of clock 2043.
case("note3 DATE '43' in 2043 cell", "06/29/26\n2043\n07/15/20", "43", "DATE",
     True, "Epic cell (no letters) -> clock SAFE contains DATE tail")
# Notes 11/12: POSTCODE over room tokens.
case("note11 POSTCODE 'RM06'", "Patient's location: RM06", "RM06", "POSTCODE",
     True, "room_token SAFE contains the POSTCODE span")
case("note12 POSTCODE 'RM08'", "Prior location: RM08", "RM08", "POSTCODE",
     True, "room_token SAFE contains the POSTCODE span")
# The two note-4 clock fragments, ratified as legitimate members of the fragment class
# (Decision 2). Strictly smaller than the clock SAFE match, so vetoed.
case("note4 DATE '16' in 1607", "\t1607\t12/19/24", "16", "DATE",
     True, "'16' strictly inside clock '1607' -> vetoed")
case("note4 MRN '0' in 0349", "\t0349\t06/30/26", "0", "MEDICAL_RECORD_NUMBER",
     True, "'0' strictly inside clock '0349' -> vetoed")

# ---- EQUAL-SPAN negatives (Decision 1): a whole 4-digit token on a tab line survives ----
# Under the old FULL-containment rule these were vetoed (the HOLD failure); strict containment
# leaves an equal-span clock match to substitution. A year cell and a clock cell, both tabular.
case("neg equal-span year '1957' (tab)", "Tonsillectomy\t \t1957", "1957", "DATE",
     False, "equal-span 4-digit year survives strict clock veto")
case("neg equal-span clock '1200' (tab)", "PROCEDURE ARRIVE\t1200", "1200", "DATE",
     False, "equal-span 4-digit clock survives (left to NER / recognizer TIME claim)")

# ---- Negatives: must SURVIVE the veto ----
case("neg full date 11/13/2025", "Date of Service: 11/13/2025 seen", "11/13/2025", "DATE",
     False, "slash date: no clock match")
case("neg colon time 06:10", "administered at 06:10 dose", "06:10", "DATE",
     False, "colon breaks the 4-digit run: no clock match")
case("neg DOB line", "DOB: 03/15/1965", "03/15/1965", "DATE_OF_BIRTH",
     False, "slash date: no clock match")
case("neg MRN M4738753", "MRN M4738753 filed", "M4738753", "MEDICAL_RECORD_NUMBER",
     False, "no clock/room match contains it")
case("neg decimal-adjacent 1234", "lab value 1234.5 mg", "1234", "DATE",
     False, "trailing period rejected by edge guard: no clock match")
# The three gold-year forms from the 114 year-sweep overlaps (must survive):
case("neg gold year 'February 15, 2024'", "Signed February 15, 2024 by clerk", "2024", "DATE",
     False, "prose line (letters, no tab, no cue): context gate blocks fire")
case("neg gold year '15 March 2024'", "finalized 15 March 2024 today", "2024", "DATE",
     False, "prose line: context gate blocks fire")
case("neg gold year '01.06.2028' (EU)", "Start 01.06.2028 end", "2028", "DATE",
     False, "period-adjacent: edge guard blocks fire")


def main():
    print("=" * 100)
    print("SAFE-SPAN VETO - detection-filter evidence table")
    print("fix present:", _fss is not None)
    print("=" * 100)
    print(f"{'case':40s} {'exp':4s} {'got':4s} status  surviving/evidence")
    print("-" * 100)
    failures = 0
    for name, src, span, etype, expect, note in CASES:
        vetoed, log, out = _run(src, span, etype)
        ok = (vetoed == expect)
        if not ok:
            failures += 1
        exp = "VETO" if expect else "keep"
        got = "VETO" if vetoed else "keep"
        print(f"{name:40s} {exp:4s} {got:4s} [{'PASS' if ok else 'FAIL'}]  {note}")
        if log.strip():
            print(f"{'':50s} log: {log.strip()}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
