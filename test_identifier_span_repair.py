"""Span-layer tests for identifier span repair (fix/identifier-span-repair, Branch 2).

Three rules in one span pass (_repair_identifier_spans), placed after _split_name_spans and
before _trim_identifier_span_edges:
  R1 over-match split: an identifier span whose edge sits inside an alpha run (letters continue
     into a word) retreats to its digit run; a pure-alpha fragment (site 7: MRN over the "M" of
     "MCHC") is dropped so the label survives.
  R2 under-match completion: a geographic span (STREET_ADDRESS) whose edge falls mid-word extends
     to the word boundary (site 8a: "3214 KETH" -> "3214 KETHLEY"), redacting more, never less.
  R3 sub-run insurance: an identifier/date span that is a proper sub-run of a longer digit run
     expands to the full run (<=6 digits) or drops (>6). Composition refinement proven by the
     stacked assertion: a ZIP+4 tail (run preceded by \d{5}-) is DROPPED, not expanded, because
     expanding surrogates only the +4 as an MRN and strands the 5-digit ZIP prefix (Branch 5
     cannot re-generalize "74804-<letters>"); dropping lets the Branch 5 backstop re-generalize
     the entire ZIP. A leading sub-run (note-2 "7" of "73084") expands cleanly.

Span layer, NO model / NO GPU. Recorded offsets for site 7 re-derived from deid_log.jsonl:
MEDICAL_RECORD_NUMBER [7792:7793]="M", left="\t", right="C" (the "M" of "MCHC").

Run:  python test_identifier_span_repair.py
"""
import io
import re
import sys
from contextlib import redirect_stdout

from deid import deidentify_text

SEED = 42

try:
    from deid import _repair_identifier_spans as _ris
except Exception:  # noqa: BLE001
    _ris = None


def _apply(src, span, etype, occ=0):
    idx = -1
    for _ in range(occ + 1):
        idx = src.find(span, idx + 1)
    assert idx >= 0, f"{span!r} not in {src!r}"
    ents = [{"type": etype, "start": idx, "end": idx + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = _ris([dict(ents[0])], src) if _ris is not None else ents
    return res, buf.getvalue()


def _out(src, span, etype, occ=0):
    idx = -1
    for _ in range(occ + 1):
        idx = src.find(span, idx + 1)
    ents = [{"type": etype, "start": idx, "end": idx + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        o = deidentify_text(src, ents, seed=SEED)
    return o, buf.getvalue()


CASES = []


def case(name, fn, note=""):
    CASES.append((name, fn, note))


# ---- R1: site 7, MRN over the "M" of "MCHC" (recorded offsets) ----
def _site7():
    src = " \tMCHC\t36.4 (H)"
    res, log = _apply(src, "M", "MEDICAL_RECORD_NUMBER")
    o, _ = _out(src, "M", "MEDICAL_RECORD_NUMBER")
    # span dropped; MCHC survives verbatim; no digit run abuts a letter in output
    return (len(res) == 0 and "drop" in log.lower() and "MCHC" in o
            and not re.search(r"\d[A-Za-z]", o),
            f"MRN 'M' dropped, MCHC survives (out={o!r})")
case("site7 MRN 'M' in MCHC dropped", _site7)

# ---- R2: site 8a, STREET_ADDRESS truncating KETHLEY ----
def _site8a():
    src = "addr 3214 KETHLEY RD end"
    res, log = _apply(src, "3214 KETH", "STREET_ADDRESS")
    ok = (len(res) == 1 and res[0]["text"] == "3214 KETHLEY" and "complete" in log.lower())
    return (ok, f"span extended to '3214 KETHLEY' (got {[r['text'] for r in res]})")
case("site8a STREET_ADDRESS completes KETHLEY", _site8a)

# stacked: with Branch 5, the KETHLEY line has no orphan LEY.
def _site8a_stack():
    src = "4250 / SHAWNEE OK 3214 KETHLEY RD"
    o, _ = _out(src, "3214 KETH", "STREET_ADDRESS")
    return ("LEY" not in o.replace("KETHLEY", "") or "KETHLEY" not in src,
            f"no orphan LEY after surrogation (out={o!r})")
case("site8a stacked no LEY orphan", _site8a_stack)

# ---- R3: '25' of '9625' (ZIP+4 tail) -> DROP; Branch 5 re-generalizes the whole ZIP ----
def _zip_tail():
    src = "SHAWNEE OK 74804-9625 end"
    res, log = _apply(src, "25", "MEDICAL_RECORD_NUMBER")  # the '25' tail of 9625 (unique)
    o, _ = _out(src, "25", "MEDICAL_RECORD_NUMBER")
    dropped = len(res) == 0 and "drop" in log.lower()
    clean = "74804" not in o and "9625" not in o and not re.search(r"\d[A-Za-z]{2,}\d", o)
    return (dropped and clean, f"ZIP+4 tail dropped, ZIP re-generalized clean (out={o!r})")
case("R3 '25' of '9625' ZIP+4 tail dropped+clean", _zip_tail)

# ---- R3: '7' of '73084' (leading sub-run) -> EXPAND to full run ----
def _lead_subrun():
    src = "SPENCER OK 73084-9167 x"
    res, log = _apply(src, "7", "MEDICAL_RECORD_NUMBER")  # first '7' is the leading digit of 73084
    ok = (len(res) == 1 and res[0]["text"] == "73084" and "expand" in log.lower())
    o, _ = _out(src, "7", "MEDICAL_RECORD_NUMBER")
    clean = "73084" not in o and "9167" not in o
    return (ok and clean, f"expanded to '73084', stacked clean (got {[r['text'] for r in res]}, out={o!r})")
case("R3 '7' of '73084' expands to full run", _lead_subrun)

# ---- R3: 7+ digit run sub-run -> DROP above cap ----
def _over_cap():
    src = "id 12345678 tail"
    res, log = _apply(src, "78", "MEDICAL_RECORD_NUMBER")  # sub-run of 8-digit run
    return (len(res) == 0 and "drop" in log.lower() and ("6" in log or ">6" in log or "8" in log),
            f"7+ digit run sub-run dropped with log (log={log.strip()!r})")
case("R3 8-digit run sub-run dropped (>6)", _over_cap)

# ---- Negatives: must NOT be touched ----
def _unchanged(src, span, etype, occ=0):
    res, _ = _apply(src, span, etype, occ)
    return (len(res) == 1 and res[0]["text"] == span, f"unchanged (got {[r['text'] for r in res]})")
case("neg alpha-prefix MRN 'M4738753'", lambda: _unchanged("mrn M4738753 filed", "M4738753", "MEDICAL_RECORD_NUMBER"))
case("neg clean MRN adjacent whitespace", lambda: _unchanged("MRN 4471829 done", "4471829", "MEDICAL_RECORD_NUMBER"))
case("neg digit run abutting decimal", lambda: _unchanged("lab 1234.5 units", "1234", "MEDICAL_RECORD_NUMBER"))
case("neg NAME span out of family", lambda: _unchanged("Dr John Smith seen", "John", "FIRST_NAME"))


def main():
    print("=" * 100)
    print("IDENTIFIER SPAN REPAIR - span-layer evidence table")
    print("pass present:", _ris is not None)
    print("=" * 100)
    print(f"{'case':46s} {'status':6s} evidence")
    print("-" * 100)
    failures = 0
    for name, fn, _note in CASES:
        ok, ev = fn()
        if not ok:
            failures += 1
        print(f"{name:46s} [{'PASS' if ok else 'FAIL'}] {ev}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
