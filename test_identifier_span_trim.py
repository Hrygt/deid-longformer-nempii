"""Substitution-layer tests for identifier-span edge trimming (fix/identifier-span-edge-trim).

Guards the MRN-fusion defect: an NER MEDICAL_RECORD_NUMBER span that over-matches its
LEFT edge to include an adjacent unit char ("%") fused the surrogate into neighboring
clinical text and consumed the "%" (production artifact: "EF 60-69MRN12594309 with ...").

These tests exercise the SUBSTITUTION layer (deid.deidentify_text) DIRECTLY: crafted
entity spans, deterministic seed, NO model / NO GPU. Detection is not exercised here
(deid_score.py already covers detection; it never calls deidentify_text).

Run:  python test_identifier_span_trim.py     (prints an evidence table, exits non-zero on failure)
"""
import io
import re
import sys
from contextlib import redirect_stdout

from deid import deidentify_text

SEED = 42  # deterministic surrogate digits

# Optional: show trimmed offsets when the fix is present (import-guarded so the RED run
# on HEAD — where the function does not exist — still fails for the RIGHT reason, i.e.
# fusion in deidentify_text output, not an ImportError at module load).
try:
    from deid import _trim_identifier_span_edges as _trim
except Exception:  # noqa: BLE001
    _trim = None


def _mk(src, span, etype):
    i = src.find(span)
    assert i >= 0, f"span {span!r} not in {src!r}"
    return i, i + len(span), [{"type": etype, "start": i, "end": i + len(span), "text": src[i:i + len(span)]}]


def _run(src, span, etype):
    """Return (out_text, captured_stdout, det_offsets, trimmed_offsets_or_None)."""
    s, e, ents = _mk(src, span, etype)
    trimmed = None
    if _trim is not None:
        te = _trim([dict(ents[0])], src)
        trimmed = (te[0]["start"], te[0]["end"]) if te else "DROPPED"
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = deidentify_text(src, ents, seed=SEED)
    return out, buf.getvalue(), (s, e), trimmed


CASES = []  # (name, src, span, etype, assertion_fn(out, log) -> (ok, note))


def case(name, src, span, etype, check):
    CASES.append((name, src, span, etype, check))


MRN = "MEDICAL_RECORD_NUMBER"
SRC = "EF 60-69% MRN 12594309 with dilated right ventricle"

# 1. THE PRODUCTION ARTIFACT — left over-match incl. unit. RED on HEAD (fusion present).
case("C: '% MRN 12594309' over-match", SRC, "% MRN 12594309", MRN,
     lambda o, l: (o.startswith("EF 60-69% ") and "69MRN" not in o and o[10].isalnum(),
                   "percent preserved + surrogate space-delimited, no fusion"))

# 2. leading-space span -> space preserved, no fusion.
case("D: ' MRN 12594309' leading space", SRC, " MRN 12594309", MRN,
     lambda o, l: (o.startswith("EF 60-69% ") and "69MRN" not in o, "space preserved"))

# 3. tight span -> unchanged behavior (invariant both sides).
case("A: 'MRN 12594309' tight", SRC, "MRN 12594309", MRN,
     lambda o, l: (o.startswith("EF 60-69% ") and o.split()[2].startswith("MRN"), "unchanged, delimited"))

# 4. digits-only span -> unchanged; literal 'MRN ' label stays before surrogate.
case("B: '12594309' digits only", SRC, "12594309", MRN,
     lambda o, l: (o.startswith("EF 60-69% MRN ") and "% MRN " in o, "unchanged"))

# 5. SSN over-match with leading % -> clean, interior hyphens of surrogate intact.
SRC_SSN = "note: % 123-45-6789 on file"
case("SSN '% 123-45-6789' over-match", SRC_SSN, "% 123-45-6789", "SSN",
     lambda o, l: (o.startswith("note: % ") and bool(re.search(r"\b\d{3}-\d{2}-\d{4}\b", o)),
                   "percent kept, hyphenated surrogate"))

# 6. ACCOUNT_NUMBER over-match -> clean.
SRC_ACCT = "ref: % ACCT00099123 end"
case("ACCOUNT '% ACCT00099123' over-match", SRC_ACCT, "% ACCT00099123", "ACCOUNT_NUMBER",
     lambda o, l: (o.startswith("ref: % ") and "% ACCT00099123" not in o, "percent kept, no fusion"))

# 7. alpha-prefix no-space span -> prefix preserved through _generate_mrn (invariant).
SRC_AP = "id MRN12594309 x"
case("alpha-prefix 'MRN12594309' no space", SRC_AP, "MRN12594309", MRN,
     lambda o, l: (o.startswith("id MRN") and o != SRC_AP, "prefix preserved, surrogated"))

# 8. span trims to empty -> entity dropped, no exception, logged, punctuation preserved.
SRC_EMPTY = "value %%% here"
case("degenerate '%%%' -> dropped", SRC_EMPTY, "%%%", MRN,
     lambda o, l: (o == SRC_EMPTY and ("drop" in l.lower() or "degenerate" in l.lower()),
                   "span preserved + drop logged"))

# 9. NAME entity in same text -> completely unaffected by the new function.
SRC_NAME = "Dr. John Smith saw the patient"
case("NAME 'John Smith' unaffected", SRC_NAME, "John Smith", "NAME",
     lambda o, l: ("John Smith" not in o and o.endswith(" saw the patient"), "name still surrogated"))


def main():
    print("=" * 100)
    print("IDENTIFIER SPAN EDGE-TRIM — substitution-layer evidence table (seed=%d)" % SEED)
    print("fix present:", _trim is not None)
    print("=" * 100)
    hdr = f"{'case':40s} {'det span':>10s} {'trim span':>12s}  output / evidence"
    print(hdr); print("-" * 100)
    failures = 0
    for name, src, span, etype, check in CASES:
        out, log, det, trimmed = _run(src, span, etype)
        ok, note = check(out, log)
        if not ok:
            failures += 1
        status = "PASS" if ok else "FAIL"
        ts = str(trimmed) if trimmed is not None else "n/a(HEAD)"
        print(f"{name:40s} {str(det):>10s} {ts:>12s}  [{status}] {out!r}")
        if log.strip():
            print(f"{'':40s} {'':>10s} {'':>12s}  log: {log.strip()}")
        print(f"{'':40s} {'':>10s} {'':>12s}  ^ expect: {note}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
