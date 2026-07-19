"""Span-layer tests for *_NAME fragment repair (fix/name-span-fragment-repair,
grader report 057dab78c412 half 2/2 — root cause).

Ratified design (all four questions ruled): a *_NAME span with a mid-token boundary on
EITHER side (adjacent char is an ASCII letter) is EXPANDED to the full word so the whole
token is surrogated and the map records FULL word -> surrogate — EXCEPT when the full
surrounding word is in MEDICAL_WHITELIST_LOWER, in which case the span is DROPPED so the
drug survives (the drop-only guard's drug-FP behavior, preserved). Two spans expanding to
the same word merge to ONE span / ONE surrogate. JohnSmith: whole-token SINGLE-surrogate
replacement supersedes the B.1 two-span shape (B.1 was a design-time experiment recorded
in FOLLOWUP_deid_ner_span_guard.md, not a committed probe); the ZERO-LEAK property is
asserted unweakened here.

Span layer, NO model / NO GPU (DEID_SKIP_MODEL_LOAD). Modeled on
test_identifier_span_repair.py.

Run:  python test_name_span_fragment_repair.py
"""
import io
import os
import re
import sys
from contextlib import redirect_stdout

os.environ.setdefault("DEID_SKIP_MODEL_LOAD", "1")

from deid import deidentify_text  # noqa: E402

SEED = 42

try:
    from api import _repair_name_span_fragments as _guard  # the fix
    GUARD = "repair"
except Exception:  # noqa: BLE001
    from api import _drop_name_subword_fragments as _guard  # HEAD (drop-only)
    GUARD = "drop-only (HEAD)"


def _ents(src, spans):
    """spans: list of (fragment, type, occ). Builds NER-like entities by offset."""
    out = []
    for frag, etype, occ in spans:
        idx = -1
        for _ in range(occ + 1):
            idx = src.find(frag, idx + 1)
        assert idx >= 0, f"{frag!r} not in {src!r}"
        out.append({"type": etype, "start": idx, "end": idx + len(frag), "text": frag})
    return out


def _apply(src, spans):
    """Run the guard alone; returns (result_entities, captured_log)."""
    ents = _ents(src, spans)
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = _guard(src, [dict(e) for e in ents])
    return res, buf.getvalue()


def _out(src, spans):
    """Guard -> deidentify_text (the extract->substitute path); returns
    (deid_text, surrogate_map, guarded_entities)."""
    res, _ = _apply(src, spans)
    buf = io.StringIO()
    with redirect_stdout(buf):
        o, m = deidentify_text(src, [dict(e) for e in res], seed=SEED, return_map=True)
    return o, m, res


def _originals(m):
    return sorted({p["original"].lower() for p in (m or {}).get("pairs", [])})


CASES = []


def case(name, fn):
    CASES.append((name, fn))


# ---- 1. TRAILING-FRAGMENT WITNESS (the 057dab78c412 shape): NER covers only the ----
# ---- trailing part of a consultant name -> whole word must be surrogated        ----
def _trailing():
    src = "Consulted Dr. Kowalczyk regarding GI bleed."
    o, m, res = _out(src, [("alczyk", "LAST_NAME", 0)])
    one = len(res) == 1 and res[0]["text"] == "Kowalczyk"
    no_frag = "Kowalczyk" not in o and "Kow" not in o and "alczyk" not in o
    full_pair = "kowalczyk" in _originals(m)
    return (one and no_frag and full_pair,
            f"spans={[r['text'] for r in res]} map={_originals(m)} out={o!r}")
case("1 trailing fragment 'alczyk' -> whole 'Kowalczyk' surrogated", _trailing)


# ---- 2. PREFIX-FRAGMENT NAME: same expectation via clause 1 (HEAD *drops* this ----
# ---- span, leaking the whole real name verbatim)                               ----
def _prefix():
    src = "Consulted Dr. Kowalczyk regarding GI bleed."
    o, m, res = _out(src, [("Kowal", "LAST_NAME", 0)])
    one = len(res) == 1 and res[0]["text"] == "Kowalczyk"
    no_leak = "Kowalczyk" not in o and "Kowal" not in o
    full_pair = "kowalczyk" in _originals(m)
    return (one and no_leak and full_pair,
            f"spans={[r['text'] for r in res]} map={_originals(m)} out={o!r}")
case("2 prefix fragment 'Kowal' -> whole 'Kowalczyk' surrogated", _prefix)


# ---- 3. WHITELIST FRAGMENT: cefepime 'ce'+'ime' witness + clozapine 'cloz' ----
# ---- sibling -> spans DROPPED, drugs survive intact                        ----
def _cefepime():
    src = "Started cefepime 2g IV q8h for pseudomonal coverage."
    o, m, res = _out(src, [("ce", "FIRST_NAME", 0), ("ime", "LAST_NAME", 0)])
    dropped = len(res) == 0
    intact = "cefepime 2g IV q8h" in o
    return (dropped and intact, f"spans={[r['text'] for r in res]} out={o!r}")
case("3a cefepime 'ce'+'ime' fragments dropped, drug intact", _cefepime)


def _clozapine():
    src = "Continue clozapine 100mg qhs, monitor ANC."
    o, m, res = _out(src, [("cloz", "FIRST_NAME", 0)])
    dropped = len(res) == 0
    intact = "clozapine 100mg" in o
    return (dropped and intact, f"spans={[r['text'] for r in res]} out={o!r}")
case("3b clozapine 'cloz' fragment dropped, drug intact", _clozapine)


# ---- 4. JohnSmith run-together: ONE surrogate, zero leak (ratified single- ----
# ---- surrogate shape; supersedes the B.1 two-span shape)                   ----
def _johnsmith():
    src = "Seen by JohnSmith at bedside."
    o, m, res = _out(src, [("John", "FIRST_NAME", 0), ("Smith", "LAST_NAME", 0)])
    one = len(res) == 1 and res[0]["text"] == "JohnSmith"
    zero_leak = not re.search(r"John|Smith", o)
    one_pair = _originals(m) == ["johnsmith"]
    return (one and zero_leak and one_pair,
            f"spans={[r['text'] for r in res]} map={_originals(m)} out={o!r}")
case("4 JohnSmith -> one merged span, one surrogate, zero leak", _johnsmith)


# ---- 5. Same-word double-span merge: two fragments of ONE word -> one span, ----
# ---- one surrogate, one map pair                                            ----
def _double():
    src = "Ms. Kowalczyk was consulted."
    o, m, res = _out(src, [("Kow", "FIRST_NAME", 0), ("alczyk", "LAST_NAME", 0)])
    one = len(res) == 1 and res[0]["text"] == "Kowalczyk"
    no_frag = "Kow" not in o and "alczyk" not in o
    one_pair = _originals(m) == ["kowalczyk"]
    return (one and no_frag and one_pair,
            f"spans={[r['text'] for r in res]} map={_originals(m)} out={o!r}")
case("5 double fragments of one word merge to one span/surrogate", _double)


# ---- 6. Real-name-still-redacts regression battery (clean boundaries: the ----
# ---- guard must not touch them; redaction still happens downstream)       ----
def _reg_clean():
    src = "Dr John Smith saw the patient."
    o, m, res = _out(src, [("John", "FIRST_NAME", 0), ("Smith", "LAST_NAME", 0)])
    untouched = [r["text"] for r in res] == ["John", "Smith"]
    redacted = not re.search(r"\bJohn\b|\bSmith\b", o)
    return (untouched and redacted, f"spans={[r['text'] for r in res]} out={o!r}")
case("6a clean 'John Smith' untouched by guard, still redacted", _reg_clean)


def _reg_initial():
    src = "Patient of A. Grant, MD."
    res, _ = _apply(src, [("A", "FIRST_NAME", 0), ("Grant", "LAST_NAME", 0)])
    ok = [r["text"] for r in res] == ["A", "Grant"]
    return (ok, f"spans={[r['text'] for r in res]}")
case("6b initial 'A. Grant' spans untouched", _reg_initial)


def _reg_hyphen():
    src = "Followed by Dr. Smith-Jones in clinic."
    o, m, res = _out(src, [("Smith", "LAST_NAME", 0)])
    untouched = [r["text"] for r in res] == ["Smith"]
    redacted = "Smith" not in o
    return (untouched and redacted, f"spans={[r['text'] for r in res]} out={o!r}")
case("6c hyphenated 'Smith'-of-'Smith-Jones' untouched (hyphen boundary)", _reg_hyphen)


def _reg_short():
    src = "Ed reviewed the imaging."
    o, m, res = _out(src, [("Ed", "FIRST_NAME", 0)])
    untouched = [r["text"] for r in res] == ["Ed"]
    redacted = not re.search(r"\bEd\b", o)
    return (untouched and redacted, f"spans={[r['text'] for r in res]} out={o!r}")
case("6d short name 'Ed' untouched, still redacted", _reg_short)


def main():
    print("=" * 100)
    print("NAME-SPAN FRAGMENT REPAIR - span-layer evidence table")
    print("guard under test:", GUARD)
    print("=" * 100)
    print(f"{'case':60s} {'status':6s} evidence")
    print("-" * 100)
    failures = 0
    for name, fn in CASES:
        try:
            ok, ev = fn()
        except AssertionError as exc:
            ok, ev = False, f"setup assert: {exc}"
        if not ok:
            failures += 1
        print(f"{name:60s} [{'PASS' if ok else 'FAIL'}] {ev}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
