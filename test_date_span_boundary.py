"""Span-layer tests for the DATE span boundary rule (fix/date-span-boundary, Branch 1).

DATE-family spans have no boundary guard at the span layer. Two rules, one new span pass
(_date_span_boundary), placed after _merge_adjacent_date_entities and before _split_name_spans:
  1. Maximal-token edge trim: a DATE-family span that extends beyond a maximal complete date
     token is trimmed to that token (start and end). A multi-cell merged span (two date tokens,
     e.g. "2019\t10/20/24") passes through untrimmed; a legitimate written / ISO / bare-year /
     MM-DD / folded date+time span is one token and is never split.
  2. No-digit drop: a DATE-family span with no digits ("Time", "QDAY", "TBD", "Saturday") is
     dropped with a logged reason, so the header word survives verbatim instead of being
     mangled to "xxxx" by the substitution-layer disciplined fallback.

Branch 3's embedded mirroring already masks the visible \nCre fusion downstream (Step 0), so
these tests assert the SPAN-LAYER CONTRACT (the trimmed/dropped entity offsets), not the old
output symptoms. Substitution layer, NO model / NO GPU, SEED unused for the span pass.

Run:  python test_date_span_boundary.py
"""
import io
import re
import sys
from contextlib import redirect_stdout

from deid import deidentify_text

SEED = 42

try:
    from deid import _date_span_boundary as _dsb
except Exception:  # noqa: BLE001
    _dsb = None


def _apply(src, span, etype):
    """Run the span pass on one crafted entity; return (result_entities, log)."""
    i = src.find(span)
    assert i >= 0, f"{span!r} not in {src!r}"
    ents = [{"type": etype, "start": i, "end": i + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = _dsb([dict(ents[0])], src) if _dsb is not None else ents
    return res, buf.getvalue()


def _out(src, span, etype):
    i = src.find(span)
    ents = [{"type": etype, "start": i, "end": i + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        o = deidentify_text(src, ents, seed=SEED)
    return o, buf.getvalue()


CASES = []


def case(name, fn, note=""):
    CASES.append((name, fn, note))


def trimmed_to(src, span, etype, token):
    res, _ = _apply(src, span, etype)
    return (len(res) == 1 and res[0]["text"] == token and "\n" not in res[0]["text"],
            f"span trimmed to {token!r} (got {[r['text'] for r in res]})")


def dropped(src, span, etype):
    res, log = _apply(src, span, etype)
    return (len(res) == 0 and "drop" in log.lower(),
            f"span dropped + logged (got {[r['text'] for r in res]}, log={log.strip()!r})")


def unchanged(src, span, etype):
    res, _ = _apply(src, span, etype)
    return (len(res) == 1 and res[0]["text"] == span,
            f"span passes through unchanged (got {[r['text'] for r in res]})")


# ---- Rule 1: header over-reach trim (three constructed header sites: notes 8/10/19 form) ----
case("site1 trim '11/13/2025\\nCre'",
     lambda: trimmed_to("Date of Service: 11/13/2025\nCreation Time: x", "11/13/2025\nCre", "DATE", "11/13/2025"))
case("site2 trim '07/28/2025\\nCre'",
     lambda: trimmed_to("Date of Service: 07/28/2025\nCreation Time: x", "07/28/2025\nCre", "DATE", "07/28/2025"))
case("site3a trim compound '6/27/2026 12:54 AM\\nCre'",
     lambda: trimmed_to("Service: 6/27/2026 12:54 AM\nCreation Time: x", "6/27/2026 12:54 AM\nCre",
                        "DATE_TIME", "6/27/2026 12:54 AM"))
# header text byte-identical outside the date: the \nCreation label survives in the output.
case("header label survives in output",
     lambda: (("\nCreation Time:" in _out("Date of Service: 11/13/2025\nCreation Time: x",
                                          "11/13/2025\nCre", "DATE")[0]),
              "newline + Creation label preserved in output"))

# ---- Rule 2: no-digit drop (site 3b "Time" + recorded QDAY/Saturday/TBD) ----
case("site3b drop 'Time' (DATE_TIME)", lambda: dropped("Creation Time: 04/15/2026", "Time", "DATE_TIME"))
case("drop 'QDAY'", lambda: dropped("give meds QDAY per plan", "QDAY", "DATE"))
case("drop 'Saturday'", lambda: dropped("seen on Saturday morning", "Saturday", "DATE"))
case("drop 'TBD'", lambda: dropped("follow-up TBD next", "TBD", "DATE"))
# word survives verbatim in output (not "xxxx").
case("'Time' survives verbatim in output",
     lambda: (("Creation Time:" in _out("Creation Time: 04/15/2026", "Time", "DATE_TIME")[0]
               and "xxxx" not in _out("Creation Time: 04/15/2026", "Time", "DATE_TIME")[0]),
              "header word 'Time' preserved, not mangled to xxxx"))

# ---- Negatives: must NOT be trimmed or dropped ----
case("neg written 'November 5, 2024'", lambda: unchanged("on November 5, 2024 seen", "November 5, 2024", "DATE"))
case("neg ISO '2024-03-15'", lambda: unchanged("date 2024-03-15 filed", "2024-03-15", "DATE"))
case("neg bare year '2019'", lambda: unchanged("status post CABG 2019 done", "2019", "DATE"))
case("neg MM/DD short '12/13'", lambda: unchanged("seen 12/13 clinic", "12/13", "DATE"))
case("neg compound '08/14/2025 5:54'", lambda: unchanged("at 08/14/2025 5:54 noted", "08/14/2025 5:54", "DATE_TIME"))
case("neg multi-cell '2019\\t10/20/24'", lambda: unchanged("hx\t2019\t10/20/24 row", "2019\t10/20/24", "DATE"))
case("neg full MM/DD/YYYY '6/30/2026'", lambda: unchanged("Date: 6/30/2026 seen", "6/30/2026", "DATE"))


def main():
    print("=" * 100)
    print("DATE SPAN BOUNDARY - span-layer evidence table")
    print("pass present:", _dsb is not None)
    print("=" * 100)
    print(f"{'case':44s} {'status':6s} evidence")
    print("-" * 100)
    failures = 0
    for name, fn, _note in CASES:
        ok, ev = fn()
        if not ok:
            failures += 1
        print(f"{name:44s} [{'PASS' if ok else 'FAIL'}] {ev}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
