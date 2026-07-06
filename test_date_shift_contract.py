"""Substitution-layer tests for the _shift_date output contract (fix/date-shift-contract).

_shift_date (deid.py) violated two invariants: it MANUFACTURED components absent from the
input (bare year 1957 -> full 05/28/1957; fragment "43" -> full 05/28/2043, which fuses to
the recorded 2003/02/2043; unparseable "Time" -> a hash-derived fake date), and it could DROP
the time component of a TIME-fold compound. The contract (adapted from Philter V1.0, JAMIA
Open 2023): components-mirroring (emit only components present in the input, in the input's
format family), length within 2 characters of the input span, round-trip validity for any
month/day output, and a no-structure fallback that never manufactures a date.

These tests exercise the SUBSTITUTION layer DIRECTLY (deid.ClinicalDeidentifier._shift_date
and deidentify_text): crafted inputs, deterministic SEED, NO model / NO GPU. This branch is
below detection; deid_score.py cannot see it (extraction-path confirmation only).

Run:  python test_date_shift_contract.py
"""
import io
import re
import sys
from contextlib import redirect_stdout

from deid import ClinicalDeidentifier, deidentify_text

SEED = 42

try:
    from deid import _DATE_CONTRACT as _contract
except Exception:  # noqa: BLE001
    _contract = None


def _sd(inp):
    d = ClinicalDeidentifier(seed=SEED)
    d.reset_cache()
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = d._shift_date(inp)
    return out


def _dt(src, span, etype):
    i = src.find(span)
    assert i >= 0, f"{span!r} not in {src!r}"
    ents = [{"type": etype, "start": i, "end": i + len(span), "text": span}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = deidentify_text(src, ents, seed=SEED)
    return out


def is_real_date(s):
    """True iff every numeric M/D/Y-shaped token in s is a real calendar date."""
    from datetime import datetime
    ok = True
    for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s):
        mo, da, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(3)) == 2:
            yr += 2000
        try:
            datetime(yr, mo, da)
        except ValueError:
            ok = False
    return ok


def within2(inp, out):
    return abs(len(out) - len(inp)) <= 2


CASES = []


def case(name, fn, check, note=""):
    CASES.append((name, fn, check, note))


# ---- Bare years: year-shaped output, length within 2 ----
for _y in ("1957", "2007", "2006", "2010", "2009", "2017"):
    case(f"bare year {_y}", lambda y=_y: _sd(y),
         lambda o, y=_y: (bool(re.fullmatch(r"\d{4}", o)) and within2(y, o),
                          f"year-shaped, len within 2 (got {o!r})"))

# ---- Fragments fed as DATE spans with adjacent digits: no fusion, length discipline ----
case("fragment '10' in 'ARRIVE 0610'", lambda: _dt("ARRIVE 0610", "10", "DATE"),
     lambda o: (o.startswith("ARRIVE 06") and "/" not in o.split("ARRIVE")[1],
                f"no manufactured date fused into 0610 (got {o!r})"), "")
case("fragment '22' in '0322 val'", lambda: _dt("0322 val", "22", "DATE"),
     lambda o: ("/" not in o and o.endswith(" val"),
                f"no full date fused (got {o!r})"), "")
case("fragment '10' shift length", lambda: _sd("10"),
     lambda o: (within2("10", o) and "/" not in o, f"len within 2, not a full date (got {o!r})"), "")
case("fragment '22' shift length", lambda: _sd("22"),
     lambda o: (within2("22", o) and "/" not in o, f"len within 2, not a full date (got {o!r})"), "")

# ---- Site 6: fragment '43' must not manufacture 2003/02/2043 ----
case("site6 '43' shift round-trips", lambda: _sd("43"),
     lambda o: (within2("43", o) and "/" not in o, f"2-char, not a full date (got {o!r})"), "")
case("site6 fusion '2043' round-trips", lambda: _dt("2043", "43", "DATE"),
     lambda o: (is_real_date(o) and "2003/02/2043" not in o,
                f"no malformed date (got {o!r})"), "")

# ---- 'Time' no-structure: no manufactured date ----
case("no-structure 'Time'", lambda: _sd("Time"),
     lambda o: (not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", o) and within2("Time", o),
                f"no manufactured date (got {o!r})"), "")

# ---- Full formats: components mirrored, value shifted ----
case("full MM/DD/YYYY", lambda: _sd("11/13/2025"),
     lambda o: (bool(re.fullmatch(r"\d{2}/\d{2}/\d{4}", o)) and o != "11/13/2025" and is_real_date(o),
                f"MM/DD/YYYY mirrored+shifted (got {o!r})"), "")
case("MM/DD/YY", lambda: _sd("11/13/25"),
     lambda o: (bool(re.fullmatch(r"\d{2}/\d{2}/\d{2}", o)) and o != "11/13/25",
                f"MM/DD/YY mirrored (got {o!r})"), "")
case("written 'November 5, 2024'", lambda: _sd("November 5, 2024"),
     lambda o: (bool(re.search(r"[A-Za-z]", o)) and "2024" in o and o != "November 5, 2024",
                f"written form mirrored (got {o!r})"), "")
def _iso_ok(o):
    from datetime import datetime
    try:
        datetime.strptime(o, "%Y-%m-%d")
        return True
    except ValueError:
        return False


case("ISO 2024-03-15", lambda: _sd("2024-03-15"),
     lambda o: (bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", o)) and o != "2024-03-15" and _iso_ok(o),
                f"ISO mirrored, valid calendar date (got {o!r})"), "")

# ---- Date+time compound (TIME-fold): time component mirrored, not dropped ----
case("compound '08/14/2025 5:54'", lambda: _sd("08/14/2025 5:54"),
     lambda o: ("5:54" in o and bool(re.search(r"\d{2}/\d{2}/\d{4}", o)) and o != "08/14/2025 5:54",
                f"date shifted, time '5:54' mirrored (got {o!r})"), "")

# ---- DOB-context date still works (age-preserving path) ----
case("DOB line still works", lambda: _dt("DOB: 03/15/1965", "03/15/1965", "DATE"),
     lambda o: ("DOB:" in o and "03/15/1965" not in o and bool(re.search(r"\d{2}/\d{2}/\d{4}", o)),
                f"DOB retyped + surrogated (got {o!r})"), "")


def main():
    print("=" * 100)
    print("DATE-SHIFT CONTRACT - substitution-layer evidence table (seed=%d)" % SEED)
    print("contract present:", _contract is not None)
    print("=" * 100)
    print(f"{'case':38s} {'status':6s} output / evidence")
    print("-" * 100)
    failures = 0
    for name, fn, check, _note in CASES:
        out = fn()
        res = check(out)
        ok, ev = res if isinstance(res, tuple) else (res, "")
        if not ok:
            failures += 1
        print(f"{name:38s} [{'PASS' if ok else 'FAIL'}] {ev}")
    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
