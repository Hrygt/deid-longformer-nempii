"""Text-layer tests for the ZIP residual backstop (fix/zip-residual-backstop).

Guards a PHI leak class found in the 21-note ICU pilot: ZIP codes that the NER left
undetected or only partially covered survived verbatim into de-identified output
(9 of 12 input ZIPs leaked full or partial ZIP material; ZIP+4 went 0 for 7). The
backstop is a post-substitution, TEXT-level sweep in deid.deidentify_text (spirit of
the patient-token backstop at deid.py:945-956) that re-generalizes surviving ZIP
patterns through the existing _generalize_zip so surrogate policy stays in one place.

These tests exercise the SUBSTITUTION/TEXT layer (deid.deidentify_text) DIRECTLY:
crafted text, optional crafted entity spans, deterministic seed, NO model / NO GPU.
Detection is not exercised here (deid_score.py / benchmark_i2b2_2014.py cover detection;
neither calls deidentify_text, and this pass lives below detection).

Run:  python test_zip_backstop.py     (prints an evidence table, exits non-zero on failure)
"""
import io
import re
import sys
from contextlib import redirect_stdout

from deid import deidentify_text

SEED = 42  # deterministic surrogate ZIP digits

# Optional: reflect whether the fix is present (import-guarded so the RED run on HEAD
# where the function does not exist still fails for the RIGHT reason, i.e. a ZIP leak
# in deidentify_text output, not an ImportError at module load).
try:
    from deid import _zip_residual_backstop as _zbk
except Exception:  # noqa: BLE001
    _zbk = None


def _run(src, span, etype):
    """Return (out_text, captured_stdout). span=None means no entity (pure passthrough)."""
    ents = []
    if span is not None:
        i = src.find(span)
        assert i >= 0, f"span {span!r} not in {src!r}"
        ents = [{"type": etype, "start": i, "end": i + len(span), "text": src[i:i + len(span)]}]
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = deidentify_text(src, ents, seed=SEED)
    return out, buf.getvalue()


CASES = []  # (name, src, span, etype, check(out, log) -> (ok, note))


def case(name, src, span, etype, check):
    CASES.append((name, src, span, etype, check))


MRN = "MEDICAL_RECORD_NUMBER"

# ---- Leak shape A: full ZIP+4 surviving verbatim (undetected). Notes 1, 8, 12, 19. ----
case("N1  full ZIP+4 'OK 74804-9638'", "Home: 14 Kethley / SHAWNEE OK 74804-9638", None, None,
     lambda o, l: ("74804-9638" not in o and bool(re.search(r"\bOK \d{5}-\d{4}\b", o)),
                   "ZIP+4 re-generalized, state kept, original gone"))
case("N8  full ZIP+4 'OK 74868-1917'", "Addr: 900 Ziegler Blvd / SEMINOLE OK 74868-1917", None, None,
     lambda o, l: ("74868-1917" not in o and bool(re.search(r"\bOK \d{5}-\d{4}\b", o)), "gone"))
case("N12 full ZIP+4 'OK 73064-4308'", "Addr: 12 Rd Ste 200 / MUSTANG OK 73064-4308", None, None,
     lambda o, l: ("73064-4308" not in o and bool(re.search(r"\bOK \d{5}-\d{4}\b", o)), "gone"))
case("N19 full ZIP+4 'OK 73096-9746'", "Addr: 1 Legacy / WEATHERFORD OK 73096-9746", None, None,
     lambda o, l: ("73096-9746" not in o and bool(re.search(r"\bOK \d{5}-\d{4}\b", o)), "gone"))

# ---- Leak shape B: state-adjacent bare 5-digit (undetected). Notes 10, 17. ----
case("N10 state+ZIP5 'OK 73115'", "Addr: Suite 300 / DEL CITY OK 73115", None, None,
     lambda o, l: ("73115" not in o and bool(re.search(r"\bOK \d{5}\b", o)),
                   "bare ZIP re-generalized, state kept"))
case("N17 state+ZIP5 'OK 74804'", "Addr: 1 Harrison St / Shawnee OK 74804", None, None,
     lambda o, l: ("74804" not in o and bool(re.search(r"\bOK \d{5}\b", o)), "gone"))

# ---- Leak shape C: truncated/fused ZIP. Note 13 (site 8): MRN over the +4 tail "25". ----
# Input "74804-9625" with MRN over the trailing "25". At Branch 5 this surrogated to
# "74804-96MRN..", a fusion left for a later branch. Branch 2's sub-run insurance now DROPS the
# ZIP+4-tail MRN (a mis-tag), so the ZIP backstop re-generalizes the whole ZIP cleanly with no
# fused MRN. Assertion updated to the post-Branch-2 stacked behavior: clean state+ZIP+4, no leak.
case("N13 truncated 'OK 74804-96'+MRN", "Addr: Kethley Rd / SHAWNEE OK 74804-9625", "25", MRN,
     lambda o, l: ("74804" not in o and "9625" not in o and bool(re.search(r"\bOK \d{5}-\d{4}\b", o))
                   and "MRN" not in o.split("SHAWNEE")[-1],
                   "ZIP re-generalized clean, no MRN fusion (Branch 2 dropped the ZIP+4-tail MRN)"))

# ---- Leak shape D: prefix-fused ZIP body behind letters. Note 2: MRN over the LEADING digit. ----
# Input "73084-9167" with MRN over the leading "7" surrogates to "MRN<d>3084-9167", the
# recorded "MRN63084-9167". Backstop must eat the ZIP-shaped tail, leave the MRN letters.
case("N2  prefix-fused 'MRN..3084-9167'", "Addr: 1 Shepherd / SPENCER OK 73084-9167", "7", MRN,
     lambda o, l: ("9167" not in o and "3084-9167" not in o and "MRN" in o,
                   "ZIP-shaped tail gone, MRN letters intact"))

# ---- NEGATIVE: room-number tokens in location context (recorded POSTCODE false positives). ----
case("NEG room 'RM06'/'RM08'", "Patient's location: RM06\nPrior location: RM08", None, None,
     lambda o, l: ("RM06" in o and "RM08" in o, "room numbers untouched"))

# ---- NEGATIVE: intended POSTCODE surrogate already emitted (idempotence). ----
# Harmless RE-GENERALIZATION (not a strict no-op): a fake state-adjacent ZIP is itself
# ZIP-shaped, so the backstop replaces it with another valid fake ZIP. Never corruption.
case("NEG idempotence 'OK 86007'", "City / OKLAHOMA CITY OK 86007", None, None,
     lambda o, l: (bool(re.search(r"\bOK \d{5}\b", o)),
                   "still a valid state+ZIP5 (harmless re-generalization)"))

# ---- NEGATIVE: vitals row, 4-digit clock times + decimal temp. ----
case("NEG vitals '0308  98.2'", "07/19/25 0308\t98.2 F (36.8 C)", None, None,
     lambda o, l: ("0308" in o and "98.2" in o and "07/19/25" in o, "vitals untouched"))

# ---- NEGATIVE: phone number (3-3-4 hyphenation, not 5-4). ----
case("NEG phone '405-395-5600'", "Phone: 405-395-5600", None, None,
     lambda o, l: ("405-395-5600" in o, "phone untouched"))

# ---- NEGATIVE: bare 5-digit with NO state context (would destroy acct/lab IDs). ----
case("NEG bare5 no-state '73115'", "Lab accession 73115 recorded", None, None,
     lambda o, l: ("73115" in o, "unconditioned 5-digit survives"))

# ---- NEGATIVE: a date (no hyphen-ZIP shape); proven anyway. ----
case("NEG date '07/19/2025'", "Date of Service: 07/19/2025", None, None,
     lambda o, l: ("07/19/2025" in o, "date untouched"))


# ---- Condition 2: surrogate-map consistency guard (return_map=True path). ----
# The README (return_map) promises the map is "Consumed by the CPT grader's re-identification
# overlay" to "restore the real entities in its output" -- exact-match dependence on output text.
# Re-generalizing an already-recorded POSTCODE surrogate would strand its map value. The guard
# skips re-generalization of a recorded surrogate; RAW undetected ZIPs (absent from the map) still
# die. RED before the guard (map value re-generalized out of the text), GREEN after.
def check_map_guard():
    src = "Clinic / OKLAHOMA CITY OK 73104 -- ref addr SHAWNEE OK 74804-9638"
    i = src.find("73104")
    ents = [{"type": "POSTCODE", "start": i, "end": i + 5, "text": "73104"}]
    out, smap = deidentify_text(src, ents, seed=SEED, return_map=True)
    pc = [p for p in smap.get("pairs", []) if p.get("type") == "POSTCODE"]
    located = bool(pc) and all(p["surrogate"] in out for p in pc)   # map value still in text
    raw_dead = "74804-9638" not in out                             # raw undetected ZIP neutralized
    ok = located and raw_dead
    note = ("POSTCODE map value(s) %r located_in_output=%s AND raw ZIP dead=%s"
            % ([p["surrogate"] for p in pc], located, raw_dead))
    return ok, note, out


def main():
    print("=" * 100)
    print("ZIP RESIDUAL BACKSTOP - text-layer evidence table (seed=%d)" % SEED)
    print("fix present:", _zbk is not None)
    print("=" * 100)
    print(f"{'case':36s} {'status':6s}  output / evidence")
    print("-" * 100)
    failures = 0
    for name, src, span, etype, check in CASES:
        out, log = _run(src, span, etype)
        ok, note = check(out, log)
        if not ok:
            failures += 1
        status = "PASS" if ok else "FAIL"
        print(f"{name:36s} [{status}] {out!r}")
        print(f"{'':36s}        ^ expect: {note}")
        if log.strip():
            print(f"{'':36s}        log: {log.strip()}")
    # Condition 2 guard case (needs the return_map=True output, handled separately).
    g_ok, g_note, g_out = check_map_guard()
    if not g_ok:
        failures += 1
    print(f"{'GUARD map-consistency (return_map)':36s} [{'PASS' if g_ok else 'FAIL'}] {g_out!r}")
    print(f"{'':36s}        ^ expect: {g_note}")
    total = len(CASES) + 1
    print("-" * 100)
    print(f"{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
