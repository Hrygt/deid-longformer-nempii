"""MANDATORY pilot-acceptance gate for the SAFE-span veto (Branch 6, Decision 4).

Why this gate exists: the synthetic Nemotron gold gate (test_veto_vs_gold.py) is
structurally blind to Epic tabular content. It has no surgical-history rows, so it did
not catch the full-containment rule dropping real procedure-history years; the Branch 6
HOLD did. This gate reads the recorded 21-note ICU pilot entity sets (from the grader's
deid_log.jsonl, read-only) and asserts the exact Decision 2 acceptance bar on real Epic
note structure.

Decision 2 bar (all asserted, HOLD on any miss):
  - Exactly 7 vetoes across the 21 notes: the three fragment sites (10 in 0610, 22 in
    0322, 43 in 2043), RM06, RM08, and the two note-4 clock fragments (16 in 1607, 0 in
    0349).
  - Zero equal-span CLOCK vetoes anywhere (strict containment; room equal-span is allowed
    by design and is not a clock veto).
  - The six surgical-history years from the HOLD table (1957, 2007, 2006, 2010, 2009,
    2017) are NOT vetoed and still reach substitution as DATE entities.

Read-only in the grader repo. Model-free (DEID_SKIP_MODEL_LOAD).
Run:  DEID_SKIP_MODEL_LOAD=1 python test_veto_pilot_audit.py
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

os.environ["DEID_SKIP_MODEL_LOAD"] = "1"
import api  # noqa: E402  (filter_safe_spans + SAFE internals; no model loaded)

GRADER = os.environ.get(
    "CC_GRADER_ROOT", r"C:\MyScripts\MetroProjects\CRITICAL_CARE_GRADER")
LOG = os.path.join(GRADER, "deid", "deid_log.jsonl")
MASKED = os.path.join(GRADER, "masked")

# Expected veto set (sid, start, end, text): the ratified 7 fragment-class members.
EXPECTED_VETOES = {
    ("9929100008", 3243, 3245, "10"),
    ("9929100020", 4739, 4741, "22"),
    ("9929100003", 6330, 6332, "43"),
    ("9929100011", 289, 293, "RM08"),
    ("9929100012", 295, 299, "RM06"),
    ("9929100004", 3540, 3542, "16"),
    ("9929100004", 4107, 4108, "0"),
}
# History years that MUST survive (sid, start, end, text).
HISTORY_YEARS = {
    ("9929100001", 2276, 2280, "1957"),
    ("9929100010", 5008, 5012, "2007"),
    ("9929100013", 2406, 2410, "2006"),
    ("9929100013", 2368, 2372, "2010"),
    ("9929100017", 2682, 2686, "2009"),
    ("9929100017", 3342, 3346, "2017"),
}


def load_rows():
    rows = {}
    for l in open(LOG, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            rows[r["study_id"]] = r
    return rows


def firing_safe(text):
    spans = [(m.start(), m.end(), "room_token") for m in api._SAFE_ROOM.finditer(text)]
    for m in api._SAFE_CLOCK.finditer(text):
        line, col = api._line_at(text, m.start())
        if api._clock_context_ok(line, col):
            spans.append((m.start(), m.end(), "clock_hhmm"))
    return spans


def main():
    rows = load_rows()
    vetoes = []           # (sid, s, e, text)
    equal_clock_vetoes = []
    kept_years = set()
    for sid in sorted(rows):
        masked = open(os.path.join(MASKED, sid + ".txt"), encoding="utf-8").read()
        ents = [{"type": t, "start": s, "end": e, "text": masked[s:e]}
                for s, e, t in rows[sid]["offsets"]]
        safe = firing_safe(masked)
        buf = io.StringIO()
        with redirect_stdout(buf):
            kept = api.filter_safe_spans(masked, ents)
        kept_ids = {(e["start"], e["end"]) for e in kept}
        for e in ents:
            key = (sid, e["start"], e["end"], masked[e["start"]:e["end"]])
            if (e["start"], e["end"]) not in kept_ids:
                vetoes.append(key)
                # was this an equal-span clock veto? (should never happen)
                for ss, se, nm in safe:
                    if nm == "clock_hhmm" and (ss, se) == (e["start"], e["end"]):
                        equal_clock_vetoes.append(key)
            elif key in HISTORY_YEARS and e["type"] == "DATE":
                kept_years.add(key)

    veto_set = set(vetoes)
    print("=" * 84)
    print("PILOT VETO AUDIT (21 notes) - Decision 2 bar")
    print("=" * 84)
    print(f"total vetoes: {len(vetoes)} (target 7)")
    for v in sorted(vetoes):
        tag = "OK" if v in EXPECTED_VETOES else "UNEXPECTED"
        print(f"  [{tag}] {v[0]} [{v[1]}:{v[2]}] {v[3]!r}")

    fails = []
    if len(vetoes) != 7:
        fails.append(f"veto count {len(vetoes)} != 7")
    if veto_set != EXPECTED_VETOES:
        missing = EXPECTED_VETOES - veto_set
        extra = veto_set - EXPECTED_VETOES
        if missing:
            fails.append(f"missing expected vetoes: {sorted(missing)}")
        if extra:
            fails.append(f"unexpected vetoes: {sorted(extra)}")
    if equal_clock_vetoes:
        fails.append(f"equal-span CLOCK vetoes present: {equal_clock_vetoes}")
    missing_years = HISTORY_YEARS - kept_years
    if missing_years:
        fails.append(f"history years NOT preserved as DATE: {sorted(missing_years)}")

    print(f"\nequal-span clock vetoes (must be 0): {len(equal_clock_vetoes)}")
    print(f"history years preserved as DATE (target 6): {len(kept_years)}/6")
    for y in sorted(HISTORY_YEARS):
        print(f"  [{'PRESERVED' if y in kept_years else 'MISSING'}] {y[0]} {y[3]!r}")

    print("-" * 84)
    if fails:
        print("RESULT: HOLD")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS (exactly 7 fragment vetoes, 0 equal-span clock vetoes, 6 years preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
