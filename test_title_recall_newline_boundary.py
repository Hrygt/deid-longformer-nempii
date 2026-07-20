"""Title-recall newline-boundary tests (fix/name-span-newline-boundary, grader
parked-register item "CCPM under-detect on 466697e91ec0" — root cause).

Witnessed defect (2026-07-19, grader corpus note 466697e91ec0): `_TITLE_RE`'s name
capture used `\\s+` separators, so the title-driven recall net swept LINE-LEADING
CLINICAL TOKENS on following lines into the name run — "Dr. Mccoy\\nBradycardia\\n
Present" emitted NAME entities for 'Bradycardia' and 'Present'; "See Dr Tracy\\n
Propranolol is being held" ate 'Propranolol', destroying the grader's prescription-
management evidence (deployed Risk fell to Low). 75/912 grader gold notes carry the
exposed "Dr <Name>\\n<Capitalized token>" shape.

Ratified fix: name-run separators are HORIZONTAL whitespace only (`[^\\S\\r\\n]`) in
both `_TITLE_RE` and `_CRED_RE` — a name run never crosses a line break. Belt-and-
suspenders: 'propranolol' and 'bradycardia' join MEDICAL_WHITELIST (clinical terms,
never plausible person names). 'Will'/'Present' are deliberately NOT whitelisted —
'Will' is a real first name and PHI safety errs toward redaction; the structural
newline fix is what protects them in the witnessed shape.

Recall-net layer, NO model / NO GPU (DEID_SKIP_MODEL_LOAD). Modeled on
test_name_span_fragment_repair.py.

Run:  python test_title_recall_newline_boundary.py
"""
import os

os.environ.setdefault("DEID_SKIP_MODEL_LOAD", "1")

from api import find_missed_names  # noqa: E402
from medical_whitelist import MEDICAL_WHITELIST  # noqa: E402

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def toks(text):
    return sorted(e["text"] for e in find_missed_names(text, []))


print("=== Witnessed shapes (466697e91ec0) ===")
t1 = ("Cardiology consulting- Dr. Mccoy\nBradycardia\nPresent on Admission: Yes\n"
      "In 50s upon arrival to ED.")
print(f"        [Dr. Mccoy + 2 line-leading tokens] -> {toks(t1)}")
check("'Dr. Mccoy\\nBradycardia\\nPresent' -> ONLY Mccoy swept",
      toks(t1) == ["Mccoy"])
t2 = "Essential tremor\nSee Dr Tracy\nPropranolol is being held; will monitor off this medication."
print(f"        [Dr Tracy + line-leading drug] -> {toks(t2)}")
check("'Dr Tracy\\nPropranolol' -> ONLY Tracy swept (the drug survives)",
      toks(t2) == ["Tracy"])
t3 = "Dr. Mccoy\nWill continue to monitor"
print(f"        [Dr. Mccoy + line-leading 'Will'] -> {toks(t3)}")
check("'Dr. Mccoy\\nWill continue' -> ONLY Mccoy ('Will' not eaten)",
      toks(t3) == ["Mccoy"])

print("\n=== Same-line recall PRESERVED ===")
check("'Dr. Mary Ann Walker' same-line multi-token name still swept in full",
      toks("Seen by Dr. Mary Ann Walker today.") == ["Ann", "Mary", "Walker"])
check("'Dr. Smith' single token still swept",
      toks("Discussed with Dr. Smith about the plan.") == ["Smith"])
check("credential form 'Robert Johnson, MD' still swept (same line)",
      toks("Signed: Robert Johnson, MD") == ["Johnson", "Robert"])

print("\n=== Credential form does not cross lines ===")
t4 = "plan per notes\nSmith\nPotassium, MD reviewed"
print(f"        [cred run across newline] -> {toks(t4)}")
check("'Smith\\nPotassium, MD' -> the run does NOT bridge the newline to 'Smith'",
      "Smith" not in toks(t4))

print("\n=== Whitelist belt-and-suspenders ===")
wl = {w.lower() for w in MEDICAL_WHITELIST}
check("'propranolol' whitelisted", "propranolol" in wl)
check("'bradycardia' whitelisted", "bradycardia" in wl)
check("'will' deliberately NOT whitelisted (real first name; PHI asymmetry)",
      "will" not in wl)
check("whitelist blocks even a SAME-LINE 'Dr. Propranolol' sweep",
      toks("held per Dr. Propranolol note") == [])

print(f"\n=== {_passed} passed, {_failed} failed ===")
import sys  # noqa: E402
sys.exit(1 if _failed else 0)
