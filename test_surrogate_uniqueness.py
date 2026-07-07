"""Substitution-layer tests for per-document surrogate NAME uniqueness
(fix/surrogate-uniqueness).

Guards the name-identity confusion class witnessed in prod (metro-physicians
grader report 62ea5da405dc; the 2026-07-06 "Smith/Smith" de-id note where the
note author and the consulting cardiologist — two different people — both
surrogated to "Smith"). Three collision shapes, all silent today:

  A. two distinct originals draw the SAME surrogate surname;
  B. an original draws ITSELF (the real name survives "de-identification");
  C. an original draws a surname that is another REAL person's name in the
     same document.

Any of these poisons the surrogate map: the engine's re-id overlay
(AWS_Longform_Refactored/reid.py) applies bidirectional-uniqueness (rule 3)
and DROPS ambiguous pairs — correctly, never guessing — so the physician's
screen and Copy-MDM text carry a fake specialist name with no warning.

These tests exercise deid.deidentify_text DIRECTLY with crafted entities and a
DETERMINISTIC scripted draw sequence: ClinicalDeidentifier is subclass-patched
so its Faker returns a scripted list of names, simulating the unlucky draws
that are random in prod. No model / no GPU / no service.

Run:  .venv\\Scripts\\python test_surrogate_uniqueness.py
      (prints an evidence table, exits non-zero on failure)
"""
import re
import sys

import deid as deid_mod
from deid import deidentify_text

PASSED = 0
FAILED = 0


def check(name, cond, evidence=""):
    global PASSED, FAILED
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASSED += 1
    else:
        FAILED += 1
    print(f"  {tag}  {name}")
    if evidence:
        print(f"        [evidence] {evidence}")


class _ScriptedFake:
    """Faker stand-in: returns names from a script, repeating the last entry
    when exhausted. Only the two methods the name path uses."""

    def __init__(self, lasts, firsts):
        self._lasts, self._firsts = list(lasts), list(firsts)
        self._li = self._fi = 0

    def last_name(self):
        v = self._lasts[min(self._li, len(self._lasts) - 1)]
        self._li += 1
        return v

    def first_name(self):
        v = self._firsts[min(self._fi, len(self._firsts) - 1)]
        self._fi += 1
        return v


def run_deid(text, spans, lasts, firsts=("Alex", "Joan", "Omar")):
    """spans: list of (literal, entity_type). Returns (out_text, pairs)."""
    ents = []
    for literal, etype in spans:
        i = text.find(literal)
        assert i >= 0, f"span {literal!r} not in text"
        ents.append({"type": etype, "start": i, "end": i + len(literal), "text": literal})
    orig_cls = deid_mod.ClinicalDeidentifier

    class _Patched(orig_cls):
        def __init__(self, seed=None):
            super().__init__(seed=seed)
            self.fake = _ScriptedFake(lasts, firsts)

    deid_mod.ClinicalDeidentifier = _Patched
    try:
        out, smap = deidentify_text(text, ents, seed=7, return_map=True)
    finally:
        deid_mod.ClinicalDeidentifier = orig_cls
    return out, (smap or {}).get("pairs", [])


def surr_for(pairs, original_token):
    """Surrogate token(s) recorded for an original token (lowered compare)."""
    return sorted({p["surrogate"] for p in pairs
                   if p["original"].lower() == original_token.lower()})


# ---------------------------------------------------------------------------
print("=== A. two distinct people must not share a surrogate surname "
      "(the Smith/Smith class) ===")
# Replacement runs in reverse document order; the script hands out "Smith" twice.
text_a = "Attending: Robert Crawford, MD. Cardiology Dr Hood consulting - case discussed today."
out_a, pairs_a = run_deid(
    text_a, [("Robert Crawford", "NAME"), ("Hood", "LAST_NAME")],
    lasts=["Smith", "Smith", "Jones", "Vargas"])
s_crawford = surr_for(pairs_a, "crawford")
s_hood = surr_for(pairs_a, "hood")
check("A1. Crawford-surrogate != Hood-surrogate",
      bool(s_crawford) and bool(s_hood) and not set(s_crawford) & set(s_hood),
      f"crawford->{s_crawford} hood->{s_hood}")

# ---------------------------------------------------------------------------
print("=== B. an original must never draw ITSELF (real name would survive) ===")
# Single mention with its entity — detection coverage of repeat mentions is the
# NER's job, not this substitution layer's; here we only assert the DRAW.
text_b = "Seen by Dr Hood for AKI today."
out_b, pairs_b = run_deid(text_b, [("Hood", "LAST_NAME")], lasts=["Hood", "Serrano"])
check("B1. real surname absent from de-identified output",
      not re.search(r"\bHood\b", out_b),
      f"output: {out_b[:90]!r}")

# ---------------------------------------------------------------------------
print("=== C. a surrogate must not equal ANOTHER real name in the document ===")
# Zebrowski sits later in the text, so reverse-order processing draws for it
# FIRST — and the script offers it "Henderson", the real nephrologist's name.
text_c = "Dr Henderson (nephrology) following. D/W Dr Zebrowski - hold plavix."
out_c, pairs_c = run_deid(
    text_c, [("Henderson", "LAST_NAME"), ("Zebrowski", "LAST_NAME")],
    lasts=["Henderson", "Wolfe", "Vargas", "Nash"])
s_zeb = surr_for(pairs_c, "zebrowski")
check("C1. Zebrowski-surrogate is not 'Henderson' (a real person here)",
      s_zeb and "henderson" not in {s.lower() for s in s_zeb},
      f"zebrowski->{s_zeb}")

# ---------------------------------------------------------------------------
print("=== D. per-person consistency preserved (control — must hold both sides) ===")
text_d = "BiV ICD placed by Dr. Elizabeth Butler. D/W Dr. Butler - hold plavix."
out_d, pairs_d = run_deid(
    text_d, [("Elizabeth Butler", "NAME"), ("Butler", "LAST_NAME")],
    lasts=["Carter", "Nash"], firsts=["Joan"])
s_butler = surr_for(pairs_d, "butler")
check("D1. both Butler mentions share ONE surrogate surname",
      len(s_butler) == 1, f"butler->{s_butler}")

# ---------------------------------------------------------------------------
print("=== E. four-doctor aggregate: pairwise-distinct, disjoint from originals ===")
text_e = ("Dr Okonkwo (cardiology) consulted. Wound care per Dr Petrakis. "
          "ICD placed by Dr Vandelay. D/W Dr Lindqvist - hold heparin.")
reals_e = ["Okonkwo", "Petrakis", "Vandelay", "Lindqvist"]
out_e, pairs_e = run_deid(
    text_e, [(r, "LAST_NAME") for r in reals_e],
    lasts=["Smith", "Smith", "Smith", "Lee", "Ray", "Kim", "Fox", "Day"])
surrs_e = [s for r in reals_e for s in surr_for(pairs_e, r)]
check("E1. every doctor got exactly one surrogate",
      len(surrs_e) == 4, f"surrogates={surrs_e}")
check("E2. surrogates pairwise distinct",
      len({s.lower() for s in surrs_e}) == len(surrs_e), f"surrogates={surrs_e}")
check("E3. no surrogate equals any real name in the doc",
      not ({s.lower() for s in surrs_e} & {r.lower() for r in reals_e}),
      f"surrogates={surrs_e}")

# ---------------------------------------------------------------------------
print("=== F. pathological pool (every draw identical) still terminates ===")
text_f = "Dr Aaa and Dr Bbb reviewed the case."
out_f, pairs_f = run_deid(
    text_f, [("Aaa", "LAST_NAME"), ("Bbb", "LAST_NAME")], lasts=["Smith"])
check("F1. produced output despite an exhausted pool (bounded retries)",
      bool(out_f) and "Aaa" not in out_f and "Bbb" not in out_f,
      f"output: {out_f!r}")

print(f"\n{PASSED}/{PASSED + FAILED} passed")
sys.exit(1 if FAILED else 0)
