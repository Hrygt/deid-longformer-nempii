"""PRIMARY gate for the SAFE-span veto (Branch 6): direct veto-versus-gold safety test,
ALL gold families (Decision 4).

deid_score.py scores the extraction path and never calls filter_safe_spans, so it cannot
see this branch. This test is the real synthetic-gold safety net: it feeds every gold PII
span through the REAL strict-containment filter_safe_spans and asserts the veto never drops
a gold span of ANY family on the pinned Nemotron split (N=1000, rev b70ffaf5). A veto on a
real PII span LEAKS, so any nonzero family count is a HOLD with the class named.

Structural blind spot (why the pilot audit gate also exists): Nemotron is synthetic and has
no Epic tabular history rows, so its date PHI is prose ("February 15, 2024"); this gate is
blind to tabular year cells. test_veto_pilot_audit.py covers that on real note structure.

Run:  DEID_SKIP_MODEL_LOAD=1 N=1000 python test_veto_vs_gold.py
"""
import ast
import io
import os
import sys
from collections import Counter
from contextlib import redirect_stdout

os.environ["DEID_SKIP_MODEL_LOAD"] = "1"  # model-free import of api
os.environ.setdefault("DEID_MODEL_PATH", "checkpoints/best_model")

import api  # noqa: E402  (filter_safe_spans + SAFE internals; no model loaded)
from labels import NEMOTRON_TO_ENTITY  # noqa: E402
from datasets import load_dataset  # noqa: E402

N = int(os.environ.get("N", "1000"))
REV = os.environ.get("NEMOTRON_REV", "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed")


def get_spans(ex):
    sp = ex.get("spans", [])
    if isinstance(sp, str):
        try:
            return ast.literal_eval(sp)
        except Exception:
            return []
    return sp


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def firing_safe_spans(text):
    spans = [(m.start(), m.end()) for m in api._SAFE_ROOM.finditer(text)]
    for m in api._SAFE_CLOCK.finditer(text):
        line, col = api._line_at(text, m.start())
        if api._clock_context_ok(line, col):
            spans.append((m.start(), m.end()))
    return spans


ds = load_dataset("nvidia/Nemotron-PII", split="test", revision=REV, streaming=True)
print(f"veto-vs-gold ALL FAMILIES | Nemotron-PII test @ rev {REV[:12]} | N={N} | metric: SAFE-veto safety (detection)")

docs = 0
gold_by_fam = Counter()
overlap_by_fam = Counter()
vetoed_by_fam = Counter()
samples = []
for ex in ds:
    text = ex["text"]
    gold = []
    for sp in get_spans(ex):
        ent = NEMOTRON_TO_ENTITY.get(sp.get("label"))
        if ent:
            gold.append((sp["start"], sp["end"], ent))
    if not gold:
        continue
    docs += 1
    for _, _, ent in gold:
        gold_by_fam[ent] += 1
    # overlap tally (context)
    safe = firing_safe_spans(text)
    for ss, se in safe:
        for gs, ge, ent in gold:
            if overlaps(ss, se, gs, ge):
                overlap_by_fam[ent] += 1
    # REAL veto: feed every gold span as an entity through the strict-rule filter
    ents = [{"type": ent, "start": gs, "end": ge, "text": text[gs:ge]} for gs, ge, ent in gold]
    buf = io.StringIO()
    with redirect_stdout(buf):
        kept = api.filter_safe_spans(text, ents)
    kept_ids = {(e["start"], e["end"], e["type"]) for e in kept}
    for gs, ge, ent in gold:
        if (gs, ge, ent) not in kept_ids:
            vetoed_by_fam[ent] += 1
            if len(samples) < 20:
                samples.append((ent, text[gs:ge], text[max(0, gs - 15):ge + 10].replace("\n", " ")))
    if docs >= N:
        break

print(f"\ngold docs scanned: {docs}")
print(f"{'FAMILY':<26}{'gold':>7}{'SAFE-overlap':>14}{'VETOED':>8}")
print("-" * 55)
for fam, c in gold_by_fam.most_common():
    print(f"{fam:<26}{c:>7}{overlap_by_fam[fam]:>14}{vetoed_by_fam[fam]:>8}")
print("-" * 55)
total_vetoed = sum(vetoed_by_fam.values())
print(f"{'TOTAL vetoed (HOLD if > 0)':<26}{'':>7}{'':>14}{total_vetoed:>8}")
if samples:
    print("\n-- VETOED gold samples (a HOLD) --")
    for ent, sv, ctx in samples:
        print(f"  [{ent}] {sv!r}  ...{ctx}...")

ok = (total_vetoed == 0)
print("\nRESULT:", "PASS (veto never drops a gold span of any family)" if ok else "HOLD (veto dropped gold PII)")
sys.exit(0 if ok else 1)
