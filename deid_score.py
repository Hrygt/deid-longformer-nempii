"""Score our de-id DETECTION against nvidia/Nemotron-PII test split (synthetic, the
model's native label space via labels.NEMOTRON_TO_ENTITY).

Recall = the leak metric: of all gold PII spans, how many did we detect (overlap)?
Reports overall + per-type, NER-only vs +recall-passes (find_missed_names), plus a
rough precision and a sample of real misses (safe: synthetic).

    N=1000 DEID_MODEL_PATH=/opt/clinical-deid/model /opt/pytorch/bin/python deid_score.py
"""
import os, ast
from collections import Counter

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402
from labels import NEMOTRON_TO_ENTITY  # noqa: E402
from datasets import load_dataset  # noqa: E402

N = int(os.environ.get("N", "1000"))
HEALTH_HINT = ("health", "medical", "clinic", "hospital", "surg", "blood", "patient", "pharma")


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


ds = load_dataset("nvidia/Nemotron-PII", split="test", streaming=True)

gold_total, rec_aug, rec_ner = Counter(), Counter(), Counter()
h_gold, h_rec = Counter(), Counter()
pred_total = pred_hit = 0
misses = []
docs = 0

for ex in ds:
    text = ex["text"]
    gold = []
    for sp in get_spans(ex):
        ent = NEMOTRON_TO_ENTITY.get(sp.get("label"))
        if ent:
            gold.append((sp["start"], sp["end"], ent, sp.get("text", "")))
    if not gold:
        continue
    docs += 1
    ner, _ = api.extract_entities(text)
    aug = ner + api.find_missed_names(text, ner)
    pner = [(e["start"], e["end"]) for e in ner if "start" in e]
    paug = [(e["start"], e["end"]) for e in aug if "start" in e]
    is_h = any(h in ex.get("domain", "").lower() for h in HEALTH_HINT)

    for gs, ge, ent, gt in gold:
        gold_total[ent] += 1
        if is_h:
            h_gold[ent] += 1
        if any(overlaps(gs, ge, ps, pe) for ps, pe in paug):
            rec_aug[ent] += 1
            if is_h:
                h_rec[ent] += 1
        elif len(misses) < 50:
            misses.append((ent, gt, text[max(0, gs - 20):ge + 20].replace("\n", " ")))
        if any(overlaps(gs, ge, ps, pe) for ps, pe in pner):
            rec_ner[ent] += 1
    for ps, pe in paug:
        pred_total += 1
        if any(overlaps(ps, pe, gs, ge) for gs, ge, _, _ in gold):
            pred_hit += 1
    if docs >= N:
        break


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


gt = sum(gold_total.values())
print(f"\n==== Nemotron-PII test: {docs} docs with PHI, {gt} gold spans ====")
print(f"\nOVERALL span recall  NER-only: {pct(sum(rec_ner.values()), gt)}   "
      f"+recall-passes: {pct(sum(rec_aug.values()), gt)}")
print(f"OVERALL precision (pred overlapping a gold span): {pct(pred_hit, pred_total)} "
      f"({pred_hit}/{pred_total})")
print(f"\n{'TYPE':<32}{'gold':>7}{'NER':>9}{'+recall':>9}")
for ent, c in gold_total.most_common():
    print(f"{ent:<32}{c:>7}{pct(rec_ner[ent], c):>9}{pct(rec_aug[ent], c):>9}")

print(f"\n-- NAME recall (the leak focus) --")
for ent in ("FIRST_NAME", "LAST_NAME"):
    print(f"  {ent}: {pct(rec_aug[ent], gold_total[ent])}  ({rec_aug[ent]}/{gold_total[ent]})")

if h_gold:
    print(f"\n-- HEALTHCARE-domain subset ({sum(h_gold.values())} gold spans) --")
    for ent, c in h_gold.most_common(12):
        print(f"  {ent:<30}{c:>6}{pct(h_rec[ent], c):>9}")

print(f"\n-- sample MISSED gold spans (synthetic; type | gold | context) --")
for ent, g, ctx in misses[:30]:
    print(f"  [{ent}] {g!r}  ...{ctx}...")
