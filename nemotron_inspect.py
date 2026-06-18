"""Inspect nvidia/Nemotron-PII (synthetic, CC-BY-4.0): enumerate domains + label
types so we can map them to our entity set before scoring. Streams the test split."""
import ast
from collections import Counter
from datasets import load_dataset


def get_spans(ex):
    sp = ex.get("spans", [])
    if isinstance(sp, str):
        try:
            return ast.literal_eval(sp)
        except Exception:
            return []
    return sp

ds = load_dataset("nvidia/Nemotron-PII", split="test", streaming=True)

domains, labels, health_labels = Counter(), Counter(), Counter()
health_examples, N = [], 0
HEALTH_HINT = ("health", "medical", "clinic", "hospital", "surg", "blood", "patient", "pharma")

for ex in ds:
    N += 1
    dom = ex.get("domain", "?")
    domains[dom] += 1
    is_health = any(h in dom.lower() for h in HEALTH_HINT)
    for sp in get_spans(ex):
        labels[sp["label"]] += 1
        if is_health:
            health_labels[sp["label"]] += 1
    if is_health and len(health_examples) < 2:
        health_examples.append(ex)
    if N >= 8000:
        break

print(f"scanned {N} test examples")
print("\n=== DOMAINS (all) ===")
for d, c in domains.most_common():
    print(f"  {c:5d}  {d}")
print("\n=== LABEL TYPES (all domains) ===")
for l, c in labels.most_common():
    print(f"  {c:6d}  {l}")
print("\n=== LABEL TYPES (healthcare-ish domains only) ===")
for l, c in health_labels.most_common():
    print(f"  {c:6d}  {l}")
print("\n=== SAMPLE HEALTHCARE RECORD (synthetic) ===")
if health_examples:
    ex = health_examples[0]
    print("domain:", ex.get("domain"), "| document_type:", ex.get("document_type"),
          "| format:", ex.get("document_format"))
    print("text[:600]:\n", ex["text"][:600])
    print("spans[:15]:", get_spans(ex)[:15])
