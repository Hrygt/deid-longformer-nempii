"""Gazetteer false-positive / recall evaluator. Compares title-only recall vs
title+gazetteer on real corpora, box-side. Output is PHI-MINIMIZED:
  - FP-likely catches (token also appears lowercased in the corpus => a common word,
    NOT PHI) are shown in full -- these are the false positives we care about.
  - name-likely catches (never lowercased => probably a real missed name = PHI) are
    reported as counts + first-char/length only.
"""
import os, re, sys
from collections import Counter

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402


def evaluate(path, label):
    text = open(path, encoding="utf-8", errors="replace").read()
    ner, _ = api.extract_entities(text)

    api.RECALL_GAZETTEER = False
    title_only = api.find_missed_names(text, ner)
    api.RECALL_GAZETTEER = True
    title_plus = api.find_missed_names(text, ner)

    to_spans = {(e["start"], e["end"]) for e in title_only}
    delta = [e for e in title_plus if (e["start"], e["end"]) not in to_spans]

    lower_words = set(re.findall(r"\b[a-z]{2,}\b", text))
    fp, namelike = Counter(), Counter()
    for e in delta:
        tok = e["text"]
        (fp if tok.lower() in lower_words else namelike)[tok] += 1

    redacted = Counter()
    for tok, c in namelike.items():
        redacted[f"{tok[0]}{'*' * (len(tok) - 1)}"] += c

    print(f"\n===================== {label} =====================")
    print(f"chars={len(text)}  NER_entities={len(ner)}  "
          f"title_recall_adds={len(title_only)}  gazetteer_extra_adds={len(delta)}")
    print(f"\n  GAZETTEER FALSE-POSITIVE-LIKELY (common words, also seen lowercased) — "
          f"{sum(fp.values())} hits / {len(fp)} distinct:")
    print("   ", dict(fp.most_common(60)) or "(none)")
    print(f"\n  GAZETTEER NAME-LIKELY (never lowercased = probable real missed name) — "
          f"{sum(namelike.values())} hits / {len(namelike)} distinct (redacted):")
    print("   ", dict(redacted.most_common(60)) or "(none)")
    total = len(delta)
    if total:
        print(f"\n  => of {total} gazetteer-extra catches: "
              f"{sum(namelike.values())} look like real names, "
              f"{sum(fp.values())} look like false positives "
              f"({100 * sum(fp.values()) / total:.0f}% FP)")


for i, path in enumerate(sys.argv[1:], 1):
    evaluate(path, f"CORPUS_{i}  ({os.path.basename(path)})")
