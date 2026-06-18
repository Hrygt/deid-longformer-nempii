"""De-id WITH vs WITHOUT the gazetteer on real corpora and show exactly what the
gazetteer adds. Each gazetteer-only catch is classified:
  - EPONYM (a clinical term that is also a surname: Foley/Murphy/...) => a COST
    (would corrupt clinical meaning); these are medical terms, not PHI, shown in full.
  - NAME-LIKELY (not an eponym, never lowercased) => the intended BENEFIT (a real
    name the NER missed); PHI, so shown redacted (first char + length) + count only.
PHI-minimized; run box-side on tmpfs corpora.
"""
import os, re
from collections import Counter

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402

# Distinctive medical eponym surnames seen in IM/hospitalist notes. (Labeling only --
# used here to estimate the eponym false-positive rate, NOT to change de-id behavior.)
EPONYMS = {
    "foley", "murphy", "homans", "homan", "babinski", "glasgow", "kernig", "brudzinski",
    "cushing", "crohn", "bell", "graves", "hodgkin", "parkinson", "wernicke", "korsakoff",
    "charcot", "barrett", "whipple", "gleason", "child", "pugh", "ranson", "mallory",
    "weiss", "boerhaave", "cullen", "battle", "kussmaul", "cheyne", "stokes", "biot",
    "romberg", "horner", "chvostek", "trousseau", "dupuytren", "heberden", "bouchard",
    "osler", "janeway", "roth", "flint", "sjogren", "raynaud", "buerger", "kawasaki",
    "henoch", "schonlein", "berger", "alport", "conn", "sheehan", "hashimoto", "plummer",
    "riedel", "dequervain", "pott", "guillain", "barre", "fisher", "ramsay", "hunt",
    "duchenne", "becker", "marfan", "ehlers", "danlos", "klinefelter", "prader", "willi",
    "digeorge", "noonan", "pancoast", "virchow", "courvoisier", "rovsing", "kehr",
    "phalen", "tinel", "allen", "mcburney", "lhermitte", "wilms", "paget", "addison",
    "wegener", "goodpasture", "reynolds", "beck", "frank", "starling", "wells", "caprini",
    "centor", "meigs", "krukenberg", "troisier", "marjolin", "neer", "lachman", "mcmurray",
    "spurling", "hawkins", "wallenberg", "lambert", "eaton", "felty", "still", "reiter",
    "behcet", "takayasu", "buerger", "sister", "grey", "turner",
}


def diff(path, label):
    text = open(path, encoding="utf-8", errors="replace").read()
    ner, _ = api.extract_entities(text)
    api.RECALL_GAZETTEER = False
    base = api.find_missed_names(text, ner)
    api.RECALL_GAZETTEER = True
    full = api.find_missed_names(text, ner)
    base_spans = {(e["start"], e["end"]) for e in base}
    delta = [e for e in full if (e["start"], e["end"]) not in base_spans]

    eponym, namelike = Counter(), Counter()
    for e in delta:
        tok = e["text"]
        (eponym if tok.lower() in EPONYMS else namelike)[tok] += 1
    redacted = Counter()
    for tok, c in namelike.items():
        redacted[f"{tok[0]}{'*' * (len(tok) - 1)}"] += c

    n = len(delta)
    print(f"\n===================== {label} =====================")
    print(f"NER entities={len(ner)}  gazetteer adds {n} replacements that OFF would not make")
    print(f"\n  COST  — eponym/clinical-term hits (would corrupt meaning): "
          f"{sum(eponym.values())} hits / {len(eponym)} distinct")
    print("   ", dict(eponym.most_common(40)) or "(none)")
    print(f"\n  BENEFIT — name-likely hits (real names NER missed): "
          f"{sum(namelike.values())} hits / {len(namelike)} distinct (redacted)")
    print("   ", dict(redacted.most_common(40)) or "(none)")
    if n:
        print(f"\n  => {sum(eponym.values())}/{n} = {100*sum(eponym.values())/n:.0f}% of "
              f"gazetteer adds are eponym FPs; {sum(namelike.values())}/{n} look like real names")

    # Diagnose name-likely: show neighbor words (token stays redacted) to tell a real
    # missed name ("...,  Smith  said...") from a recurring non-name token.
    ctx = {}
    for e in delta:
        tok = e["text"]
        if tok.lower() in EPONYMS:
            continue
        red = f"{tok[0]}{'*' * (len(tok) - 1)}"
        s, en = e["start"], e["end"]
        left = re.findall(r"[A-Za-z]+", text[max(0, s - 18):s])
        right = re.findall(r"[A-Za-z]+", text[en:en + 18])
        ctx.setdefault(red, Counter())
        ctx[red][f"{(left[-1] if left else '^')} _ {(right[0] if right else '$')}"] += 1
    print("  name-likely neighbor patterns (left _ right):")
    for red, c in ctx.items():
        print(f"    {red}: {dict(c.most_common(5))}")


import sys
for i, p in enumerate(sys.argv[1:], 1):
    diff(p, f"CORPUS_{i} ({os.path.basename(p)})")
