"""Probe surviving patient-name tokens in the de-id output: show neighbor context
(token masked, neighbors are post-scrub surrogates/clinical words) to tell a real
leak ("Mr. [X] was admitted") from a benign collision ("St. [X] Hospital")."""
import os, re, sys
from collections import Counter

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402
from deid import deidentify_text  # noqa: E402

TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rn", "np", "pa",
          "phd", "jr", "sr", "ii", "iii", "iv"}


def words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in TITLES]


text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ner, _ = api.extract_entities(text)
aug = ner + api.find_missed_names(text, ner)
ents = api.filter_whitelisted_entities(aug)
ents = ents + api.patient_sweep(text, ents)
nm = api.resolve_names(text, ents)
deid = deidentify_text(text, [e.copy() for e in ents], 42, name_map=nm)

firsts, lasts = Counter(), Counter()
for e in ents:
    w = words(e.get("text", ""))
    if not w:
        continue
    if e["type"] == "FIRST_NAME":
        firsts[w[0]] += 1
    elif e["type"] == "LAST_NAME":
        lasts[w[-1]] += 1
    elif e["type"] == "NAME":
        if len(w) >= 2:
            firsts[w[0]] += 1
        lasts[w[-1]] += 1
pf = firsts.most_common(1)[0][0] if firsts else None
pl = lasts.most_common(1)[0][0] if lasts else None
print("patient tokens (masked):", (pf[0] + '*' * (len(pf) - 1)) if pf else None,
      (pl[0] + '*' * (len(pl) - 1)) if pl else None,
      "-> surrogate", {k: v for k, v in (nm or {}).items() if v})

for tok, lab in [(pf, "FIRST"), (pl, "LAST")]:
    if not tok:
        continue
    hits = list(re.finditer(r"\b" + re.escape(tok) + r"\b", deid, re.I))
    print(f"\n== '{lab}' token survives {len(hits)}x (masked [X]) ==")
    for m in hits[:15]:
        s, e = m.start(), m.end()
        left = re.findall(r"\S+", deid[max(0, s - 35):s])[-4:]
        right = re.findall(r"\S+", deid[e:e + 35])[:4]
        print("   ...", " ".join(left), "[X]", " ".join(right), "...")
