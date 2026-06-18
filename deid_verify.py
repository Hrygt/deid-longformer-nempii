"""Re-de-id a corpus with the current pipeline and report the two fixes' effect:
malformed-date count + provider-context leak candidates (name tokens surviving into
the de-id output from the original). PHI-minimized (counts + redacted)."""
import os, re, sys
os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402
from deid import deidentify_text  # noqa: E402

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
ner, _ = api.extract_entities(text)
aug = ner + api.find_missed_names(text, ner)
ents = api.filter_whitelisted_entities(aug)
ents = ents + api.patient_sweep(text, ents)
nm = api.resolve_names(text, ents)
deid = deidentify_text(text, [e.copy() for e in ents], 42, name_map=nm)

bad = re.findall(r"\b\d{1,2}/\d{1,2}/\d{5,}\b", deid)
print("malformed dates (>=5-digit year):", len(bad), bad[:6])

# PRECISE leak check: provider name-tokens in the ORIGINAL that no entity covers, so they
# won't be replaced. No surrogate-coincidence noise. [A-Z][a-z]{2,} skips all-caps creds.
spans = sorted((e["start"], e["end"]) for e in ents if "start" in e and "end" in e)


def covered(a, b):
    return any(a < ce and cs < b for cs, ce in spans)


prov_pats = [r"\b[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+){0,2}\s*,\s*"
             r"(?:RN|LPN|MD|DO|NP|DNP|APRN|PharmD|MSW|LCSW|DPM|CRNA|CNM|FNP)\b",
             r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}",
             r"\bNurse\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"]
uncovered = []
for p in prov_pats:
    for m in re.finditer(p, text):
        for wm in re.finditer(r"[A-Z][a-z]{2,}", m.group(0)):
            ws, we = m.start() + wm.start(), m.start() + wm.end()
            if wm.group(0).lower() in ("nurse",) or covered(ws, we):
                continue
            uncovered.append(wm.group(0))
print("provider name-tokens in original NOT covered (would LEAK):", len(uncovered),
      [t[0] + "*" * (len(t) - 1) for t in uncovered][:20])
