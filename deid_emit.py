"""Run the FULL production de-id pipeline over a corpus, write the de-identified
output, and print a PHI-safe quality + leak readout. Used to preview what the M&M
slide builder will receive. Args: <in1> <out1> [<in2> <out2> ...]"""
import os, re, sys
from collections import Counter

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")
import api  # noqa: E402
from deid import deidentify_text  # noqa: E402

NAME_T = ("FIRST_NAME", "LAST_NAME", "NAME")
TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rn", "np", "pa",
          "phd", "jr", "sr", "ii", "iii", "iv"}


def words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in TITLES]


def run(inp, outp):
    text = open(inp, encoding="utf-8", errors="replace").read()
    ner, _ = api.extract_entities(text)
    aug = ner + api.find_missed_names(text, ner)
    ents = api.filter_whitelisted_entities(aug)
    ents = ents + api.patient_sweep(text, ents)
    name_map = api.resolve_names(text, ents)
    deid = deidentify_text(text, [e.copy() for e in ents], 42, name_map=name_map)
    open(outp, "w", encoding="utf-8").write(deid)

    by = Counter(e["type"] for e in ents)
    firsts, lasts, alltoks = Counter(), Counter(), Counter()
    for e in ents:
        if e["type"] not in NAME_T:
            continue
        w = words(e.get("text", ""))
        if not w:
            continue
        for t in w:
            alltoks[t] += 1
        if e["type"] == "FIRST_NAME":
            firsts[w[0]] += 1
        elif e["type"] == "LAST_NAME":
            lasts[w[-1]] += 1
        else:
            if len(w) >= 2:
                firsts[w[0]] += 1
            lasts[w[-1]] += 1
    pf = firsts.most_common(1)[0][0] if firsts else None
    pl = lasts.most_common(1)[0][0] if lasts else None

    def resid(tok):
        return len(re.findall(r"\b" + re.escape(tok) + r"\b", deid, re.I)) if tok else 0

    survive_d = survive_n = 0
    for w, _ in alltoks.most_common(80):
        if w in api.MEDICAL_WHITELIST_LOWER:
            continue
        n = resid(w)
        if n:
            survive_d += 1
            survive_n += n

    print(f"\n========== {os.path.basename(inp)} ==========")
    print(f"chars={len(text):,}  PII entities replaced={len(ents)}")
    print("by type:", dict(by.most_common()))
    print("patient surrogate:", {k: v for k, v in (name_map or {}).items() if v} or "(fallback faker)")
    print(f"patient ORIGINAL first-token still in output: {resid(pf)}x | last-token: {resid(pl)}x  (want 0)")
    print(f"any original name-token surviving (excl. whitelist): "
          f"{survive_d} distinct / {survive_n} occ  (want ~0)")


for i in range(1, len(sys.argv), 2):
    run(sys.argv[i], sys.argv[i + 1])
