# Clinical De-id — Runtime Pipeline Reference

How the **de-identification service** turns a clinical note into surrogate-replaced
text at request time. This is the runtime/inference reference (not training — see
`MODEL_CARD.md`). The portal-side / app view is in
`metro-physicians/docs/applications/deidentify.md`.

Surrogate replacement, NOT redaction: every PHI span is swapped for a consistent,
realistic stand-in, so the output still reads like a real note.

## Files

| File | Role |
|------|------|
| `api.py` | FastAPI service. `extract_entities` (NER + chunking), `find_missed_names`, `patient_sweep`, `resolve_names`, `filter_whitelisted_entities`, the `/deid/process` + `/deid/health` endpoints. |
| `deid.py` | `deidentify_text` + the `ClinicalDeidentifier` surrogate engine (name/date/ID replacement, consistency caches, patient backstop, date sanitizer). |
| `labels.py` | 25 PHI `ENTITY_TYPES` + `NEMOTRON_TO_ENTITY` (the model's label space = Nemotron-PII's). |
| `medical_whitelist.py` | Terms that must NOT be surrogated: labs/chem, role words (`Pt`…), ultra-distinctive eponyms. |

## Request flow (`process_text`)

```
extract_entities            NER over chunked text -> {start,end,text,type}[]
  + find_missed_names        recall net (title-before, credential-after, opt-in gazetteer)
  -> filter_whitelisted      drop medical terms / role words / eponyms
  -> patient_sweep           tag capitalized occurrences of the patient's own name tokens
  -> resolve_names           patient = most-frequent name; sex from text -> sex-matched surrogate map
  -> deidentify_text(name_map)   replace, with a patient-name backstop + date sanitizer
```

## Mechanisms (and the bug each fixes)

- **Token-consistent surrogates** (`_first_tokens`/`_last_tokens`, keyed by the
  first/last *word*): `Veronica Brown` / `Brown` / `Mr. Brown` → the SAME fake;
  reset per document. (Was: per-`(type,text)` keying → different fakes; and
  `A. Grant` vs `Grant` keyed differently.)
- **Sex-matched patient surrogate** (`resolve_names` + `_detect_sex`): "60yo male"
  no longer becomes "Jennifer". Deterministic (an LLM-clustering attempt echoed
  single tokens and mislabeled clinicians — reverted).
- **`_split_name_spans`** (after the name-combine, before replace): splits a span
  the NER ran across a sentence boundary (`Lee. Whitehead` → two names), keeping the
  period; single-letter initials (`A. Grant`) are not split.
- **`find_missed_names`** — recall for names the NER misses:
  *title-before* (`Dr./Mr./Nurse X`), *credential-after* (`Name, RN/DO/MD`,
  `Last, First, Cred`, comma-gated), and an *opt-in gazetteer*
  (`DEID_RECALL_GAZETTEER`, **off** — measured ~0 real-name recall on M&M corpora).
- **`patient_sweep` + patient backstop** (in `deidentify_text`): the NER sometimes
  *detects* but doesn't *replace* a patient mention (covered by a DATE span / a
  partial `Michael <mid> Grant` replacement), leaking the name. The backstop uses
  `name_map` to replace any capitalized surviving patient token. Drives patient-name
  recall to ~100%.
- **Date merge + sanitizer**: `_merge_adjacent_date_entities` folds an interspersed
  `TIME` into the date span so date+time shifts as ONE unit (was `08/14/202554`);
  a post-process strips digit garbage trailing a complete date. `_parse_date` has a
  dateutil fallback; the unparseable-date fallback hashes the input (no collapse).
- **Eponym whitelist** (`medical_whitelist.py`): ultra-distinctive only
  (`Babinski`, `Wernicke`, `Dupuytren`…). Common-surname eponyms (`Parkinson`,
  `Cushing`, `Crohn`…) are deliberately EXCLUDED — whitelisting them would leak a
  patient actually named that, and the NER already preserves them.

## Config flags (env)

`DEID_RESOLVE` (patient sex-matching; on) · `DEID_RECALL_TITLES` (title + credential
recall; on) · `DEID_RECALL_GAZETTEER` (off) · `DEID_MODEL_PATH`
(`/opt/clinical-deid/model`).

## Evaluation harnesses

- `deid_score.py` — recall scorecard vs **nvidia/Nemotron-PII** test split (synthetic,
  CC-BY-4.0, the model's native labels). **Baseline: 98.3% span recall, FIRST 99.2% /
  LAST 99.4%, ~100% SSN/MRN/DOB/phone/email/address.** Run before/after any change.
- `deid_spike.py` — Presidio comparison (no output gain; remaining leaks are
  NER-level). `deid_gaz_eval.py` / `deid_gaz_diff.py` — gazetteer FP study.
  `deid_emit.py` / `deid_leak_probe.py` / `deid_verify.py` — corpus leak checks
  (counts + redacted; safe).

## Deploy & persistence

- Service: `clinical-deid.service` on GPU box `i-0cbcf0d83ab29ff96`, `:8001`, venv
  `/opt/pytorch`, code in `/opt/clinical-deid/`.
- The box **re-clones this repo on its nightly 11 PM reboot** → every fix MUST be on
  `origin/main` or it's lost. (Repo renamed to `deid-longformer-nempii`; old clone
  URL works via redirect.)
- Live deploy: base64 the changed files over SSM to `/opt/clinical-deid/`, `python
  -m py_compile`, `systemctl restart clinical-deid`, smoke-test `/deid/health`.

## PHI handling when testing

Use **synthetic** data over SSM (command logs are retained — never put real PHI in a
command). Stage real corpora via a **presigned S3 URL** (SSE-KMS bucket) → fetch to
`/dev/shm` (tmpfs) → `shred` → delete the object. The de-id *output* is safe; the
only risk is a residual leak. Bedrock calls (sex-resolution) run under the AWS BAA.
