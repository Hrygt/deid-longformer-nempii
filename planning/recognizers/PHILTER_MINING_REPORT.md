# Philter Mining Report: what to cannibalize for the clinical-deid recognizer layer

Date: 2026-07-05. Sources examined: BCHSI/philter-ucsf (research code, 313-entry
philter_delta.json config, filters/ tree) and SironaMedical/philter-lite (cleaned
pip-installable fork). Purpose: seed the "evolve, don't rewrite" hybrid layer for
clinical-deid with battle-tested patterns, adapted to our entity families and the
failure classes recorded in the 21-note ICU pilot.

## 0. Licensing (checked first)

Both repos are BSD 3-Clause, Regents of the University of California (2018).
Commercially compatible with RIGGSMED use. Requirements: retain the copyright
notice and license text for any copied pattern files or derived code, and do not
use UC's name for endorsement. Action: if we vendor patterns, include a
THIRD_PARTY_LICENSES file citing Philter (Norgeot et al., npj Digital Medicine
2020) and the BSD-3 text. Patterns we rewrite from scratch inspired by theirs
carry no obligation, but attribution is cheap and honest either way.

## 1. The architecture worth copying (not the engine)

Philter's engine is a config-driven ordered filter list. Each entry:

    { "title": ..., "type": "regex" | "set" | "regex_context" | "pos_matcher",
      "exclude": true | false, "filepath": ..., "phi_type": "DATE" | ... }

Two facts matter for us:

1. exclude=false entries are SAFE markers and run FIRST (131 of 313 entries in
   philter_delta). They claim spans as not-PHI before any PHI pattern runs.
   Example, filters/regex/safe/bp_safe.txt:
       \b(?i)bp(\:)?([ ]{1,2})?([6-9][0-9]|1[0-9][0-9])\/([2-9][0-9]|1[0-9][0-9])\b
   Value-bounded BP readings are claimed as safe so date patterns cannot eat
   "120/80". This is precisely our vitals-row problem, generalized.
2. exclude=true entries mark PHI, each tagged with a phi_type for surrogation.

Mapping to clinical-deid: our substitution pipeline already has ordered span
passes; the evolution is a detection-side recognizers/ module producing spans in
our {type,start,end,text} shape, unioned with NER output, plus a SAFE-span set
that vetoes both regex and NER detections. The SAFE veto is the piece we do not
have at all today, and it is the correct home for the clock-time (0610) and
room-number (RM06) classes.

## 2. Pattern sets worth taking, mapped to our recorded failures

### 2.1 Dates: the value-bounded grammar (44 files, filters/regex/dates/)

Philter's date fields are value-bounded, never lazy digit counts:
    month: (0?[1-9]|1[0-2])     day: ([1-2][0-9]|3[0-1]|0?[1-9])
Example (mm_dd_yyyy_transformed.txt):
    ((0?[1-9]|1[0-2])[\-\.\/\:]([1-2][0-9]|3[0-1]|0?[1-9])[\-.\/\:]\d{4}|...)

Why this matters to us: Branch 1's "maximal complete date token" grammar should
be built from these bounded alternations. A bounded grammar cannot validate "43"
(from 2043) or "10" (from 0610) as a date, and it defines exactly where a
legitimate date ENDS, which is the edge-trim question. The 44 files enumerate
written-date forms (month_name_dd_yyyy with ordinals, day_name variants, season
of yyyy, ranges like MM/DD-MM/DD) that our _parse_date accepts only partially.
Take: the bounded field vocabulary and the format inventory. Skip: their
"?"-tolerant variants (mm_dd_yy accepts literal ? placeholders, a UCSF corpus
quirk we do not have).

### 2.2 Addresses and ZIP (36 files, filters/regex/addresses/)

city_state_zip_transformed.txt is the composite that would have caught all 12
pilot ZIPs including the 0-for-7 ZIP+4 class: city token(s), then full state
name OR two-letter abbreviation (full 50-state alternation inline), then
[0-9]{5}(\-[0-9]{4})?. city_zip.txt adds a precision trick worth copying
verbatim: a negative lookahead excluding pager/page/pgr/MD lines so phone-ish
contexts are not eaten.

Adaptation for us: Epic address blocks are uppercase and line-structured; we
need the state-abbr + ZIP core (Branch 5 already ships it at the backstop
layer) promoted to a detection-time recognizer emitting STREET_ADDRESS /
POSTCODE spans, so the surrogates are family-correct instead of backstop
generalizations. The street-form files (at_street_number_dash_street, box_room,
corner_of_street) are a ready inventory for the 752LEY under-match class.

### 2.3 IDs (38 files, filters/regex/mrn_id/)

Named by shape: CCC-DD-DDDDD, CCDDDD-D-CC, DD-DDDDDC, DDCCCDDD, etc. (C=char,
D=digit). The lesson is the method, not the specific shapes: enumerate the ID
formats your institution actually emits and name each pattern after its shape.
For us that means an Epic-derived inventory: our MRN formats, CSN/account
formats, order IDs. Their model_number.txt sits in the same directory tagged
exclude=true, reminding us device model numbers are a false-positive class for
ID patterns.

### 2.4 Contact (filters/regex/contact/)

phone xxx_xxx_xxxx variants, pager (a distinct family we currently fold into
phone), email, url. Their pager handling plus the pager-exclusion lookahead in
city_zip is a matched pair: pager numbers are PHI-adjacent AND a ZIP false-
positive source. Take both sides.

### 2.5 The safe library (131 files, filters/regex/safe/)

The largest single asset. Categories directly relevant to our pilot failures:
bp_safe (vitals), units_safe, a1c_safe, AM_safe, age_safe, measurement
patterns. Missing from Philter but needed by us, same idiom: an HHMM clock-time
safe pattern for tabular vitals/procedure rows (\b([01]\d|2[0-3])[0-5]\d\b in
time-column context) and an RM\d{2} room-token safe pattern. Both of our
recorded false-positive classes become one-line SAFE entries in this scheme.

### 2.6 Date shifting (Philter V1.0, JAMIA Open 2023 paper, philter-ucsf code)

Their shift design: per-patient fixed offset, normalize detected date into a
datetime, infer missing components from defaults, then emit ONLY the components
present in the original text. That last clause is the fix for our site 6 class
and the 2-digit expansion defect by construction: a 2-character detected span
can never emit a 10-character surrogate, because output components mirror input
components. Recommend adopting this contract into _shift_date during Branch 3.

## 3. What NOT to take

- The engine itself (coordinate-map scanning, POS tagger dependency, in-place
  asterisk masking). Ours is better for surrogation and already has baselines.
- Their precision profile. On the VA transferability study Philter's F1 dropped
  to 0.34 excluding dates: rules-first over-redacts out of domain. We keep the
  NER as the primary name detector and use regex only where patterns are
  genuinely regular. The safe library is how Philter claws precision back, and
  we take that idea with the patterns.
- The 3.2MB whitelist JSON wholesale. Useful reference, but it encodes UCSF
  corpus statistics; ours should be grown from our own gold corpus.

## 4. Proposed integration (post corpus gate, own session)

Phase A: recognizers/ module with structured families only: postcode/address
block, phone/fax/pager, ssn, email, rigid dates. Spans union with NER before
the merge passes; conflict rule: recognizer wins on structured families, NER
wins on names. Each recognizer is one branch with red seeds from recorded
failures.
Phase B: SAFE-span veto set (clock times, RM tokens, vitals, units), applied to
both recognizer and NER spans before substitution. This is where the 0610 class
dies permanently, complementing Branch 4's expansion guard.
Phase C: Epic ID-shape inventory (from our own templates), Philter-style
shape-named files.
Phase D: adopt the components-mirroring date-shift contract in _shift_date.

Gates carry over unchanged: Nemotron + i2b2 detection baselines per branch
(recognizers CAN move detection numbers, unlike the substitution branches, so
here the gates finally measure the thing that changes), the 21-note fusion and
ZIP acceptance sweeps, and gold_delta-style before/after evidence.

## 5. Files staged in this package

- structured_recognizers_seed.py: curated starter module, patterns adapted from
  the sources above to our entity families, with per-pattern provenance
  comments, our span shape, and unit self-checks. It is a SEED for the future
  branch, not production code, and has not touched the clinical-deid repo.
