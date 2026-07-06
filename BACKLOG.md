# clinical-deid backlog

Follow-up ledger. Opened 2026-07-05 after the fusion-fix batch merge
(tag deploy-2026-07-05-fusion-batch, payload SHA 12b89297).

## Branch 7: ZIP-prefix backstop extension (full-run tag over a ZIP+4 tail)

Branch 2's R3 sub-run insurance drops a ZIP+4 tail only when the NER tags a proper
SUB-run of the +4 (the real pilot's "25" of "9625"). When the NER tags the ENTIRE +4
run as an identifier, R3 does not fire (it is not a sub-run), the +4 is surrogated as
"MRN<digits>", and the Branch 5 backstop cannot re-generalize "74804-<letters>", so the
5-digit ZIP prefix survives. No real pilot note exhibits this; surfaced by the post-deploy
synthetic round trip.

Red seed: input "SHAWNEE OK 74804-9638" with the NER tagging the full "9638" as
MEDICAL_RECORD_NUMBER produces "74804-MRN1043" (74804 leaks). Fix direction: extend R3 (or
the Branch 5 backstop) to drop / re-generalize a full-run identifier tag that is the +4 of
a ZIP+4, so the ZIP backstop owns the whole ZIP. Owner: recognizer campaign.

## ASG schedule discrepancy (flag for Gary)

The starter doc / runbook notes describe a nightly 1-to-0 scale-down of the clinical-deid
box at 23:00. The actual ASG `cpt-dnn-service-asg` (instance i-09bb5c4ce5fa0a55f) has NO
scheduled actions (account-wide, none), Min 0 / Max 1 / Desired 1, one instance running
24/7. Decision needed: is the box intended to run 24/7 (then the doc is stale and the
"07:00-23:00 window" is advisory only), or should a nightly scale-down be (re)added to cut
GPU cost? 24/7 GPU cost vs stale doc. Owner: Gary.

## Accuracy Reporter: tabular-fusion residual retirement

The grader's scan_fusion "tabular_fusion" signature was a residual-detection heuristic for
the DATE-in-tabular class. With the batch at zero fusion signatures on the 21-note pilot,
retire (or downgrade) the tabular-fusion residual check once the first full 650-note batch
runs clean through the deployed service. Pending: first clean 650-batch. Owner: grader.

## NER detection errors neutralized at the span layer, not fixed at detection

This session's span-layer passes (Branches 1/2/6) neutralize several NER mis-tags AFTER
detection, so the outputs are clean, but the underlying detection errors remain and should
be fixed at the model / recognizer level:

- full ZIP+4 tail ("9638") tagged MEDICAL_RECORD_NUMBER
- "SH" (prefix of "SHAWNEE") tagged EMPLOYEE_ID
- procedure names ("ESOPHAGOGASTRODUODENOSCOPY") tagged UNIQUE_ID
- room tokens ("RM06"/"RM08") tagged POSTCODE

Owner: recognizer campaign (structured recognizers + SAFE library) and the ModernBERT
Phase 2 data argument (more Epic-tabular training data to teach the model these are not
identifiers).
