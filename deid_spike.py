"""Side-by-side SPIKE (not wired into prod): current Faker string-replacer vs a
Presidio span-based consistent-surrogate replacer. Reuses our Clinical-Longformer NER.

Run box-side so no PHI transits SSM. Self-contained synthetic note (no real PHI) by
default; pass a file path arg to run on your own text.

    DEID_MODEL_PATH=/opt/clinical-deid/model /opt/pytorch/bin/python deid_spike.py
"""
import os, re, sys

os.environ.setdefault("DEID_MODEL_PATH", "/opt/clinical-deid/model")

import api  # noqa: E402  (extract_entities, filter_whitelisted_entities, resolve_names)
from deid import deidentify_text  # noqa: E402

from faker import Faker  # noqa: E402
from presidio_anonymizer import AnonymizerEngine  # noqa: E402
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig  # noqa: E402
from presidio_anonymizer.operators import Operator, OperatorType  # noqa: E402

NAME_TYPES = {"FIRST_NAME", "LAST_NAME", "NAME"}
TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rn", "np", "pa",
          "phd", "jr", "sr", "ii", "iii", "iv"}


def _words(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in TITLES]


class ConsistentSurrogate(Operator):
    """Span-based operator. Shared `mapping` dict (passed by reference in params) keeps
    every variant of a person consistent WITHIN one document; `name_map` carries the
    patient's sex-matched override. State is per-anonymize-call = reset each document."""

    def operator_name(self):
        return "csurrogate"

    def operator_type(self):
        return OperatorType.Anonymize

    def validate(self, params=None):
        return

    def operate(self, text, params=None):
        role = params["role"]
        mapping = params["mapping"]
        nm = params["name_map"]
        fk = params["faker"]
        w = _words(text)
        if not w:
            return text

        def first_sur(tok):
            if tok in nm.get("first", {}):
                return nm["first"][tok]
            mapping.setdefault(("first", tok), fk.first_name())
            return mapping[("first", tok)]

        def last_sur(tok):
            if tok in nm.get("last", {}):
                return nm["last"][tok]
            mapping.setdefault(("last", tok), fk.last_name())
            return mapping[("last", tok)]

        if role == "first":
            return first_sur(w[0])
        if role == "last":
            return last_sur(w[-1])
        if len(w) == 1:
            return last_sur(w[0])
        return first_sur(w[0]) + " " + last_sur(w[-1])


def presidio_deid(text, entities, name_map):
    """Replace only NAME-type entities by SPAN (un-merged), leaving everything else."""
    eng = AnonymizerEngine()
    eng.add_anonymizer(ConsistentSurrogate)
    mapping, fk = {}, Faker()
    fk.seed_instance(20240601)
    results, cfg = [], {}
    shared = {"mapping": mapping, "name_map": name_map, "faker": fk}
    for e in entities:
        if e["type"] in NAME_TYPES:
            results.append(RecognizerResult(entity_type=e["type"], start=e["start"],
                                            end=e["end"], score=1.0))
    role = {"FIRST_NAME": "first", "LAST_NAME": "last", "NAME": "full"}
    for et, r in role.items():
        cfg[et] = OperatorConfig("csurrogate", dict(role=r, **shared))
    if not results:
        return text
    return eng.anonymize(text=text, analyzer_results=results, operators=cfg).text


SYNTHETIC = (
    "HPI: Michael A. Grant is a 60-year-old male with CKD and prior MI who presents "
    "with left flank pain. The patient was admitted overnight under Dr. Sarah Chen. "
    "Mr. Grant reports the pain began 3 days ago. On exam, Grant was afebrile. He was "
    "also evaluated by Dr. Mark Lee. Whitehead from case management coordinated "
    "discharge. Michael Grant's wife, Linda, was at bedside. Dr. Chen and Dr. Lee "
    "agreed on the plan. Nurse Iqbal administered morphine. Mr. Grant will follow up "
    "with Dr. Chen next week."
)


def main():
    text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else SYNTHETIC
    entities, _ = api.extract_entities(text)
    entities = api.filter_whitelisted_entities(entities)
    name_map = api.resolve_names(text, entities)

    names = [(e["text"], e["type"]) for e in entities if e["type"] in NAME_TYPES]
    print("PATIENT name_map:", {k: v for k, v in name_map.items() if v})
    print("NAME entities detected:", names)
    print("\n========== ORIGINAL ==========\n" + text)
    print("\n========== CURRENT (Faker string-replace + merge) ==========\n"
          + deidentify_text(text, [e.copy() for e in entities], 42, name_map=name_map))
    print("\n========== PRESIDIO (span-based, un-merged, consistent map) ==========\n"
          + presidio_deid(text, entities, name_map))


if __name__ == "__main__":
    main()
