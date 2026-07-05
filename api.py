# api.py - Clinical De-identification FastAPI Service
# Port 8001 on EC2 alongside CPT service (port 8000)
# v1.2.1 - Use deidentify_text from deid.py (includes DOB cleanup)

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import time
import os
import re

from labels import ID2LABEL, ENTITY_TYPES
from deid import ClinicalDeidentifier, deidentify_text, normalize_name
from medical_whitelist import MEDICAL_WHITELIST_LOWER

# === Configuration ===
MODEL_PATH = os.environ.get("DEID_MODEL_PATH", "checkpoints/best_model")
MAX_TOKENS = 4096
CHUNK_SIZE = 3500  # Leave room for special tokens
CHUNK_OVERLAP = 256  # Token overlap between chunks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_deployed_version() -> str:
    """Return the deployed commit SHA written to the VERSION file at deploy time
    (user-data captures `git rev-parse HEAD` before .git is discarded; the reconcile
    path rewrites it on overlay). Degrades to "unknown" on ANY problem — missing file,
    unreadable, or empty — and never raises, so /deid/health can never be broken by it."""
    try:
        path = os.environ.get("DEID_VERSION_FILE", "/opt/clinical-deid/VERSION")
        with open(path) as f:
            sha = f.read().strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


DEPLOYED_VERSION = _read_deployed_version()

# === Load Model (once at startup) ===
# DEID_SKIP_MODEL_LOAD lets model-free unit tests import this module (and its pure
# regex passes like filter_safe_spans) without a GPU/checkpoint. Off in production.
if os.environ.get("DEID_SKIP_MODEL_LOAD"):
    print("DEID_SKIP_MODEL_LOAD set: skipping model load (test/import mode)")
    tokenizer = None
    model = None
else:
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded: {model.num_parameters():,} parameters")

# === FastAPI App ===
app = FastAPI(
    title="Clinical De-identification API",
    description="PHI de-identification for clinical notes using Clinical-Longformer",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request/Response Models ===
class DeidRequest(BaseModel):
    text: str = Field(..., description="Clinical note text to de-identify")
    return_entities: bool = Field(False, description="Include detected entities in response")
    return_map: bool = Field(False, description="Include the original<->surrogate map for output-side re-identification")
    seed: Optional[int] = Field(None, description="Random seed for reproducible replacements")

class DeidResponse(BaseModel):
    deidentified_text: str
    entities: Optional[list] = None
    surrogate_map: Optional[dict] = None
    token_count: int
    processing_time_ms: float
    chunks_processed: int = 1

class ValidateRequest(BaseModel):
    text: str

class ValidateResponse(BaseModel):
    valid: bool
    token_count: int
    exceeds_limit: bool
    requires_chunking: bool
    estimated_chunks: int
    message: str

class BatchRequest(BaseModel):
    notes: list[str]
    return_entities: bool = False
    seed: Optional[int] = None

class BatchResponse(BaseModel):
    results: list[DeidResponse]
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    max_tokens: int
    chunk_size: int
    chunk_overlap: int
    version: str = "unknown"


# === Chunking Logic ===
def find_sentence_boundary(text: str, target_pos: int, search_range: int = 200) -> int:
    """Find the nearest sentence boundary to target_pos within search_range."""
    # Look for sentence endings near target position
    search_start = max(0, target_pos - search_range)
    search_end = min(len(text), target_pos + search_range)
    search_text = text[search_start:search_end]
    
    # Find all sentence boundaries in search range
    # Match period/question/exclamation followed by space and capital letter, or newlines
    boundaries = []
    for match in re.finditer(r'[.!?]\s+(?=[A-Z])|\n\s*\n', search_text):
        abs_pos = search_start + match.end()
        boundaries.append(abs_pos)
    
    if not boundaries:
        # No sentence boundary found, try just newlines
        for match in re.finditer(r'\n', search_text):
            abs_pos = search_start + match.end()
            boundaries.append(abs_pos)
    
    if not boundaries:
        return target_pos  # Fall back to target position
    
    # Return boundary closest to target
    return min(boundaries, key=lambda x: abs(x - target_pos))


def chunk_text(text: str) -> list[dict]:
    """
    Split text into overlapping chunks that fit within model's token limit.
    Returns list of dicts with 'text', 'start_char', 'end_char' keys.
    """
    # First check if chunking is even needed
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= MAX_TOKENS:
        return [{"text": text, "start_char": 0, "end_char": len(text)}]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Encode from current position to find chunk boundary
        remaining_text = text[current_pos:]
        
        # Binary search for the right chunk size in characters
        # Start with an estimate based on average chars per token
        avg_chars_per_token = len(text) / len(tokens)
        estimated_chars = int(CHUNK_SIZE * avg_chars_per_token)
        
        # Adjust to find actual token boundary
        low, high = estimated_chars // 2, min(estimated_chars * 2, len(remaining_text))
        
        while low < high:
            mid = (low + high + 1) // 2
            test_text = remaining_text[:mid]
            test_tokens = tokenizer.encode(test_text, add_special_tokens=True)
            
            if len(test_tokens) <= CHUNK_SIZE:
                low = mid
            else:
                high = mid - 1
        
        chunk_char_end = low
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if current_pos + chunk_char_end < len(text):
            # Calculate overlap in characters
            overlap_chars = int(CHUNK_OVERLAP * avg_chars_per_token)
            boundary = find_sentence_boundary(
                remaining_text, 
                chunk_char_end - overlap_chars,
                search_range=overlap_chars
            )
            if boundary > overlap_chars:  # Ensure we're making progress
                chunk_char_end = boundary
        
        chunk_text_content = remaining_text[:chunk_char_end]
        chunks.append({
            "text": chunk_text_content,
            "start_char": current_pos,
            "end_char": current_pos + chunk_char_end
        })
        
        # Move position, accounting for overlap
        if current_pos + chunk_char_end >= len(text):
            break
        
        # Calculate overlap
        overlap_chars = int(CHUNK_OVERLAP * avg_chars_per_token)
        next_pos = current_pos + chunk_char_end - overlap_chars
        
        # Ensure we're making progress
        if next_pos <= current_pos:
            next_pos = current_pos + chunk_char_end
        
        current_pos = next_pos
    
    return chunks


def merge_entities(all_entities: list[list[dict]], chunks: list[dict]) -> list[dict]:
    """
    Merge entities from multiple chunks, deduplicating overlapping detections.
    Entities are considered duplicates if they overlap significantly.
    """
    # Adjust entity positions to absolute positions in original text
    adjusted_entities = []
    
    for chunk_idx, (entities, chunk) in enumerate(zip(all_entities, chunks)):
        for entity in entities:
            adjusted = entity.copy()
            adjusted["start"] = chunk["start_char"] + entity["start"]
            adjusted["end"] = chunk["start_char"] + entity["end"]
            adjusted["_chunk"] = chunk_idx  # Track source chunk for debugging
            adjusted_entities.append(adjusted)
    
    if not adjusted_entities:
        return []
    
    # Sort by start position
    adjusted_entities.sort(key=lambda x: (x["start"], -x["end"]))
    
    # Deduplicate overlapping entities
    merged = []
    for entity in adjusted_entities:
        # Check if this entity overlaps with any already merged entity
        is_duplicate = False
        for existing in merged:
            # Calculate overlap
            overlap_start = max(entity["start"], existing["start"])
            overlap_end = min(entity["end"], existing["end"])
            
            if overlap_start < overlap_end:  # There is overlap
                overlap_len = overlap_end - overlap_start
                entity_len = entity["end"] - entity["start"]
                existing_len = existing["end"] - existing["start"]
                
                # If overlap is >50% of either entity, consider duplicate
                if overlap_len > 0.5 * entity_len or overlap_len > 0.5 * existing_len:
                    is_duplicate = True
                    # Keep the longer entity
                    if entity_len > existing_len:
                        merged.remove(existing)
                        merged.append(entity)
                    break
        
        if not is_duplicate:
            merged.append(entity)
    
    # Remove internal tracking field and sort by position
    for entity in merged:
        entity.pop("_chunk", None)
    
    merged.sort(key=lambda x: x["start"])
    return merged


# === Core Inference Logic ===
def extract_entities_single(text: str) -> list[dict]:
    """Run model inference on a single chunk and extract entity spans."""
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        return_offsets_mapping=True,
        padding=True,
    )
    
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    
    # Convert BILOU predictions to entity spans
    entities = []
    current_entity = None
    
    for idx, (pred_id, offsets) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens
        if offsets[0] == 0 and offsets[1] == 0:
            continue
            
        label = ID2LABEL.get(pred_id, "O")
        
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        tag, entity_type = label.split("-", 1)
        
        if tag == "B":  # Beginning
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            }
        elif tag == "I" and current_entity and current_entity["type"] == entity_type:
            # Inside - extend current entity
            current_entity["end"] = offsets[1]
            current_entity["text"] = text[current_entity["start"]:offsets[1]]
        elif tag == "L" and current_entity and current_entity["type"] == entity_type:
            # Last - close current entity
            current_entity["end"] = offsets[1]
            current_entity["text"] = text[current_entity["start"]:offsets[1]]
            entities.append(current_entity)
            current_entity = None
        elif tag == "U":  # Unit (single token entity)
            if current_entity:
                entities.append(current_entity)
            entities.append({
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            })
            current_entity = None
        else:
            # Invalid sequence - close and start new
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": entity_type,
                "start": offsets[0],
                "end": offsets[1],
                "text": text[offsets[0]:offsets[1]]
            }
    
    if current_entity:
        entities.append(current_entity)
    
    # Post-process: merge adjacent entities of the same type
    # This fixes cases where model predicts B-DATE, B-DATE, B-DATE instead of B-DATE, I-DATE, L-DATE
    entities = merge_adjacent_entities(entities, text)
    
    return entities


def merge_adjacent_entities(entities: list[dict], text: str, max_gap: int = 2) -> list[dict]:
    """
    Merge adjacent entities of the same type that are separated by at most max_gap characters.
    This fixes BILOU prediction errors where multi-token entities get split.
    
    Args:
        entities: List of entity dicts with 'type', 'start', 'end', 'text'
        text: Original text (to extract merged text spans)
        max_gap: Maximum character gap between entities to consider them mergeable
    
    Returns:
        Merged entity list
    """
    if not entities:
        return entities
    
    # Sort by start position
    sorted_ents = sorted(entities, key=lambda x: x["start"])
    
    merged = []
    current = sorted_ents[0].copy()
    
    for next_ent in sorted_ents[1:]:
        gap = next_ent["start"] - current["end"]
        same_type = next_ent["type"] == current["type"]
        
        # Merge if same type and adjacent (gap <= max_gap chars, typically punctuation/spaces)
        if same_type and 0 <= gap <= max_gap:
            # Check that gap only contains expected characters (punctuation, spaces, slashes, dashes)
            gap_text = text[current["end"]:next_ent["start"]]
            if all(c in " /-.:," for c in gap_text):
                # Extend current entity
                current["end"] = next_ent["end"]
                current["text"] = text[current["start"]:current["end"]]
                continue
        
        # Can't merge - save current and start new
        merged.append(current)
        current = next_ent.copy()
    
    merged.append(current)
    return merged


def _drop_name_subword_fragments(text: str, entities: list) -> list:
    """NER span-sanity guard: drop *_NAME entities that are mid-token sub-fragments of a longer
    LOWERCASE-continuing word — the NER's subword surrogation false-positive that mangles
    medication tokens BEFORE the grader sees them (clozapine->'cloz'->raymondozapine,
    Valium->'Val', methadone->'m'/'meth'). Dropping the fragment lets the drug token survive.

    CRITERION (length-INDEPENDENT; fragments are 1-4 chars so a length rule misses them): drop
    iff the character immediately AFTER the span (text[end]) is a LOWERCASE ascii letter — i.e.
    the span is a prefix cutting into a longer lowercase word. A real name boundary is never
    that: it is whitespace / punctuation, OR a camelCase UPPERCASE (the run-together
    'JohnSmith' -> NER spans 'John'(+'S') and 'Smith'(+' ') is PRESERVED — 'S' is uppercase and
    'Smith' has a right boundary). The left side is deliberately NOT checked: 'Smith' is
    preceded by the lowercase 'n' of 'John', so a left-side lowercase test would wrongly drop a
    real name (proven by the B.1 JohnSmith probe). Empirically every observed drug fragment is a
    prefix continuing lowercase, so the right-continuation test catches them all leak-free.

    *_NAME types ONLY — never touches SSN / MRN / DOB / DATE / ADDRESS / etc."""
    NAME_TYPES = ("FIRST_NAME", "LAST_NAME", "NAME")
    out = []
    for e in entities:
        if e.get("type") in NAME_TYPES:
            en = e.get("end")
            if en is not None and en < len(text) and ("a" <= text[en] <= "z"):
                continue  # mid-lowercase-word fragment -> drop (the subword surrogation bug)
        out.append(e)
    return out


def extract_entities(text: str) -> tuple[list[dict], int]:
    """
    Extract entities from text, automatically chunking if needed.
    Returns (entities, num_chunks).
    """
    chunks = chunk_text(text)

    if len(chunks) == 1:
        # No chunking needed
        return _drop_name_subword_fragments(text, extract_entities_single(text)), 1

    # Process each chunk
    all_chunk_entities = []
    for chunk in chunks:
        chunk_entities = extract_entities_single(chunk["text"])
        all_chunk_entities.append(chunk_entities)

    # Merge entities from all chunks
    merged_entities = merge_entities(all_chunk_entities, chunks)

    # Update entity text from original document (in case of boundary issues)
    for entity in merged_entities:
        entity["text"] = text[entity["start"]:entity["end"]]

    # NER span-sanity guard (subword surrogation FP) — applied to the FINAL spans (offsets
    # into the original text), so it is measured by deid_score.py (which calls extract_entities).
    return _drop_name_subword_fragments(text, merged_entities), len(chunks)


def filter_whitelisted_entities(entities: list[dict]) -> list[dict]:
    """
    Remove entities that match medical terminology whitelist.
    This prevents false positives like "Anion Gap" being replaced with fake names.
    Handles both single entities and adjacent name entities that form medical terms.
    Also filters out common time words misclassified as DATE entities.
    """
    if not entities:
        return entities
    
    # Common time words that get misclassified as DATE
    DATE_FALSE_POSITIVES = {
        "week", "weeks", "day", "days", "month", "months", "year", "years",
        "hour", "hours", "minute", "minutes", "second", "seconds",
        "morning", "afternoon", "evening", "night", "tonight", "today",
        "tomorrow", "yesterday", "daily", "weekly", "monthly", "yearly",
        "am", "pm", "noon", "midnight"
    }
    
    # Sort entities by position for adjacency checking
    sorted_entities = sorted(entities, key=lambda x: x["start"])
    
    # First pass: mark entities that are part of whitelisted multi-word terms
    skip_indices = set()
    
    for i in range(len(sorted_entities) - 1):
        curr = sorted_entities[i]
        next_ent = sorted_entities[i + 1]
        
        # Check if two adjacent name entities form a whitelisted term
        if (curr["type"] in ("FIRST_NAME", "LAST_NAME", "NAME") and 
            next_ent["type"] in ("FIRST_NAME", "LAST_NAME", "NAME")):
            
            # Check if they're actually adjacent (within a few chars, allowing for space)
            gap = next_ent["start"] - curr["end"]
            if 0 <= gap <= 3:  # Allow for space or small gap
                combined = f"{curr['text']} {next_ent['text']}".strip().lower()
                # Also try without space for hyphenated terms
                combined_no_space = f"{curr['text']}{next_ent['text']}".lower()
                
                if combined in MEDICAL_WHITELIST_LOWER or combined_no_space in MEDICAL_WHITELIST_LOWER:
                    skip_indices.add(i)
                    skip_indices.add(i + 1)
    
    # Second pass: filter entities
    filtered = []
    for i, entity in enumerate(sorted_entities):
        # Skip if marked for removal due to multi-word match
        if i in skip_indices:
            continue
            
        # Check single-entity matches for names
        if entity["type"] in ("FIRST_NAME", "LAST_NAME", "NAME"):
            text_lower = entity["text"].strip().lower()
            if text_lower in MEDICAL_WHITELIST_LOWER:
                continue  # Skip whitelisted term
        
        # Check for date false positives (common time words)
        if entity["type"] in ("DATE", "DATE_OF_BIRTH", "DATE_TIME"):
            text_lower = entity["text"].strip().lower()
            if text_lower in DATE_FALSE_POSITIVES:
                continue  # Skip common time words
        
        filtered.append(entity)
    
    return filtered


# === SAFE-span veto (Branch 6) ===
# Two SAFE patterns adapted from planning/recognizers/structured_recognizers_seed.py
# (Philter safe-filter idiom, BSD-3, UCSF Regents; Norgeot et al. 2020). A SAFE match
# claims text no detector may surrogate; a detected entity FULLY CONTAINED in a firing
# SAFE match is vetoed before name resolution and substitution.

# clock HHMM, value-bounded to valid clock values. Edge lookarounds reject adjacency to
# digits, slashes, colons, hyphens, PERIODS and COMMAS. The last two harden against the
# European date year (01.06.2028) and are the separator half of the fail-safe; the context
# gate below is the other half. Kills sites 4/5 + note 3 (DATE over an HHMM clock tail).
# Provenance: seed clock_hhmm.
_SAFE_CLOCK = re.compile(r"(?<![\d/:.,-])(?:[01]\d|2[0-3])[0-5]\d(?![\d/:.,-])")
# room token, fires unconditionally (the containment rule bounds it). Kills notes 11/12
# (POSTCODE over RM06/RM08). Provenance: seed room_token.
_SAFE_ROOM = re.compile(r"\bRM\d{2}\b")
# time cue that licenses a clock match OUTSIDE a tabular/cell line. Derived from the 158
# pilot matches: "at" is the only clean cue (18 "at HHMM"); IN/TO/OF/AROUND rejected (they
# precede years, e.g. "in 2019"); DING/STE/ALT were substring artifacts of ending/Suite.
_TIME_CUE = re.compile(r"(?i)\bat\s*$")


def _line_at(text: str, pos: int):
    """(line_text, column_within_line) for absolute offset pos."""
    ls = text.rfind("\n", 0, pos) + 1
    le = text.find("\n", pos)
    if le == -1:
        le = len(text)
    return text[ls:le], pos - ls


def _clock_context_ok(line: str, col: int) -> bool:
    """Context gate: a clock match fires only in tabular context (a tab), an Epic
    tabular-row cell (a line with no ASCII letters, a bare value column), or immediately
    after a time cue. Prose date lines have letters, no tab and no cue, so they never fire,
    which is what protects the 114 gold DATE/DOB year overlaps from the year-sweep."""
    if "\t" in line:
        return True
    if not any(c.isalpha() for c in line):
        return True
    return bool(_TIME_CUE.search(line[max(0, col - 8):col]))


def filter_safe_spans(text: str, entities: list[dict]) -> list[dict]:
    """Veto entities that fall inside a SAFE span (clock time / room token). Detection-side
    sibling of filter_whitelisted_entities: runs immediately after it and before
    patient_sweep, so vetoed spans never reach name resolution or substitution.

    CONTAINMENT rule: clock_hhmm vetoes only by STRICT (proper) containment -- the entity must
    be strictly smaller than and inside the match, so an equal-span 4-digit token is never
    vetoed and flows to substitution unchanged. room_token keeps FULL containment (equal
    allowed; RM tokens have no PHI collision class). Overlap without containment never vetoes.
    Each veto is logged like the other drop paths (print), so deid_log provenance stays complete."""
    if not entities:
        return entities
    safe = []  # (start, end, pattern_name)
    for m in _SAFE_ROOM.finditer(text):
        safe.append((m.start(), m.end(), "room_token"))
    for m in _SAFE_CLOCK.finditer(text):
        line, col = _line_at(text, m.start())
        if _clock_context_ok(line, col):
            safe.append((m.start(), m.end(), "clock_hhmm"))
    if not safe:
        return entities
    kept = []
    for e in entities:
        s, en = e.get("start"), e.get("end")
        vetoed_by = None
        if s is not None and en is not None:
            for ss, se, name in safe:
                if ss <= s and en <= se:  # entity contained in a SAFE match
                    # Decision 1: clock veto is STRICT -- an equal-span 4-digit token is left to
                    # the NER's judgment because a bare year and a clock time are indistinguishable
                    # at the pattern level, and an un-shifted year would let a reader triangulate
                    # the per-note date-shift offset. room_token keeps full (equal) containment.
                    if name == "clock_hhmm" and (s, en) == (ss, se):
                        continue  # equal-span clock: never vetoed (flows to substitution)
                    vetoed_by = name
                    break
        if vetoed_by is not None:
            print(f"[deid] SAFE-veto {vetoed_by} dropped {e.get('type')} span "
                  f"{s}:{en} {text[s:en]!r}")
            continue
        kept.append(e)
    return kept


# deidentify_text is now imported from deid.py (includes DOB cleanup)

RESOLVE_ENABLED = os.environ.get("DEID_RESOLVE", "on").lower() not in ("off", "0", "false", "no")

_TITLES = {"dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rn", "np", "pa",
           "phd", "jr", "sr", "ii", "iii", "iv"}


def _detect_sex(text: str) -> str:
    """Best-effort patient sex from the note text. Explicit sex words dominate;
    pronouns break ties. \\bmale\\b does NOT match inside 'female' (no word boundary)."""
    tl = text.lower()
    male = len(re.findall(r"\b(?:he|him|his|mr)\b", tl)) + 4 * len(re.findall(r"\b(?:male|gentleman)\b", tl))
    female = len(re.findall(r"\b(?:she|her|hers|mrs|ms|miss)\b", tl)) + 4 * len(re.findall(r"\b(?:female|woman|lady)\b", tl))
    if male > female:
        return "M"
    if female > male:
        return "F"
    return "U"


def resolve_names(text: str, entities: list) -> dict:
    """Identify THE PATIENT (the most-frequently-named person in the chart) and assign
    one sex-matched surrogate, keyed so every variant ("Michael", "Mr. Grant",
    "Michael A Grant") resolves to it. Deterministic — no LLM, so it can't echo a real
    name or mislabel a clinician. Returns {} on any issue (de-id then falls back to
    per-name faker surrogates). Disable with DEID_RESOLVE=off."""
    if not RESOLVE_ENABLED:
        return {}
    try:
        from collections import Counter

        def words(s):
            return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in _TITLES]

        firsts, lasts = Counter(), Counter()
        for e in entities:
            typ, w = e.get("type"), words(e.get("text") or "")
            if not w:
                continue
            if typ == "FIRST_NAME":
                firsts[w[0]] += 1
            elif typ == "LAST_NAME":
                lasts[w[-1]] += 1
            elif typ == "NAME":
                if len(w) >= 2:
                    firsts[w[0]] += 1
                lasts[w[-1]] += 1
        if not firsts and not lasts:
            return {}
        pat_first = firsts.most_common(1)[0][0] if firsts else None
        pat_last = lasts.most_common(1)[0][0] if lasts else None
        sex = _detect_sex(text)

        from faker import Faker
        import zlib
        fk = Faker()
        fk.seed_instance(zlib.crc32(((pat_first or "") + (pat_last or "")).encode()) & 0xFFFFFFFF)
        first_map, last_map = {}, {}
        if pat_first:
            first_map[pat_first] = (fk.first_name_male() if sex == "M"
                                    else fk.first_name_female() if sex == "F"
                                    else fk.first_name())
        if pat_last:
            last_map[pat_last] = fk.last_name()
        return {"first": first_map, "last": last_map, "full": {}}
    except Exception as exc:  # noqa: BLE001
        print(f"resolve_names fallback: {exc}")
        return {}


# --- Recall safety net: catch names the neural NER misses (e.g. "Nurse Iqbal") ---
RECALL_TITLES = os.environ.get("DEID_RECALL_TITLES", "on").lower() not in ("off", "0", "false", "no")
RECALL_GAZETTEER = os.environ.get("DEID_RECALL_GAZETTEER", "off").lower() in ("on", "1", "true", "yes")

_TITLE_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Mx|Nurse|Doctor|Prof|Professor|RN|NP|PA)\.?\s+"
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})"
)

# Credential-AFTER a name ("Robert Smith, MD", "Garcia, Taylor, RN"): a comma before the
# credential keeps precision high. Catches provider names the NER missed in this format
# (the title-BEFORE pass above does not see them). Comma required, so the verb "do" / the
# view "PA" don't false-match.
_CRED_RE = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+(?:,?\s+[A-Z][a-zA-Z'\-]+){0,2})\s*,\s*"
    r"(?:RN|LPN|MD|DO|NP|DNP|APRN|PA-C|PharmD|RPh|MSW|LCSW|LSW|DPM|DDS|CRNA|CNM|MBBS|FNP|AGNP|ACNP)\b"
)


def _load_gazetteer():
    """Known first/last names from Faker's bundled lists (no network)."""
    try:
        from faker.providers.person.en_US import Provider as P
        names = set()
        for attr in ("first_names", "first_names_male", "first_names_female",
                     "first_names_nonbinary", "last_names"):
            for n in getattr(P, attr, []) or []:
                n = n.strip().lower()
                if len(n) >= 2 and n.isalpha():
                    names.add(n)
        return names
    except Exception:  # noqa: BLE001
        return set()


_GAZETTEER = _load_gazetteer()


def find_missed_names(text: str, entities: list) -> list:
    """Recall net for names the NER missed. (1) Title-driven: a capitalized word after
    Dr./Mr./Nurse/... is almost certainly a name (high precision). (2) Gazetteer
    (opt-in, DEID_RECALL_GAZETTEER=on): a Title-case mid-sentence token that is a known
    name. Skips spans already covered and whitelisted medical terms. Returns NEW
    entities with offsets into `text`."""
    covered = [(e["start"], e["end"]) for e in entities if "start" in e and "end" in e]

    def is_covered(a, b):
        return any(a < ce and cs < b for cs, ce in covered)

    new = []
    if RECALL_TITLES:
        for rx in (_TITLE_RE, _CRED_RE):  # name BEFORE a title, and name BEFORE a credential
            for m in rx.finditer(text):
                base = m.start(1)
                for wm in re.finditer(r"[A-Za-z'\-]+", m.group(1)):
                    ws, we, tok = base + wm.start(), base + wm.end(), wm.group(0)
                    if len(tok) < 2 or tok.lower() in MEDICAL_WHITELIST_LOWER or is_covered(ws, we):
                        continue
                    new.append({"type": "NAME", "start": ws, "end": we, "text": tok, "score": 0.9})
                    covered.append((ws, we))
    if RECALL_GAZETTEER and _GAZETTEER:
        # Precision filter: a candidate whose lowercase form also occurs as a normal
        # word in THIS document ("Day"/day, "White"/white, "Case"/case) is a common
        # word, not a name. Real surnames almost never appear lowercased. This alone
        # removed ~all gazetteer false positives on the June/May M&M corpora.
        doc_lower = set(re.findall(r"\b[a-z]{2,}\b", text))
        for wm in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
            ws, we, tok = wm.start(1), wm.end(1), wm.group(1)
            low = tok.lower()
            if (low not in _GAZETTEER or low in MEDICAL_WHITELIST_LOWER
                    or low in doc_lower or is_covered(ws, we)):
                continue
            prev = text[max(0, ws - 2):ws].strip()
            if not prev or prev.endswith((".", "!", "?", ":", ";")):  # skip sentence-initial caps
                continue
            new.append({"type": "NAME", "start": ws, "end": we, "text": tok, "score": 0.6})
            covered.append((ws, we))
    return new


def patient_sweep(text: str, entities: list) -> list:
    """Once the patient is identifiable (the most-frequent name token), add entities for
    every CAPITALIZED whole-word occurrence of the patient's first/last token that the
    NER missed — e.g. it caught 'Grant' but left the adjacent 'Michael', or missed a
    name in an emergency-contact/header field. Capitalized-only ('Grant'/'GRANT', not
    the word 'grant') so common words aren't swept. Drives patient-name recall on long
    notes toward 100%, which matters most for the M&M subject."""
    from collections import Counter
    firsts, lasts = Counter(), Counter()
    for e in entities:
        if e.get("type") not in ("FIRST_NAME", "LAST_NAME", "NAME"):
            continue
        w = [x for x in re.findall(r"[a-z]+", (e.get("text") or "").lower()) if x not in _TITLES]
        if not w:
            continue
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
    covered = [(e["start"], e["end"]) for e in entities if "start" in e]

    def is_cov(a, b):
        return any(a < ce and cs < b for cs, ce in covered)

    new = []
    for tok, typ in ((pf, "FIRST_NAME"), (pl, "LAST_NAME")):
        if not tok or len(tok) < 2 or tok in MEDICAL_WHITELIST_LOWER:
            continue
        for m in re.finditer(r"\b" + re.escape(tok) + r"\b", text, re.I):
            if m.group(0).islower():  # skip the common-word lowercase form
                continue
            ws, we = m.span()
            if is_cov(ws, we):
                continue
            new.append({"type": typ, "start": ws, "end": we, "text": m.group(0), "score": 0.95})
            covered.append((ws, we))
    return new


# === API Endpoints ===
@app.get("/deid/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=DEVICE,
        max_tokens=MAX_TOKENS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        version=DEPLOYED_VERSION,
    )


@app.post("/deid/validate", response_model=ValidateResponse)
async def validate_text(request: ValidateRequest):
    """Pre-flight validation for text size."""
    tokens = tokenizer.encode(request.text, add_special_tokens=True)
    token_count = len(tokens)
    exceeds = token_count > MAX_TOKENS
    requires_chunking = exceeds
    
    # Estimate number of chunks needed
    if requires_chunking:
        effective_chunk_size = CHUNK_SIZE - CHUNK_OVERLAP
        estimated_chunks = (token_count + effective_chunk_size - 1) // effective_chunk_size
    else:
        estimated_chunks = 1
    
    message = f"Text has {token_count} tokens"
    if requires_chunking:
        message += f" (will be processed in ~{estimated_chunks} chunks)"
    
    return ValidateResponse(
        valid=True,  # Always valid now with chunking support
        token_count=token_count,
        exceeds_limit=exceeds,
        requires_chunking=requires_chunking,
        estimated_chunks=estimated_chunks,
        message=message,
    )


@app.post("/deid/process", response_model=DeidResponse)
async def process_text(request: DeidRequest):
    """De-identify a single clinical note (automatically chunks if needed)."""
    start_time = time.time()
    
    # Count tokens for response
    tokens = tokenizer.encode(request.text, add_special_tokens=True)
    
    # Extract entities (handles chunking automatically)
    entities, num_chunks = extract_entities(request.text)

    # Recall net: add names the NER missed (title-driven; opt-in gazetteer)
    entities = entities + find_missed_names(request.text, entities)

    # Filter out whitelisted medical terms (prevents false positives)
    entities = filter_whitelisted_entities(entities)

    # SAFE-span veto: drop NER FPs contained in a clock-time / room-token SAFE match
    # (clock-tail DATE FPs, RM0x POSTCODE FPs) before name resolution or substitution.
    entities = filter_safe_spans(request.text, entities)

    # Sweep NER-missed occurrences of the patient's own name tokens (recall safety net)
    entities = entities + patient_sweep(request.text, entities)

    # Resolve person-name identities (coref + sex-matched surrogates); {} = fall back.
    name_map = resolve_names(request.text, entities)

    # De-identify (optionally capturing the original<->surrogate map for output-side re-id)
    if request.return_map:
        deidentified, surrogate_map = deidentify_text(
            request.text, entities, request.seed, name_map=name_map, return_map=True)
    else:
        deidentified = deidentify_text(request.text, entities, request.seed, name_map=name_map)
        surrogate_map = None

    processing_time = (time.time() - start_time) * 1000

    return DeidResponse(
        deidentified_text=deidentified,
        entities=entities if request.return_entities else None,
        surrogate_map=surrogate_map,
        token_count=len(tokens),
        processing_time_ms=round(processing_time, 2),
        chunks_processed=num_chunks,
    )


@app.post("/deid/batch", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """De-identify multiple clinical notes."""
    start_time = time.time()
    
    if len(request.notes) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 notes")
    
    results = []
    for i, note in enumerate(request.notes):
        note_start = time.time()
        tokens = tokenizer.encode(note, add_special_tokens=True)
        
        entities, num_chunks = extract_entities(note)
        # Recall net: add names the NER missed (title-driven; opt-in gazetteer)
        entities = entities + find_missed_names(note, entities)
        # Filter out whitelisted medical terms
        entities = filter_whitelisted_entities(entities)
        # SAFE-span veto (same as /deid/process): drop clock-tail / room-token FPs
        entities = filter_safe_spans(note, entities)
        entities = entities + patient_sweep(note, entities)
        name_map = resolve_names(note, entities)
        # Use seed + index for reproducibility across batch
        seed = request.seed + i if request.seed else None
        deidentified = deidentify_text(note, entities, seed, name_map=name_map)
        
        results.append(DeidResponse(
            deidentified_text=deidentified,
            entities=entities if request.return_entities else None,
            token_count=len(tokens),
            processing_time_ms=round((time.time() - note_start) * 1000, 2),
            chunks_processed=num_chunks,
        ))
    
    return BatchResponse(
        results=results,
        total_processing_time_ms=round((time.time() - start_time) * 1000, 2),
    )


# Mount static files for UI
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
