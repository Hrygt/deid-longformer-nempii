# deid.py - Clinical De-identification Surrogate Generator
# v2.4.0 - Fix short date parsing (MM/DD) and consistent date fallback
#
# Key features:
# - Consistent replacement within documents (same PHI → same fake)
# - Format preservation (phone, SSN, dates match original format)
# - Realistic surrogates (not bracketed placeholders)
# - Date shifting preserves temporal relationships
# - Short date support (12/13 → parsed as current year, shifted consistently)
# - Geographic consistency (city/state/zip match)
# - Name normalization (Dr. Sarah Johnson, MD → same fake as Sarah Johnson)
# - DOB line cleanup (prevents concatenated date artifacts)
# - Age-preserving DOB (±2 year jitter keeps patient age realistic)
# - Context-aware DOB detection (DATE after "DOB:" gets age-preserving replacement)
# - Adjacent name combination (FIRST_NAME + LAST_NAME → NAME for consistent cache lookup)
# - DATE entity merging (fixes model fragmenting dates into pieces)

from faker import Faker
from datetime import datetime, timedelta
import random
import re
from typing import Optional, Tuple

# Whitelist of clinical terms (lowercased) used only to flag surrogate-name collisions
# in the optional re-identification map. Guarded import so existing callers that don't
# have the module on path keep working (the flag just degrades to "no collision known").
try:
    from medical_whitelist import MEDICAL_WHITELIST_LOWER
except Exception:  # noqa: BLE001
    MEDICAL_WHITELIST_LOWER = set()


def normalize_name(name: str) -> str:
    """
    Normalize a clinician/patient name so that variants like
    'Dr. Sarah Elizabeth Johnson, MD' and 'Sarah E. Johnson, MD'
    map to the same cache key.
    """
    n = name.lower()
    # Remove common titles/credentials
    n = re.sub(r'\b(dr\.?|md|do|phd|rn|np|pa\-c|dpm|dds|od|pharmd|pt|ot|cna|lpn|lvn|aprn|crna|dnp|mph|ms|ma|bs|ba|jr\.?|sr\.?|ii|iii|iv)\b', '', n)
    # Remove non-letter characters except spaces
    n = re.sub(r'[^a-z\s]', ' ', n)
    # Collapse multiple spaces
    n = re.sub(r'\s+', ' ', n).strip()
    parts = n.split()
    if len(parts) >= 2:
        # First + last name only; ignore middle/initials
        return f"{parts[0]} {parts[-1]}"
    return n.strip()


# DOB line cleanup regex patterns
DOB_LINE_RE = re.compile(r'^(DOB:\s*)(.+)$', re.MULTILINE | re.IGNORECASE)
DATE_RE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')

# Marker for the _shift_date output contract (fix/date-shift-contract). Tests import-guard
# this to label whether the components-mirroring contract is present.
_DATE_CONTRACT = True


def _clean_dob_lines(text: str) -> str:
    """
    Post-process DOB lines so that if multiple dates ended up on the same line
    (e.g., from overlapping entity replacements), we keep only the first clean date.
    """
    def repl(match: re.Match) -> str:
        prefix, rest = match.groups()
        dates = DATE_RE.findall(rest)
        if not dates:
            return match.group(0)
        # Use the first detected date; discard any extra concatenated dates
        return f"{prefix}{dates[0]}"
    return DOB_LINE_RE.sub(repl, text)


def _merge_adjacent_date_entities(entities: list, text: str) -> list:
    """
    Merge adjacent DATE/DATE_TIME/DATE_OF_BIRTH entities that got fragmented by the model.
    For example, "03/15/1965" might get split into "03", "/", "15", "/", "1965".
    This merges them back into a single entity.
    """
    date_types = {"DATE", "DATE_TIME", "DATE_OF_BIRTH"}
    
    # Sort by position
    sorted_ents = sorted(entities, key=lambda x: x["start"])
    
    merged = []
    i = 0
    while i < len(sorted_ents):
        curr = sorted_ents[i]
        
        if curr["type"] not in date_types:
            merged.append(curr.copy())
            i += 1
            continue
        
        # Try to combine with following date-like entities
        group_start = curr["start"]
        group_end = curr["end"]
        j = i + 1
        
        while j < len(sorted_ents):
            next_ent = sorted_ents[j]
            # Accept any date-type entity or even short gaps
            if next_ent["type"] not in date_types:
                gap = next_ent["start"] - group_end
                if gap > 3:  # Too far apart
                    break
                # Fold a TIME the model split out of the date into the SAME span so the
                # date+time shifts as one unit; otherwise the pieces shift independently
                # and run together ("08/14/2025" + "5:54" -> "08/14/202554"). Don't absorb
                # other entity types (could hide a real identifier sitting next to a date).
                if next_ent["type"] == "TIME":
                    group_end = max(group_end, next_ent["end"])
                    j += 1
                    continue
                break
            
            gap = next_ent["start"] - group_end
            # Allow gap of up to 1 char for slashes/dashes that might not be entities
            if gap > 1:
                break
            
            # Check gap only contains date separators
            gap_text = text[group_end:next_ent["start"]]
            if gap_text and not all(c in "/-" for c in gap_text):
                break
            
            group_end = next_ent["end"]
            j += 1
        
        # Create merged entity
        combined_text = text[group_start:group_end]
        # Only merge if it looks like a date pattern
        if j > i + 1 and DATE_RE.search(combined_text):
            merged.append({
                "type": "DATE",  # Will be re-classified by context detection
                "start": group_start,
                "end": group_end,
                "text": combined_text
            })
            i = j
        else:
            merged.append(curr.copy())
            i += 1
    
    return merged


def _combine_adjacent_name_entities(entities: list, text: str) -> list:
    """
    Combine adjacent FIRST_NAME/LAST_NAME entities into single NAME entities.
    This ensures that split names like "Sarah" + "Johnson" get the same cache key
    as full names like "Dr. Sarah Elizabeth Johnson, MD".
    """
    name_types = {"FIRST_NAME", "LAST_NAME", "NAME"}
    
    # Sort by position
    sorted_ents = sorted(entities, key=lambda x: x["start"])
    
    combined = []
    i = 0
    while i < len(sorted_ents):
        curr = sorted_ents[i]
        
        if curr["type"] not in name_types:
            combined.append(curr.copy())
            i += 1
            continue
        
        # Try to combine with following name entities
        group_start = curr["start"]
        group_end = curr["end"]
        j = i + 1
        
        while j < len(sorted_ents):
            next_ent = sorted_ents[j]
            if next_ent["type"] not in name_types:
                break
            
            gap = next_ent["start"] - group_end
            # Allow gap of up to 15 chars for ", MD" or middle initials etc.
            if gap > 15:
                break
            
            # Check gap only contains expected chars
            gap_text = text[group_end:next_ent["start"]]
            # Allow spaces, punctuation, and single uppercase letters (initials)
            if gap_text and not re.match(r'^[\s.,\-\']+[A-Z]?\.?[\s.,\-\']*$|^[\s.,\-\']*$', gap_text):
                break
            
            group_end = next_ent["end"]
            j += 1
        
        if j > i + 1:
            # Combined multiple entities into one NAME
            combined_text = text[group_start:group_end]
            combined.append({
                "type": "NAME",
                "start": group_start,
                "end": group_end,
                "text": combined_text
            })
            i = j
        else:
            # Single name entity - keep as-is but will be handled by normalize_name
            combined.append(curr.copy())
            i += 1
    
    return combined


class ClinicalDeidentifier:
    def __init__(self, seed: Optional[int] = None):
        self.fake = Faker('en_US')
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        
        # Cache for within-document consistency
        self._cache = {}
        self._date_shift = None
        self._current_location = None  # For geographic consistency
        # Track generated full names for first/last consistency
        self._full_name_cache = {}
        # Per-token name caches: the same given/surname token always maps to the
        # same fake, so "Veronica Brown", "Veronica", and "Brown" stay consistent.
        self._first_tokens = {}
        self._last_tokens = {}

    def reset_cache(self):
        """Call between documents to reset consistency caches."""
        self._cache = {}
        self._date_shift = random.randint(-365, -30)  # Shift 1-12 months back
        self._current_location = None
        self._full_name_cache = {}
        self._first_tokens = {}
        self._last_tokens = {}

    def _get_cached(self, original: str, entity_type: str, generator_fn):
        """Ensure consistent replacement within a document."""
        # Use normalized key for names so variants map to same fake
        if entity_type in ("FIRST_NAME", "LAST_NAME", "NAME"):
            norm = normalize_name(original)
            key = f"NAME:{norm}"
        else:
            key = f"{entity_type}:{original.strip()}"
        
        if key not in self._cache:
            self._cache[key] = generator_fn()
        return self._cache[key]
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Match the case pattern of the original text."""
        if original.isupper():
            return replacement.upper()
        elif original.islower():
            return replacement.lower()
        elif original.istitle():
            return replacement.title()
        else:
            return replacement
    
    _NAME_TITLES = frozenset({
        "dr", "mr", "mrs", "ms", "miss", "prof", "md", "do", "rn", "np", "pa",
        "phd", "jr", "sr", "ii", "iii", "iv",
    })

    def _fake_first(self, token: str) -> str:
        # Key by the FIRST real name word so "Michael" and "Michael A" collapse the
        # same; titles are dropped so "Dr Michael" -> "michael".
        words = [w for w in re.findall(r"[a-z]+", token.lower()) if w not in self._NAME_TITLES]
        t = words[0] if words else ""
        if not t:
            return token
        if t not in self._first_tokens:
            ov = getattr(self, "_name_overrides", {}).get("first", {}).get(t)
            self._first_tokens[t] = ov or self.fake.first_name()
        return self._first_tokens[t]

    def _fake_last(self, token: str) -> str:
        # Key by the LAST real name word so "A. Grant", "Grant", "Mr. Grant" all map
        # to the same surname surrogate (was: stripped to "agrant" vs "grant").
        words = [w for w in re.findall(r"[a-z]+", token.lower()) if w not in self._NAME_TITLES]
        t = words[-1] if words else ""
        if not t:
            return token
        if t not in self._last_tokens:
            ov = getattr(self, "_name_overrides", {}).get("last", {}).get(t)
            self._last_tokens[t] = ov or self.fake.last_name()
        return self._last_tokens[t]

    def _fake_name(self, text: str) -> str:
        """Compose a full-name surrogate from per-token caches, so a full name and
        its standalone first/last variants resolve to the SAME fake tokens."""
        # LLM-resolved override (coref-clustered, sex-matched) takes precedence.
        full_ov = getattr(self, "_name_overrides", {}).get("full", {})
        if full_ov:
            key = normalize_name(text)
            if key in full_ov:
                return full_ov[key]
        cleaned = re.sub(r"[^a-zA-Z,\s]", " ", text)
        if "," in cleaned:  # "Last, First" form
            last_part, _, first_part = cleaned.partition(",")
            lasts = [p for p in last_part.split() if p.lower() not in self._NAME_TITLES]
            firsts = [p for p in first_part.split() if p.lower() not in self._NAME_TITLES]
            first = firsts[0] if firsts else ""
            last = lasts[-1] if lasts else ""
        else:
            parts = [p for p in cleaned.split() if p.lower() not in self._NAME_TITLES]
            if not parts:
                return text
            if len(parts) == 1:  # single token (e.g. "Mr. Watkins") -> surname
                return self._fake_last(parts[0])
            first, last = parts[0], parts[-1]
        out = []
        if first:
            out.append(self._fake_first(first))
        if last:
            out.append(self._fake_last(last))
        return " ".join(out) if out else self.fake.name()

    def replace(self, text: str, entity_type: str) -> str:
        """Replace detected PHI with realistic surrogate data."""

        # === NAMES === (token-level caches keep first/last/full variants consistent)
        if entity_type == "FIRST_NAME":
            return self._preserve_case(text, self._fake_first(text))

        elif entity_type == "LAST_NAME":
            return self._preserve_case(text, self._fake_last(text))

        elif entity_type == "NAME":
            return self._preserve_case(text, self._fake_name(text))
        
        # === DATES ===
        elif entity_type == "DATE":
            return self._shift_date(text)
        
        elif entity_type == "DATE_OF_BIRTH":
            return self._generate_dob(text)
        
        elif entity_type == "DATE_TIME":
            return self._shift_datetime(text)
        
        elif entity_type == "TIME":
            # Time alone is not PHI per HIPAA Safe Harbor
            return text
        
        # === AGE ===
        elif entity_type == "AGE":
            return self._generalize_age(text)
        
        # === IDENTIFIERS ===
        elif entity_type == "SSN":
            return self._generate_ssn(text)
        
        elif entity_type == "MEDICAL_RECORD_NUMBER":
            return self._generate_mrn(text)
        
        elif entity_type == "HEALTH_PLAN_BENEFICIARY_NUMBER":
            return self._generate_id(text, prefix="HPBN")
        
        elif entity_type == "ACCOUNT_NUMBER":
            return self._generate_id(text, prefix="ACCT")
        
        elif entity_type == "CUSTOMER_ID":
            return self._generate_id(text, prefix="CID")
        
        elif entity_type == "EMPLOYEE_ID":
            return self._generate_id(text, prefix="EMP")
        
        elif entity_type == "UNIQUE_ID":
            return self._generate_id(text, prefix="UID")
        
        elif entity_type == "CERTIFICATE_LICENSE_NUMBER":
            return self._generate_license(text)
        
        elif entity_type == "BIOMETRIC_IDENTIFIER":
            return self._generate_id(text, prefix="BIO")
        
        elif entity_type == "NPI":
            # National Provider Identifier (unique clinician ID)
            return self._generate_npi(text)
        
        elif entity_type in ("GROUP_NUMBER", "INSURANCE_GROUP_NUMBER", "GROUP_ID"):
            # Insurance group number / plan group id
            return self._generate_id(text, prefix="GRP")
        
        # === CONTACT INFO ===
        elif entity_type == "PHONE_NUMBER":
            return self._generate_phone(text)
        
        elif entity_type == "FAX_NUMBER":
            return self._generate_phone(text)  # Same format as phone
        
        elif entity_type == "EMAIL":
            return self._get_cached(text, entity_type, self._generate_email)
        
        # === ADDRESSES ===
        elif entity_type == "STREET_ADDRESS":
            return self._get_cached(text, entity_type, self._generate_street)
        
        elif entity_type == "CITY":
            return self._get_cached(text, entity_type, self._generate_city)
        
        elif entity_type == "COUNTY":
            return self._get_cached(text, entity_type, lambda: self.fake.city() + " County")
        
        elif entity_type == "STATE":
            # State alone is generally allowed per HIPAA, but replace if detected
            return self._get_location()[1]  # Get consistent state
        
        elif entity_type == "POSTCODE":
            return self._generalize_zip(text)
        
        elif entity_type == "COUNTRY":
            # Country is allowed per HIPAA
            return text
        
        # === FALLBACK ===
        else:
            # Unknown type - generate generic ID
            return self._generate_id(text, prefix="ID")
    
    # === DATE HANDLING ===
    
    def _shift_date(self, text: str) -> str:
        """Shift a date by the per-note offset under a COMPONENTS-MIRRORING output contract
        (adapted from Philter V1.0, JAMIA Open 2023): emit ONLY the components present in the
        input, in the input's format family, never manufacturing a component the input lacked
        (bare year in -> bare year out; MM/DD in -> MM/DD out). The surrogate is length-
        disciplined (within 2 characters of the input for numeric output) and round-trip valid
        (any month/day output re-parses as a real calendar date).

        Bare-year semantics (decided): a bare year is anchored mid-year (Jul 1) and shifted by
        the same day offset, then emitted year-only. If the shift lands in the same year, that
        same year IS the offset's answer (a year has no finer granularity to move) and a bare
        year is Safe Harbor-permissible, so 2009 in / 2009 out is legal output; it is the shift
        result, not a formatting accident.

        No-structure fallback: input with no parseable date is NOT manufactured into a date
        (Branch 1 owns dropping it at the span layer); it degrades to a same-length, digit-
        preserving placeholder, so a stray fragment can never fuse a 10-char date into a
        neighbor. Input is never passed through raw."""
        if self._date_shift is None:
            self._date_shift = random.randint(-365, -30)
        text = text.strip()

        core = self._shift_date_core(text)
        if core is not None:
            return core
        # Multi-token or label/time-fused span (a TIME-fold "MM/DD/YYYY  4:13 PM", a merged
        # "2019\t10/20/24", a label-fused "Time: 6/30/2026 12:12 AM"): shift each embedded
        # date/year token in place and MIRROR everything else (labels, times, tabs, fused
        # suffixes). Times are not PHI; a fused non-date suffix is Branch 1's span-trim
        # territory and is preserved, never garbled into a malformed date.
        emb = self._shift_embedded(text)
        if emb is not None:
            return emb
        # No date structure at all: same-length, digit-preserving placeholder (never a date).
        return self._disciplined_fallback(text)

    _EMBEDDED_DATE = re.compile(
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|\b(?:19|20)\d{2}\b")

    def _shift_embedded(self, text: str) -> Optional[str]:
        """Shift each embedded date-shaped or bare-year token, mirroring all other characters.
        Returns the rewritten span, or None if it held no such token (a genuine fragment for the
        disciplined fallback). Each token goes through _shift_date_core, so components-mirroring,
        length discipline, and round-trip validity hold per token."""
        def repl(m):
            shifted = self._shift_date_core(m.group(0))
            return shifted if shifted is not None else m.group(0)
        out = self._EMBEDDED_DATE.sub(repl, text)
        return out if out != text else None

    def _shift_date_core(self, text: str) -> Optional[str]:
        """Parse+shift one date under the contract. Returns the shifted string, or None when
        the text has no parseable date structure (caller applies the disciplined fallback)."""
        parsed = self._parse_date(text)
        if not parsed:
            return None
        dt, fmt = parsed
        out = (dt + timedelta(days=self._date_shift)).strftime(fmt)
        # Round-trip validity: the emitted value must re-parse under its OWN format as a real
        # calendar date, so a malformed value (2003/02/2043-class) can never be returned.
        try:
            datetime.strptime(out, fmt)
        except ValueError:
            return None
        # Length discipline (numeric family only; written month names vary legitimately by
        # name length and are governed by components-mirroring + round-trip instead).
        if not any(c.isalpha() for c in out) and abs(len(out) - len(text)) > 2:
            print(f"[deid] date-shift length-discipline fallback: {text!r} -> {out!r}")
            return None
        return out

    def _disciplined_fallback(self, text: str) -> str:
        """No parseable date structure -> never manufacture a date. Emit a same-length,
        digit-preserving placeholder: each digit is shifted deterministically by the per-note
        offset, other characters are preserved except alphabetics (redacted to 'x', since alpha
        in an unparseable 'date' span is not a date). Length is preserved, so a fragment can
        never fuse a long surrogate into its neighbor, and input never passes through raw."""
        k = (self._date_shift or 0) % 10
        return "".join(
            str((int(c) + k) % 10) if c.isdigit() else ("x" if c.isalpha() else c)
            for c in text
        )
    
    def _generate_dob(self, text: str) -> str:
        """Generate realistic DOB preserving approximate age (±2 years)."""
        parsed = self._parse_date(text)
        if parsed:
            dt, fmt = parsed
            # Small year jitter (±2 years) to preserve approximate age
            year_jitter = random.randint(-2, 2)
            # Random month and day
            new_month = random.randint(1, 12)
            new_day = random.randint(1, 28)  # Safe for all months
            
            try:
                shifted = dt.replace(year=dt.year + year_jitter, month=new_month, day=new_day)
                return shifted.strftime(fmt)
            except ValueError:
                # Handle edge cases
                shifted = dt.replace(year=dt.year + year_jitter, month=new_month, day=15)
                return shifted.strftime(fmt)
        
        # Couldn't parse - generate realistic adult DOB
        return self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%m/%d/%Y")
    
    def _shift_datetime(self, text: str) -> str:
        """Shift datetime, preserving format including time."""
        if self._date_shift is None:
            self._date_shift = random.randint(-365, -30)
        
        text = text.strip()
        
        # Common datetime formats
        datetime_formats = [
            ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"),
            ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M"),
            ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"),
            ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M"),
            ("%m/%d/%y %H:%M", "%m/%d/%y %H:%M"),
            ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S"),
            ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%SZ"),
        ]
        
        for parse_fmt, out_fmt in datetime_formats:
            try:
                dt = datetime.strptime(text, parse_fmt)
                shifted = dt + timedelta(days=self._date_shift)
                return shifted.strftime(out_fmt)
            except ValueError:
                continue
        
        # Try date-only parsing
        return self._shift_date(text)
    
    def _parse_date(self, text: str) -> Optional[Tuple[datetime, str]]:
        """Parse date and return (datetime, format_string)."""
        # Order matters - try more specific formats first
        formats = [
            # ISO formats
            ("%Y-%m-%d", "%Y-%m-%d"),
            # US formats
            ("%m/%d/%Y", "%m/%d/%Y"),
            ("%m-%d-%Y", "%m-%d-%Y"),
            ("%m/%d/%y", "%m/%d/%y"),
            ("%m-%d-%y", "%m-%d-%y"),
            # Written formats
            ("%B %d, %Y", "%B %d, %Y"),
            ("%b %d, %Y", "%b %d, %Y"),
            ("%d %B %Y", "%d %B %Y"),
            ("%d %b %Y", "%d %b %Y"),
            # European formats
            ("%d/%m/%Y", "%d/%m/%Y"),
            ("%d-%m-%Y", "%d-%m-%Y"),
            # Other
            ("%Y%m%d", "%Y%m%d"),
        ]
        
        for parse_fmt, out_fmt in formats:
            try:
                dt = datetime.strptime(text.strip(), parse_fmt)
                # Sanity check - year should be reasonable
                if 1900 <= dt.year <= 2100:
                    return (dt, out_fmt)
            except ValueError:
                continue
        
        # Try regex for partial dates like "12/07/25" (with year)
        match = re.match(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
        if match:
            m, d, y = match.groups()
            try:
                if len(y) == 2:
                    y = "20" + y if int(y) < 50 else "19" + y
                dt = datetime(int(y), int(m), int(d))
                # Preserve original format
                sep = "/" if "/" in text else "-"
                if len(match.group(3)) == 2:
                    fmt = f"%m{sep}%d{sep}%y"
                else:
                    fmt = f"%m{sep}%d{sep}%Y"
                return (dt, fmt)
            except ValueError:
                pass
        
        # Try regex for SHORT dates like "12/13" or "12/15" (MM/DD without year)
        # Assume current year for these clinical encounter dates
        short_match = re.match(r'^(\d{1,2})[/\-](\d{1,2})$', text.strip())
        if short_match:
            m, d = short_match.groups()
            try:
                current_year = datetime.now().year
                dt = datetime(current_year, int(m), int(d))
                sep = "/" if "/" in text else "-"
                # Use short format (no year) to preserve original appearance
                fmt = f"%m{sep}%d"
                return (dt, fmt)
            except ValueError:
                pass

        # Bare 4-digit YEAR: mirror as a year (components-mirroring, year in / year out).
        # Anchor mid-year (Jul 1) so the day offset yields the shifted year without an
        # end-of-year formatting bias. Emitting "%Y" keeps output year-only.
        year_match = re.match(r'^(\d{4})$', text.strip())
        if year_match:
            y = int(year_match.group(1))
            if 1900 <= y <= 2100:
                return (datetime(y, 7, 1), "%Y")

        # Bare 1-3 digit fragment: no date structure. Return None so _shift_date's disciplined
        # fallback handles it, instead of the old dateutil-coerces-a-fragment-into-a-full-date
        # bug ("22" -> 06/14/2026, "43" -> 05/28/2043 fusing to 2003/02/2043).
        if re.fullmatch(r'\d{1,3}', text.strip()):
            return None

        # Robust fallback: dateutil handles structured forms the explicit list misses
        # (2024/11/05, 11.5.2024, "Nov 5 2024", "5 Nov 2024"). GUARDED to inputs carrying date
        # structure (a separator or an alphabetic month) so a bare number is never coerced.
        if re.search(r'[A-Za-z]', text) or re.search(r'[/.\-]', text):
            try:
                from dateutil import parser as _du
                dt = _du.parse(text.strip())
                if 1900 <= dt.year <= 2100:
                    return (dt, "%m/%d/%Y")
            except Exception:
                pass

        return None
    
    # === IDENTIFIER GENERATION ===
    
    def _generate_ssn(self, original: str) -> str:
        """Generate fake SSN preserving format."""
        # Detect format
        if re.match(r'\d{3}-\d{2}-\d{4}', original):
            return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
        elif re.match(r'\d{9}', original):
            return f"{random.randint(100000000, 999999999)}"
        else:
            # Default formatted
            return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"
    
    def _generate_mrn(self, original: str) -> str:
        """Generate fake MRN matching original format/length."""
        original = original.strip()
        
        # Extract any prefix (letters at start)
        prefix_match = re.match(r'^([A-Za-z]+)[-_]?', original)
        prefix = prefix_match.group(1) if prefix_match else "MRN"
        
        # Count digits in original
        digits = re.findall(r'\d', original)
        num_digits = len(digits) if digits else 8
        
        # Detect separator
        sep = "-" if "-" in original else ""
        
        # Generate new number with same digit count
        new_num = ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])
        
        return f"{prefix}{sep}{new_num}"
    
    def _generate_id(self, original: str, prefix: str = "ID") -> str:
        """Generate fake ID preserving approximate length."""
        original = original.strip()
        
        # Count alphanumeric characters
        alphanum = re.findall(r'[A-Za-z0-9]', original)
        length = len(alphanum) if alphanum else 8
        
        # Generate random alphanumeric of similar length
        chars = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=max(4, length)))
        
        return chars
    
    def _generate_npi(self, original: str) -> str:
        """Generate fake NPI (National Provider Identifier)."""
        original = original.strip()
        
        # NPI is always 10 digits
        # But preserve format if there's a prefix
        prefix_match = re.match(r'^([A-Za-z]+)[-_:\s]*', original)
        if prefix_match:
            prefix = prefix_match.group(1)
            return f"{prefix}{random.randint(1000000000, 9999999999)}"
        
        # Standard 10-digit NPI
        return str(random.randint(1000000000, 9999999999))
    
    def _generate_license(self, original: str) -> str:
        """Generate fake license/certificate number."""
        original = original.strip()
        
        # Try to match format (letters + numbers pattern)
        letters = len(re.findall(r'[A-Za-z]', original))
        numbers = len(re.findall(r'\d', original))
        
        if letters > 0 and numbers > 0:
            letter_part = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=letters))
            number_part = ''.join(random.choices('0123456789', k=numbers))
            return f"{letter_part}{number_part}"
        elif numbers > 0:
            return ''.join(random.choices('0123456789', k=max(6, numbers)))
        else:
            return self.fake.bothify(text="??######")
    
    # === CONTACT GENERATION ===
    
    def _generate_phone(self, original: str) -> str:
        """Generate fake phone number preserving format."""
        original = original.strip()
        
        # Generate base phone number
        area = random.randint(200, 999)
        exchange = random.randint(200, 999)
        subscriber = random.randint(1000, 9999)
        
        # Detect format from original
        if re.match(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', original):
            # (123) 456-7890
            sep = "-" if "-" in original else ("." if "." in original else "-")
            space = " " if " " in original[5:8] else ""
            return f"({area}){space}{exchange}{sep}{subscriber}"
        
        elif re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', original):
            # 123-456-7890 or 123.456.7890
            sep = "-" if "-" in original else ("." if "." in original else "-")
            return f"{area}{sep}{exchange}{sep}{subscriber}"
        
        elif re.match(r'\d{10}', original):
            # 1234567890
            return f"{area}{exchange}{subscriber}"
        
        elif re.match(r'\+?1?\s*\(?\d{3}\)?', original):
            # +1 (123) 456-7890 or similar
            return f"+1 ({area}) {exchange}-{subscriber}"
        
        else:
            # Default format
            return f"({area}) {exchange}-{subscriber}"
    
    def _generate_email(self) -> str:
        """Generate realistic email."""
        return self.fake.email()
    
    # === ADDRESS GENERATION ===
    
    def _get_location(self) -> Tuple[str, str, str]:
        """Get consistent city, state, zip for this document."""
        if self._current_location is None:
            # Generate a consistent location
            city = self.fake.city()
            state = self.fake.state_abbr()
            zipcode = self.fake.zipcode()
            self._current_location = (city, state, zipcode)
        return self._current_location
    
    def _generate_street(self) -> str:
        """Generate realistic street address."""
        return self.fake.street_address()
    
    def _generate_city(self) -> str:
        """Generate city (consistent within document)."""
        return self._get_location()[0]
    
    def _generalize_zip(self, original: str) -> str:
        """
        Per HIPAA Safe Harbor: 
        - Keep first 3 digits if population >20,000
        - Otherwise generalize to 000XX
        In practice, we generate a fake zip with same format.
        """
        original = original.strip()
        
        # Generate fake zip
        fake_zip = self.fake.zipcode()
        
        # Match format
        if re.match(r'\d{5}-\d{4}', original):
            # ZIP+4 format
            return f"{fake_zip[:5]}-{random.randint(1000, 9999)}"
        elif re.match(r'\d{5}', original):
            return fake_zip[:5]
        elif re.match(r'\d{3}', original):
            # Just first 3 digits
            return fake_zip[:3]
        else:
            return fake_zip[:5]
    
    # === AGE HANDLING ===
    
    def _generalize_age(self, original: str) -> str:
        """
        HIPAA Safe Harbor requires ages 90+ to be generalized.
        We generalize 89+ to be safe.
        """
        try:
            # Extract numeric age
            match = re.search(r'\d+', original)
            if match:
                age = int(match.group())
                if age >= 89:
                    # Replace the number with "90+"
                    return re.sub(r'\d+', '90+', original)
            return original
        except:
            return original


# === Utility for batch processing ===

def _split_name_spans(entities: list, text: str) -> list:
    """Split a NAME/FIRST/LAST span that swallowed a sentence boundary ("Lee. Whitehead"
    -> two names) and tighten each piece to its alpha edges, so the period and the second
    person's surrogate stay distinct (and "Lee" keys consistently with other "Lee"s).
    A single letter before the period (a middle initial like "A. Grant") is NOT a
    boundary, so initials are preserved."""
    out = []
    for ent in entities:
        if ent.get("type") not in ("NAME", "FIRST_NAME", "LAST_NAME"):
            out.append(ent)
            continue
        s = ent["start"]
        raw = text[s:ent["end"]]
        cuts = [0]
        for m in re.finditer(r"(?<=[A-Za-z]{2})\.\s+", raw):
            cuts.append(m.end())
        cuts.append(len(raw))
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            la, rb = a, b
            while la < rb and not raw[la].isalpha():
                la += 1
            while rb > la and not raw[rb - 1].isalpha():
                rb -= 1
            if rb > la:
                ne = dict(ent)
                ne["start"], ne["end"], ne["text"] = s + la, s + rb, text[s + la:s + rb]
                pieces.append(ne)
        out.extend(pieces if pieces else [ent])
    return out


# Identifier families whose canonical (and surrogate) form is ALWAYS alphanumeric at both
# edges, so a non-alphanumeric character at a detected span edge is NER over-match and never
# part of the identifier. Enumerated from labels.ENTITY_TYPES: the direct-identifier number
# types (SSN / MEDICAL_RECORD_NUMBER / HEALTH_PLAN_BENEFICIARY_NUMBER) plus the "Other IDs"
# block. PHONE_NUMBER / FAX_NUMBER are DELIBERATELY EXCLUDED — their surrogate legitimately
# begins with "(", so trimming a leading paren would corrupt them. NAME / DATE / AGE / geo
# families are out of scope (NAME edges are handled by _split_name_spans).
_IDENTIFIER_TYPES = frozenset({
    "SSN",
    "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUMBER",
    "ACCOUNT_NUMBER",
    "CUSTOMER_ID",
    "EMPLOYEE_ID",
    "UNIQUE_ID",
    "BIOMETRIC_IDENTIFIER",
    "CERTIFICATE_LICENSE_NUMBER",
})


def _trim_identifier_span_edges(entities: list, raw_text: str) -> list:
    """Tighten identifier-family spans (MRN/SSN/account/...) to their alphanumeric edges.
    The NER occasionally over-matches an identifier span's LEFT (or right) boundary into an
    adjacent unit/punctuation char (e.g. "% MRN 12594309"); because substitution is a faithful
    offset slice with no delimiter guarantee, that swallowed char is consumed and the surrogate
    fuses into neighboring text ("EF 60-69MRN12594309 ..."). Advancing start past leading
    non-alphanumerics and retreating end past trailing non-alphanumerics leaves only the
    identifier itself. EDGES ONLY — interior characters (SSN hyphens, MRN prefixes) are
    untouched. A span that trims to empty held no identifier and is DROPPED, so we never
    substitute an empty span. NAME/DATE families are out of scope (NAME: _split_name_spans)."""
    out = []
    for ent in entities:
        if ent.get("type") not in _IDENTIFIER_TYPES:
            out.append(ent)
            continue
        s, e = ent["start"], ent["end"]
        while s < e and not raw_text[s].isalnum():
            s += 1
        while e > s and not raw_text[e - 1].isalnum():
            e -= 1
        if s >= e:
            # Degenerate: the detected span was all non-alphanumeric — no identifier to
            # redact. Drop it rather than emit a garbage surrogate. (Matches the pipeline's
            # print() diagnostic idiom; other drops — subword fragments, whitelist — are silent.)
            print(f"[deid] dropped degenerate {ent.get('type')} span "
                  f"{ent['start']}:{ent['end']} {raw_text[ent['start']:ent['end']]!r} "
                  f"(no alphanumeric content)")
            continue
        if (s, e) != (ent["start"], ent["end"]):
            ne = dict(ent)
            ne["start"], ne["end"], ne["text"] = s, e, raw_text[s:e]
            out.append(ne)
        else:
            out.append(ent)
    return out


# Maximal complete date token, value-bounded per the Philter idiom (planning/recognizers/
# structured_recognizers_seed.py DATE_MAXIMAL_TOKEN, extended to cover every shape the Branch 3
# _parse_date accepts: the numeric slash/dash forms, ISO, YYYYMMDD, written month-name forms,
# MM/DD short, bare year, and an optional folded time). Never lazy digit counts. Ordered so the
# longest/most-specific form wins at a position (full date before MM/DD short before bare year).
_DT_MON = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
           r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DT_MM = r"(?:1[0-2]|0?[1-9])"          # 10-12 before 1-9 (ordered alternation is first-match)
_DT_DD = r"(?:3[01]|[12]\d|0?[1-9])"    # 30-31, 10-29, then 1-9, so "15" is not cut to "1"
_DT_YYYY = r"(?:19|20)\d{2}"
_DT_TIME = r"(?:[ \t]+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap]\.?[Mm]\.?)?)?"
_DATE_TOKEN = re.compile(
    r"(?:"
    + _DT_YYYY + r"-" + _DT_MM + r"-" + _DT_DD                                   # ISO YYYY-MM-DD
    + r"|" + _DT_MM + r"[/-]" + _DT_DD + r"[/-]\d{2,4}"                          # MM/DD/YYYY or /YY
    + r"|" + _DT_MON + r"\.?\s+" + _DT_DD + r"(?:st|nd|rd|th)?,?\s+" + _DT_YYYY  # Month D, YYYY
    + r"|" + _DT_DD + r"(?:st|nd|rd|th)?\s+" + _DT_MON + r"\.?\s+" + _DT_YYYY    # D Month YYYY
    + r"|" + _DT_MM + r"[/-]" + _DT_DD                                          # MM/DD short
    + r"|" + _DT_YYYY + _DT_MM + _DT_DD                                         # YYYYMMDD
    + r"|" + _DT_YYYY                                                           # bare year
    + r")" + _DT_TIME
)

_DATE_FAMILY = ("DATE", "DATE_OF_BIRTH", "DATE_TIME")


def _date_span_boundary(entities: list, text: str) -> list:
    """DATE-family span boundary guard (span layer). Two rules:

    (1) No-digit DROP: a DATE-family span with no digit ("Time", "QDAY", "TBD", "Saturday") is
        not a date; drop it so the word survives verbatim instead of being mangled to "xxxx" by
        the substitution-layer disciplined fallback. (Digit-bearing garbage still flows to that
        fallback; this rule owns only the no-digit case.)
    (2) Maximal-token TRIM: a span containing exactly one complete date token (with any leading
        non-token garbage or trailing non-date text, e.g. a header "11/13/2025\\nCre") is trimmed
        to that token. A multi-cell span (two or more date tokens, e.g. "2019\\t10/20/24") passes
        through untrimmed so Branch 3's embedded shift handles each token; a legitimate written /
        ISO / bare-year / MM-DD / folded date+time span is one token spanning the whole span and
        is never split (the recall regression named in the foundation report).

    Runs after _merge_adjacent_date_entities (tightens merged spans, not fragments) and before
    _split_name_spans / _trim_identifier_span_edges. The later DOB lookback retype (substitution
    loop) is unaffected: it keys on text before the span start, and a DOB date is a clean single
    token with no leading garbage, so its start does not move."""
    out = []
    for ent in entities:
        if ent.get("type") not in _DATE_FAMILY:
            out.append(ent)
            continue
        s, e = ent["start"], ent["end"]
        span = text[s:e]
        if not any(c.isdigit() for c in span):
            print(f"[deid] date-span drop (no digit) {ent.get('type')} {s}:{e} {span!r}")
            continue
        toks = list(_DATE_TOKEN.finditer(span))
        if len(toks) == 1:
            ts, te = toks[0].start(), toks[0].end()
            if (ts, te) != (0, len(span)):
                ne = dict(ent)
                ne["start"], ne["end"], ne["text"] = s + ts, s + te, text[s + ts:s + te]
                print(f"[deid] date-span trim {ent.get('type')} {s}:{e}->{ne['start']}:{ne['end']} "
                      f"{span!r}->{ne['text']!r}")
                out.append(ne)
                continue
        out.append(ent)  # clean single token, multi-cell, or no recognized token: pass through
    return out


def _build_surrogate_map(subs: list) -> dict:
    """Assemble the original<->surrogate pairs that were ACTUALLY substituted, for
    output-side re-identification. This is additive observation only — it reads values
    the scrubber already produced and never affects the de-identified text.

    Each pair carries a `collision` flag: True when the SURROGATE token matches a
    medical/common whitelist word (computed against the surrogate, never the original),
    so the downstream re-id pass can refuse to swap a fake name that doubles as a real
    clinical word. Token-level name pairs are emitted too ("Anjali Patel"/"Megan Riley"
    yields "Patel"/"Riley") so a lone surname paraphrased downstream stays recoverable.
    """
    pairs = {}

    def add(orig: str, surr: str, etype: str):
        orig, surr = (orig or "").strip(), (surr or "").strip()
        if not orig or not surr or orig == surr:
            return
        collision = any(tok in MEDICAL_WHITELIST_LOWER for tok in surr.lower().split())
        pairs[(orig.lower(), surr.lower(), etype)] = {
            "original": orig, "surrogate": surr, "type": etype, "collision": collision,
        }

    for orig, surr, etype in subs:
        add(orig, surr, etype)
        if etype in ("NAME", "FIRST_NAME", "LAST_NAME"):
            o_tok, s_tok = orig.split(), surr.split()
            # Only split into token pairs when counts align, so tokens can't be mis-paired.
            if len(o_tok) == len(s_tok) and len(o_tok) > 1:
                for i, (ot, st) in enumerate(zip(o_tok, s_tok)):
                    add(ot, st, "FIRST_NAME" if i == 0 else "LAST_NAME")

    return {"version": 1, "pairs": list(pairs.values())}


# US state / DC abbreviations. A bare 5-digit ZIP is swept ONLY when one of these appears
# immediately before it; an unconditioned \d{5} would destroy account numbers, lab accession
# IDs, and vitals rows. The pilot's real address blocks are all Oklahoma ("SHAWNEE OK 74804"),
# so OK must stay in the set even though OK/IN/OR double as English words (shapes B/C/D also
# require a ZIP-shaped token, which bounds that collision).
_US_STATE_ABBR = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
    "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()
_STATE_ALT = "|".join(_US_STATE_ABBR)

# leak shape A: full ZIP+4 surviving verbatim (pilot notes 1/8/12/19: "OK 74804-9638").
# Left guard forbids a letter/digit/hyphen before (so a run start is clean); right guard
# forbids a longer digit run. Fires UNCONDITIONALLY: a 5-4 hyphenation is ZIP-distinctive
# (phones are 3-3-4, SSNs 3-2-4, dates use slashes), so it cannot match those shapes.
_RE_ZIP4 = re.compile(r"(?<![A-Za-z\d-])\d{5}-\d{4}(?![-\d])")

# leak shape C: truncated/fused ZIP (pilot note 13 / site 8: "74804-96MRN68", or "74804-96"
# at line end). 5 digits, hyphen, 1-3 digits that do NOT complete a 4-digit +4 (that is shape
# A). Consumes ZIP material ONLY; any fused surrogate letters ("MRN68") sit OUTSIDE the match
# and are left for the fusion branches. Composes with Branch 4 (digit-suffix leftward span
# expansion, not yet cut): whichever runs, the ZIP material dies. Do NOT weaken either pass on
# the assumption the other will catch it.
_RE_ZIP_TRUNC = re.compile(r"(?<![A-Za-z\d-])\d{5}-\d{1,3}(?![-\d])")

# leak shape D: prefix-fused ZIP body behind a letter run (pilot note 2: "MRN63084-9167",
# where the ZIP's LEADING digit was mis-tagged and surrogated to "MRN.."). The ZIP-shaped tail
# \d{4,5}-\d{4} sits directly after letters; a state/address-supported line gates it (below) so
# it does not catch letter-prefixed identifiers.
_RE_ZIP_PREFIX = re.compile(r"(?<=[A-Za-z])\d{4,5}-\d{4}(?![-\d])")

# leak shape B: state-adjacent bare 5-digit ZIP (pilot notes 10/17: "OK 73115"). The state
# token is the gate; group(1) keeps "STATE ", group(2) is the bare ZIP passed to _generalize_zip.
# (?![-\d]) forbids a ZIP+4 (shape A) or a longer digit run (a 6+-digit id).
_RE_STATE_ZIP5 = re.compile(r"(\b(?:" + _STATE_ALT + r")\b\s+)(\d{5})(?![-\d])")

# a line "supports" the letter-adjacent shapes C and D only if it carries a state token
# (address-shaped); without this gate, \d{5}-\d{1,3} could match numeric ranges elsewhere.
_RE_STATE_TOKEN = re.compile(r"\b(?:" + _STATE_ALT + r")\b")


def _zip_residual_backstop(text: str, deid: "ClinicalDeidentifier", surrogate_map: dict = None) -> str:
    """Post-substitution TEXT sweep: re-generalize ZIP patterns that survived detection.

    A pilot-measured PHI leak class: undetected or partially-covered ZIPs passed verbatim into
    de-identified output (9 of 12 input ZIPs leaked full or partial ZIP material; the ZIP+4
    form went 0 for 7). Mirrors the patient-token backstop above: a text-level re.sub net below
    the substitution loop. All surrogate policy is delegated to deid._generalize_zip (which
    expects a BARE ZIP string), so the state prefix is split off here and reattached;
    _generalize_zip itself is not modified.

    surrogate_map (optional, return_map path only): the already-built re-id map. When present, a
    ZIP that is ALREADY a recorded POSTCODE surrogate is left untouched, so its map value stays
    locatable in the output for the CPT grader's re-identification overlay (README: return_map ->
    "restore the real entities in its output"). Raw undetected input ZIPs are never in the map, so
    their leak neutralization is never skipped; only the harmless re-generalization of an already
    clean, already-mapped surrogate is suppressed. None -> no protection, so the non-map call path
    is unchanged (a bare fake ZIP is still harmlessly re-generalized).

    Idempotence: on already-clean output with no map, the pass is a harmless RE-GENERALIZATION,
    not a strict no-op. A prior fake ZIP is itself ZIP-shaped and state-adjacent, so it is replaced
    by another valid fake ZIP of the same format (still de-identified). Never a corruption."""
    protected = set()
    if surrogate_map:
        for p in surrogate_map.get("pairs", []):
            if p.get("type") == "POSTCODE" and p.get("surrogate"):
                protected.add(p["surrogate"])

    def gen(zipstr):
        # Surrogate-map consistency guard: a ZIP that is already a recorded POSTCODE surrogate is
        # clean and the re-id map points at it; re-generalizing would strand that map value. Skip.
        # Raw leaked ZIPs are absent from the map, so they are still neutralized below.
        if zipstr in protected:
            return zipstr
        return deid._generalize_zip(zipstr)

    out = []
    for line in text.split("\n"):
        # A: full ZIP+4 (unconditional) -- replace the digit run, keep surrounding text.
        line = _RE_ZIP4.sub(lambda m: gen(m.group(0)), line)
        # B: state-adjacent bare 5-digit -- keep the "STATE " prefix, re-generalize the ZIP.
        line = _RE_STATE_ZIP5.sub(lambda m: m.group(1) + gen(m.group(2)), line)
        if _RE_STATE_TOKEN.search(line):
            # C: truncated/fused ZIP -- consume ZIP material only, leave fused letters.
            line = _RE_ZIP_TRUNC.sub(lambda m: gen(m.group(0)), line)
            # D: prefix-fused ZIP body behind letters -- eat the ZIP tail, leave the letters.
            line = _RE_ZIP_PREFIX.sub(lambda m: gen(m.group(0)), line)
        out.append(line)
    return "\n".join(out)


def deidentify_text(text: str, entities: list, seed: int = None, name_map: dict = None,
                    return_map: bool = False):
    """
    Replace entities in text with surrogate data.

    Args:
        text: Original clinical text
        entities: List of dicts with 'type', 'start', 'end', 'text' keys
        seed: Random seed for reproducibility
        return_map: When True, return (deidentified_text, surrogate_map) instead of
            just the text. The map is additive and does not change scrubbing.

    Returns:
        De-identified text, or (text, surrogate_map) when return_map=True.
    """
    deid = ClinicalDeidentifier(seed=seed)
    deid.reset_cache()
    # LLM-resolved name overrides (coref clusters + sex-matched surrogates), if provided.
    deid._name_overrides = name_map or {}

    # Pre-process: Merge fragmented DATE entities back together
    # Model sometimes splits "03/15/1965" into "03", "/", "15", etc.
    entities = _merge_adjacent_date_entities(entities, text)

    # DATE span boundary guard: trim a DATE-family span to its maximal date token (header
    # over-reach like "11/13/2025\nCre") and drop a no-digit DATE span ("Time"/"QDAY") so the
    # word survives verbatim. After the merge (so it tightens merged spans), before the name/
    # identifier edge passes.
    entities = _date_span_boundary(entities, text)

    # Pre-process: Combine adjacent FIRST_NAME/LAST_NAME into single NAME entities
    # This ensures "Sarah" + "Johnson" gets same cache key as "Sarah Elizabeth Johnson"
    entities = _combine_adjacent_name_entities(entities, text)

    # Split spans that swallowed a sentence boundary ("Lee. Whitehead") and tighten edges
    entities = _split_name_spans(entities, text)

    # Tighten identifier spans (MRN/SSN/account/...) to alphanumeric edges so an NER left/
    # right over-match into an adjacent unit char ("% MRN 12594309") can't fuse the surrogate
    # into neighboring text. Runs on every path reaching the reverse-slice loop below.
    entities = _trim_identifier_span_edges(entities, text)

    # Sort by start position (reverse) for safe replacement
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

    # (original, surrogate, type) actually substituted — feeds the optional re-id map.
    _subs = []
    result = text
    for entity in sorted_entities:
        entity_type = entity["type"]
        
        # Context-aware DOB detection: if DATE entity follows "DOB:" treat as DATE_OF_BIRTH
        # This preserves patient age since model doesn't distinguish DOB from other dates
        if entity_type == "DATE":
            # Look at 50 chars before entity for DOB context
            lookback_start = max(0, entity["start"] - 50)
            context = text[lookback_start:entity["start"]].lower()
            if "dob:" in context or "dob :" in context or "date of birth" in context or "birth date" in context or "birthdate" in context:
                entity_type = "DATE_OF_BIRTH"
        
        replacement = deid.replace(entity["text"], entity_type)
        _subs.append((entity["text"], replacement, entity_type))
        result = result[:entity["start"]] + replacement + result[entity["end"]:]
    
    # Patient-token backstop: name_map carries the resolved patient surrogate. The NER
    # sometimes DETECTS the patient name but the replacement still leaves a token (covered
    # by a DATE/DATE_TIME span, or a partial "Michael <mid> Grant" replacement). Sweep any
    # CAPITALIZED surviving occurrence of the patient's own first/last token so the M&M
    # subject's name cannot leak. Lowercase forms (common words) are left untouched.
    for _kind in ("first", "last"):
        for _orig, _surr in ((name_map or {}).get(_kind) or {}).items():
            if len(_orig) >= 2:
                _bk_type = "FIRST_NAME" if _kind == "first" else "LAST_NAME"

                def _bk(m, s=_surr, t=_bk_type):
                    if m.group(0).islower():
                        return m.group(0)
                    _subs.append((m.group(0), s, t))  # record the actual backstop swap
                    return s

                result = re.sub(r"\b" + re.escape(_orig) + r"\b", _bk, result, flags=re.I)

    # Safety net: if any date fragments still shifted independently and ran together,
    # drop digits trailing a complete date ("08/14/202554" -> "08/14/2025"). This only
    # ever matches an already-malformed value (a real date is not immediately followed
    # by bare digits).
    result = re.sub(r"(\b\d{1,2}/\d{1,2}/\d{4})\d+", r"\1", result)

    # Build the re-id map from the ACTUAL substitutions BEFORE the ZIP backstop (return_map path
    # only), so the backstop can protect already-recorded POSTCODE surrogates from harmless
    # re-generalization and keep every map value locatable in the output for the CPT grader's
    # re-identification overlay (README: return_map). The map reads from _subs, not the text, so
    # building it here vs at return is value-identical; None on the non-map path leaves that path
    # unchanged (raw leaked ZIPs, absent from the map, are still neutralized either way).
    surrogate_map = _build_surrogate_map(_subs) if return_map else None

    # ZIP residual backstop: re-generalize any ZIP pattern that survived detection (undetected
    # or partially-covered ZIPs leaked verbatim in the pilot). Text-level, below detection; all
    # surrogate policy stays in _generalize_zip. Sits after the trailing-date sanitizer and
    # before _clean_dob_lines -- the two are independent (DOB cleanup only rewrites ^DOB: lines
    # and ZIP surrogates carry no date shape), so this grouping keeps the residual text sweeps
    # together after substitution.
    result = _zip_residual_backstop(result, deid, surrogate_map=surrogate_map)

    # Post-process DOB line(s) to keep a single clean date
    result = _clean_dob_lines(result)

    if return_map:
        return result, surrogate_map
    return result
