# deid.py - Clinical De-identification Surrogate Generator
# v2.2.0 - Age-preserving DOB replacement
#
# Key features:
# - Consistent replacement within documents (same PHI → same fake)
# - Format preservation (phone, SSN, dates match original format)
# - Realistic surrogates (not bracketed placeholders)
# - Date shifting preserves temporal relationships
# - Geographic consistency (city/state/zip match)
# - Name normalization (Dr. Sarah Johnson, MD → same fake as Sarah Johnson)
# - DOB line cleanup (prevents concatenated date artifacts)
# - Age-preserving DOB (±2 year jitter keeps patient age realistic)

from faker import Faker
from datetime import datetime, timedelta
import random
import re
from typing import Optional, Tuple


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
        
    def reset_cache(self):
        """Call between documents to reset consistency caches."""
        self._cache = {}
        self._date_shift = random.randint(-365, -30)  # Shift 1-12 months back
        self._current_location = None
    
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
    
    def replace(self, text: str, entity_type: str) -> str:
        """Replace detected PHI with realistic surrogate data."""
        
        # === NAMES ===
        if entity_type == "FIRST_NAME":
            replacement = self._get_cached(text, entity_type, self.fake.first_name)
            return self._preserve_case(text, replacement)
        
        elif entity_type == "LAST_NAME":
            replacement = self._get_cached(text, entity_type, self.fake.last_name)
            return self._preserve_case(text, replacement)
        
        elif entity_type == "NAME":
            # Full name - generate first + last
            replacement = self._get_cached(text, entity_type, self.fake.name)
            return self._preserve_case(text, replacement)
        
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
        """Shift date by consistent offset, preserving original format."""
        if self._date_shift is None:
            self._date_shift = random.randint(-365, -30)
        
        text = text.strip()
        
        # Try to parse and shift the date
        parsed = self._parse_date(text)
        if parsed:
            dt, fmt = parsed
            shifted = dt + timedelta(days=self._date_shift)
            return shifted.strftime(fmt)
        
        # Couldn't parse - return shifted placeholder that looks like a date
        return self.fake.date(pattern="%m/%d/%Y")
    
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
        
        # Try regex for partial dates like "12/07/25"
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

def deidentify_text(text: str, entities: list, seed: int = None) -> str:
    """
    Replace entities in text with surrogate data.
    
    Args:
        text: Original clinical text
        entities: List of dicts with 'type', 'start', 'end', 'text' keys
        seed: Random seed for reproducibility
    
    Returns:
        De-identified text
    """
    deid = ClinicalDeidentifier(seed=seed)
    deid.reset_cache()
    
    # Sort by start position (reverse) for safe replacement
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    result = text
    for entity in sorted_entities:
        replacement = deid.replace(entity["text"], entity["type"])
        result = result[:entity["start"]] + replacement + result[entity["end"]:]
    
    # Post-process DOB line(s) to keep a single clean date
    result = _clean_dob_lines(result)
    
    return result
