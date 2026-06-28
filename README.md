# deid-LONGFORMER-NemPII

**HIPAA-compliant clinical de-identification that beats commercial solutions—at zero cost.**

A fine-tuned Clinical-Longformer model for Protected Health Information (PHI) detection and replacement in clinical text, achieving **97.74% F1** on held-out test data.

## Why This Exists

Commercial de-identification solutions are expensive and produce unusable output:

| Solution | F1 Score | Cost | Replacement Quality |
|----------|----------|------|---------------------|
| AWS Comprehend Medical | ~83-93% | $14.5K/1M notes | Basic placeholders |
| John Snow Labs | 96-97% | Enterprise license | Basic placeholders |
| **deid-LONGFORMER-NemPII** | **97.74%** | **Free/self-hosted** | **Realistic surrogates** |

Most tools just redact PHI with `[REDACTED]` or `***`, leaving text that's difficult to read and impossible to use for downstream NLP tasks. This model generates **realistic surrogate data** that preserves clinical meaning while protecting patient privacy.

## Acknowledgments

This project stands on the shoulders of excellent prior work:

### Inspiration: obi/deid_bert_i2b2

This model was directly inspired by [**obi/deid_bert_i2b2**](https://huggingface.co/obi/deid_bert_i2b2) from the Open Biomedical Informatics team (Prajwal Kailas, Max Homilius, Shinichi Goto). Their work on ClinicalBERT-based de-identification using the I2B2 2014 dataset demonstrated the viability of transformer-based approaches for this task. The [robust-deid](https://github.com/obi-ml-public/ehr_deidentification) framework they developed provided invaluable reference for architecture decisions and evaluation methodology.

### Base Model: yikuan8/Clinical-Longformer

The base model is [**yikuan8/Clinical-Longformer**](https://huggingface.co/yikuan8/Clinical-Longformer) by Li, Yikuan et al. This clinical knowledge-enriched Longformer was pre-trained on MIMIC-III clinical notes and supports sequences up to 4,096 tokens—critical for processing real-world clinical documents that often exceed BERT's 512-token limit.

> Li, Yikuan, et al. "A comparative study of pretrained language models for long clinical text." *Journal of the American Medical Informatics Association* 30.2 (2023): 340-347.

### Training Data: NVIDIA Nemotron-PII

Training data comes from the healthcare subset of [**NVIDIA's Nemotron-PII**](https://huggingface.co/datasets/nvidia/Nemotron-PII) dataset (3,630 records, CC BY 4.0 license). This synthetic dataset provides diverse PHI patterns without exposing real patient data.

## Key Differentiators

Unlike competitors that just redact text, this system generates **clinically useful surrogates**:

### Age-Preserving DOB Replacement
Dates of birth are replaced with fake DOBs that preserve the patient's age within ±2 years. A 67-year-old patient stays clinically 65-69, not "[REDACTED]".

### Context-Aware Detection
The model recognizes that a DATE entity following "DOB:" should receive age-preserving treatment, not standard date shifting.

### Name Consistency
Multiple references to the same person map to the same fake name:
- "Dr. Sarah Elizabeth Johnson, MD" → "Dr. Maria Rodriguez, MD"
- "Sarah E. Johnson" → "Maria Rodriguez"
- "Dr. Johnson" → "Dr. Rodriguez"

### Temporal Consistency
All dates in a document shift by the same random offset. If admission was January 15 and discharge was January 20, the 5-day relationship is preserved.

### Geographic Consistency
City, state, and ZIP code replacements are coherent—you won't get "Phoenix, NY 33101".

### Format Preservation
Phone numbers, SSNs, and dates maintain their original format:
- `(555) 123-4567` → `(555) 987-6543` (not `5559876543`)
- `01/15/2024` → `03/22/2024` (not `2024-03-22`)

### Medical Term Protection
A whitelist prevents false positives on medical terms:
- "Anion Gap" stays "Anion Gap" (not replaced as a name)
- "BUN" stays "BUN"
- "2 weeks" stays "2 weeks" (not detected as a date)

### Adjacent Entity Merging
When the model fragments entities across tokens, post-processing merges them:
- `["Jan", "uary", " ", "15"]` → `"January 15"` (single DATE entity)

## Model Architecture

```
Base Model:     yikuan8/Clinical-Longformer
Parameters:     148M
Max Length:     4,096 tokens
Task:           Token Classification (NER)
Tagging:        BILOU scheme
Classes:        101 (25 PHI types × 4 BILOU tags + O)
```

### PHI Categories (25 types)

```
NAME, FIRST_NAME, LAST_NAME, DATE, DATE_OF_BIRTH, DATE_TIME,
TIME, AGE, SSN, MEDICAL_RECORD_NUMBER, HEALTH_PLAN_BENEFICIARY_NUMBER,
ACCOUNT_NUMBER, CERTIFICATE_LICENSE_NUMBER, PHONE_NUMBER, FAX_NUMBER,
EMAIL, STREET_ADDRESS, CITY, STATE, POSTCODE, COUNTRY,
BIOMETRIC_IDENTIFIER, UNIQUE_ID, CUSTOMER_ID, EMPLOYEE_ID
```

## Installation

```bash
git clone https://github.com/Hrygt/deid-longformer-nempii.git
cd deid-longformer-nempii
pip install -r requirements.txt
```

Download model weights from HuggingFace:

```python
from huggingface_hub import snapshot_download
snapshot_download('riggsmed/deid-LONGFORMER-NemPII', local_dir='model')
```

## Quick Start

### Python API

```python
from deid import deidentify_text

text = """
PATIENT: John Smith
DOB: 01/15/1957
MRN: 123456789

Mr. Smith presented to the ED on 12/09/2024 with chest pain.
Contact: (405) 555-1234
"""

result = deidentify_text(text)
print(result["deidentified_text"])
```

Output:
```
PATIENT: Robert Johnson
DOB: 03/22/1955
MRN: 987654321

Mr. Johnson presented to the ED on 02/15/2025 with chest pain.
Contact: (555) 987-6543
```

### FastAPI Service

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST http://localhost:8001/deidentify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Smith, DOB 01/15/1957"}'
```

#### Re-identification map (`return_map`)

`/deid/process` accepts **`return_map`** (bool, default `false`) — **independent of `return_entities`**. When `true`, the response includes a `surrogate_map` that lets a trusted downstream caller restore the real entities in its *output* (the model still only ever sees scrubbed text):

```jsonc
// POST /deid/process  { "text": "...", "return_map": true }
{
  "deidentified_text": "... Dr. Dawn Williams ...",
  "surrogate_map": {
    "version": 1,
    "pairs": [
      {"original": "Anjali Patel", "surrogate": "Dawn Williams", "type": "NAME",       "collision": false},
      {"original": "Anjali",       "surrogate": "Dawn",          "type": "FIRST_NAME", "collision": false},
      {"original": "Patel",        "surrogate": "Williams",      "type": "LAST_NAME",  "collision": false}
    ]
  }
}
```

`pairs` includes full-name **and** token-level name pairs (so a lone surname can be restored). `collision` is `true` when the **surrogate** token matches the medical/common whitelist (the caller should then skip it). `return_map` is off by default, so existing callers are unaffected. Consumed by the CPT grader's re-identification overlay.

## Training

To train on your own data:

```bash
python train.py \
  --model_name yikuan8/Clinical-Longformer \
  --train_file data/train.json \
  --val_file data/val.json \
  --output_dir checkpoints \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 2e-5
```

Data format (JSONL):
```json
{"text": "Patient John Smith...", "entities": [{"start": 8, "end": 18, "label": "NAME"}]}
```

## Evaluation Results

Evaluated on 20% held-out split from Nemotron-PII healthcare subset:

| Metric | Score |
|--------|-------|
| **F1** | **97.74%** |
| Precision | 97.62% |
| Recall | 97.86% |

### Per-Entity Performance

| Entity Type | F1 | Support |
|-------------|-----|---------|
| NAME | 98.2% | 1,247 |
| DATE | 97.9% | 2,103 |
| PHONE_NUMBER | 99.1% | 312 |
| SSN | 98.7% | 89 |
| STREET_ADDRESS | 96.4% | 445 |
| ... | ... | ... |

## Live Demo

Try it at: **https://deid.riggsmedai.com**

## License

- **Model weights**: Apache 2.0
- **Code**: Apache 2.0
- **Training data**: CC BY 4.0 (NVIDIA Nemotron-PII)

## Citation

If you use this model in your research, please cite:

```bibtex
@software{riggs2024deid,
  author = {Riggs, Gary},
  title = {deid-LONGFORMER-NemPII: Clinical De-identification with Realistic Surrogates},
  year = {2024},
  url = {https://github.com/Hrygt/deid-longformer-nempii}
}
```

And please also cite the foundational work this builds upon:

```bibtex
@article{li2023comparative,
  title={A comparative study of pretrained language models for long clinical text},
  author={Li, Yikuan and Wehbe, Ramsey M and Ahmad, Faraz S and Wang, Hanyin and Luo, Yuan},
  journal={Journal of the American Medical Informatics Association},
  volume={30},
  number={2},
  pages={340--347},
  year={2023}
}

@misc{obi_deid,
  author = {Kailas, Prajwal and Homilius, Max and Goto, Shinichi},
  title = {Robust De-ID: De-Identification of Medical Notes using Transformer Architectures},
  year = {2022},
  url = {https://github.com/obi-ml-public/ehr_deidentification}
}
```

## Author

**Gary Riggs, MD**  
Medical Director, Metro Physician Group  
Master of Science in Data Science candidate, Northwestern University

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Built with ❤️ for healthcare AI that respects patient privacy.*
