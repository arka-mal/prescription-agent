# Prescription Processing Pipeline
**Agentic AI + Neuro-Symbolic AI — Assignment Demo**

A multi-agent pipeline that processes handwritten and printed medical prescriptions using a neuro-symbolic architecture.

---

## Architecture

```
Prescription Image
        ↓
[OCR / HTR Agent]          ← Google Cloud Vision (Neural)
 Handwriting + print OCR
 Mixed-script detection
 Bounding box extraction
        ↓
[Layout Agent]             ← Groq LLM / llama-3.3-70b (Neural)
 Semantic field segmentation
 Patient / Drug / Dosage / Prescriber zones
        ↓
[Prescription Agent]       ← Groq LLM + Symbolic Rules (Neuro-Symbolic)
 Drug name extraction      ← Neural (LLM)
 Brand → Generic mapping   ← Symbolic (Drug KB)
 Sig normalization         ← Symbolic (Rule engine)
 FHIR schema validation    ← Symbolic (Validator)
 HITL confidence gate      ← Symbolic (Threshold rules)
        ↓
Structured Output
FHIR-lite MedicationRequest JSON
```

### Neuro-Symbolic Design Rationale

| Component | Type | Why |
|---|---|---|
| Google Vision OCR | Neural | Handles handwriting, mixed scripts, variable layouts |
| Layout segmentation (LLM) | Neural | No fixed schema — flexible field detection |
| Sig parser | **Symbolic** | "1-0-1", "BD", "TDS" must map deterministically |
| Drug KB lookup | **Symbolic** | Brand→generic is a lookup, not inference |
| Confidence validator | **Symbolic** | Safety gate must be auditable and rule-based |
| FHIR output | **Symbolic** | Schema conformance = deterministic rules |

---

## Project Structure

```
prescription-agent/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── agents/
│   ├── ocr_agent.py              # Google Vision OCR/HTR
│   ├── layout_agent.py           # Groq LLM segmentation
│   └── prescription_agent.py     # Groq LLM + symbolic normalization
├── symbolic/
│   ├── sig_parser.py             # Rule engine: sig → TimingStructure
│   ├── drug_resolver.py          # Brand → generic KB lookup
│   └── validator.py              # Confidence gate + HITL flagging
├── models/
│   └── rx_schema.py              # Pydantic schemas (OCR/Layout/Rx/FHIR)
├── data/
│   └── drug_kb.json              # Indian drug brand→generic dictionary
├── utils/
│   ├── pipeline.py               # Pipeline orchestrator
│   └── ui_helpers.py             # Streamlit rendering helpers
└── .streamlit/
    └── secrets.toml.template     # API key template (do not commit actual keys)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/prescription-agent
cd prescription-agent
pip install -r requirements.txt
```

### 2. Get API keys

**Groq (free):**
1. Go to [console.groq.com](https://console.groq.com)
2. Create an account → API Keys → Create Key
3. Copy the `gsk_...` key

**Google Cloud Vision (free tier: 1000 images/month):**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Enable the **Cloud Vision API**
4. IAM & Admin → Service Accounts → Create → Add key → JSON
5. Download the JSON credentials file

### 3. Run locally

```bash
streamlit run app.py
```

Upload your GCP JSON and enter your Groq key in the sidebar.

---

## Deploy to Streamlit Cloud

1. Push repo to GitHub (ensure `.gitignore` excludes credentials)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as entry point
4. In **Secrets**, add:
```toml
[api_keys]
GROQ_API_KEY = "gsk_..."
```
5. For GCP credentials on Streamlit Cloud, use the sidebar file uploader

---

## Sig Parser — Supported Patterns

| Input | Parsed Output |
|---|---|
| `1-0-1` | Morning: 1, Afternoon: 0, Night: 1 |
| `1-1-1` | Morning: 1, Afternoon: 1, Night: 1, freq=3 |
| `½ tab OD hs` | Night: 0.5, freq=1, bedtime |
| `BD x 5 days` | freq=2, duration=5 days |
| `TDS after food` | freq=3, food=after food |
| `twice daily` | freq=2 |
| `SOS` | As needed |
| `1-0-0-1` | Morning: 1, Night: 1 (4-part) |

---

## Drug KB — Sample Mappings

| Brand | Generic |
|---|---|
| Augmentin | amoxicillin-clavulanate |
| Crocin / Dolo | paracetamol |
| Combiflam | ibuprofen-paracetamol |
| Pan / Pantop | pantoprazole |
| Azithral | azithromycin |
| Glycomet | metformin |
| Thyronorm | levothyroxine |

---

## Limitations & Future Work

- HTR accuracy degrades on highly idiosyncratic handwriting (publishable research gap)
- Mixed-script (Bengali/Hindi) dosage instructions partially supported
- Drug KB is manually curated — needs integration with RxNorm / CDSCO API
- No DDI (drug-drug interaction) checking in current version
- FHIR output is simplified — full R4 conformance requires additional mapping
