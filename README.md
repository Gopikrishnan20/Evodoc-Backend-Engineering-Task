## LLM Choice: Meditron-7B

**Why Meditron over a generic model:**

Fine-tuned on **PubMed**, **MedQA**, and **medical guidelines** (NEJM, WHO) — not general web text
Specifically trained for clinical decision support tasks, not chat or coding
Demonstrates measurably higher accuracy on clinical drug interaction questions vs Mistral-7B base
Produced by EPFL; openly available weights; no API dependency (zero cloud exposure)
Produces more conservative "flag and review" responses on uncertain interactions — exactly the behaviour needed for patient safety

---

## Architecture

```
POST /analyse
     │
     ├── Validate input 
     │
     ├── Check cache 
     │     └── HIT → return cached result immediately
     │
     ├── RULE ENGINE (always runs, no LLM needed):
     │     ├── Allergy Detection   — 25 drug classes, exact + class + cross-reactivity
     │     └── Contraindications   — 13 conditions × 40+ drug rules
     │
     ├── LLM ENGINE (Ollama, async, 25 s timeout):
     │     ├── Meditron via /api/chat with format=json (deterministic, temp=0)
     │     ├── Validate every field (severity enum, non-empty strings, no hallucinations)
     │     └── FALLBACK if LLM unavailable/invalid → 22-entry hardcoded dataset
     │
     ├── Risk Scoring (0–100, transparent breakdown)
     │     ├── Interactions:      critical 25, high 20, medium 10, low 5  (max 40)
     │     ├── Allergy alerts:    critical 20, high 15, medium 8          (max 35)
     │     └── Contraindications: high 12, medium 7, low 3               (max 25)
     │
     ├── Determine overall_risk_level + safe_to_prescribe
     │
     ├── Cache result
     │
     └── Return DrugSafetyResponse
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/evodoc-drug-safety.git
cd evodoc-drug-safety
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local Ollama)
```

### 3. Pull the LLM

```bash
ollama pull meditron
# Verify:
ollama list
```

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run tests

```bash
pytest tests/ -v --tb=short
```

```bash
docker run -d -p 6379:6379 redis:7-alpine
# Then add to .env:
# REDIS_URL=redis://localhost:6379
```

---

## API Usage

### POST /analyse

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "proposed_medicines": ["Warfarin", "Aspirin", "Amoxicillin"],
    "patient_history": {
      "current_medications": ["Metformin", "Atenolol"],
      "known_allergies": ["penicillin"],
      "conditions": ["diabetes", "kidney disease"],
      "age": 67,
      "weight": 72.5
    }
  }'
```

**Example response:**

```json
{
  "interactions": [
    {
      "drug_a": "Warfarin",
      "drug_b": "Aspirin",
      "severity": "high",
      "mechanism": "Warfarin inhibits clotting factor synthesis; aspirin inhibits platelet aggregation via COX-1, producing additive bleeding risk.",
      "clinical_recommendation": "Avoid concurrent use unless clinically indicated. If necessary, use lowest effective aspirin dose (75 mg) and monitor INR closely.",
      "source_confidence": "high"
    }
  ],
  "allergy_alerts": [
    {
      "medicine": "Amoxicillin",
      "reason": "Patient has a documented penicillin allergy. Amoxicillin belongs to the penicillin drug class.",
      "severity": "critical",
      "allergy_class": "penicillin"
    }
  ],
  "contraindication_alerts": [
    {
      "medicine": "Aspirin",
      "condition": "kidney disease",
      "reason": "NSAIDs reduce renal prostaglandin-mediated afferent arteriolar dilation, precipitating acute kidney injury.",
      "severity": "high"
    }
  ],
  "safe_to_prescribe": false,
  "overall_risk_level": "high",
  "requires_doctor_review": true,
  "source": "llm",
  "cache_hit": false,
  "processing_time_ms": 1840,
  "patient_risk_score": 85,
  "risk_score_breakdown": {
    "interaction_score": 20.0,
    "allergy_score": 20.0,
    "contraindication_score": 12.0,
    "total": 52.0,
    "explanation": "Score derived from 1 drug interaction(s) (20/40 pts), 1 allergy alert(s) (20/35 pts), and 1 contraindication(s) (12/25 pts)."
  }
}
```

## Fallback Dataset

`data/fallback_interactions.json` contains **22 drug-drug interactions** sourced from BNF, Lexicomp, and Micromedex, covering:

- Anticoagulant interactions (warfarin + aspirin, ibuprofen, fluconazole, amiodarone)
- Statin interactions (simvastatin + amiodarone, clarithromycin, amlodipine)
- Life-threatening combinations (sildenafil + nitrates → critical; MAOIs + SSRIs → critical)
- Common Indian prescribing scenarios (ciprofloxacin + theophylline, metronidazole + alcohol)
- Antiplatelet interactions (clopidogrel + omeprazole)

The fallback is activated when: (a) Ollama is unreachable, (b) LLM times out, or (c) LLM returns invalid JSON. A `requires_doctor_review: true` flag is always set when the fallback fires.

---

## Project Structure

```
evodoc-drug-safety/
├── main.py                         # FastAPI app, routes, middleware
├── engine.py                       # Core analysis logic (LLM + rules + scoring)
├── cache.py                        # Caching layer (in-memory + Redis)
├── models.py                       # Pydantic request/response models
├── prompts/
│   └── system_prompt.txt           # Medical LLM system prompt (Bonus A)
├── data/
│   └── fallback_interactions.json  # 22 hardcoded drug interactions
├── tests/
│   └── test_engine.py              # 40+ unit and integration tests
├── requirements.txt
├── .env.example
└── README.md
```
