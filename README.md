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

### POST /analyse — Drug safety check

The main endpoint. Send medicines + patient history, get back a full safety assessment.

**Request body:**

```json
{
  "proposed_medicines": ["Warfarin", "Aspirin", "Amoxicillin"],
  "patient_history": {
    "current_medications": ["Metformin", "Atenolol"],
    "known_allergies": ["penicillin"],
    "conditions": ["diabetes", "kidney disease"],
    "age": 67,
    "weight": 72.5
  }
}
```

**Local:**

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"proposed_medicines":["Warfarin","Aspirin","Amoxicillin"],"patient_history":{"current_medications":["Metformin","Atenolol"],"known_allergies":["penicillin"],"conditions":["diabetes","kidney disease"],"age":67,"weight":72.5}}'
```

**Deployed:**

```bash
curl -X POST https://web-production-60757.up.railway.app/analyse \
  -H "Content-Type: application/json" \
  -d '{"proposed_medicines":["Warfarin","Aspirin","Amoxicillin"],"patient_history":{"current_medications":["Metformin","Atenolol"],"known_allergies":["penicillin"],"conditions":["diabetes","kidney disease"],"age":67,"weight":72.5}}'
```

**Response fields explained:**

| Field                     | What it means                                                          |
| ------------------------- | ---------------------------------------------------------------------- |
| `interactions`            | Drug pairs that clash, with mechanism and what the doctor should do    |
| `allergy_alerts`          | Medicines that match the patient's known allergies or their drug class |
| `contraindication_alerts` | Medicines dangerous given the patient's conditions                     |
| `safe_to_prescribe`       | False if any critical or high severity finding exists                  |
| `overall_risk_level`      | high / medium / low — derived from all findings combined               |
| `requires_doctor_review`  | True if LLM was uncertain, or fallback fired, or any critical finding  |
| `source`                  | "llm" if Meditron answered, "fallback" if rule engine was used         |
| `cache_hit`               | True if this exact patient history was checked before within the hour  |
| `processing_time_ms`      | How long the full analysis took                                        |
| `patient_risk_score`      | 0–100 score combining all findings (Bonus B)                           |
| `risk_score_breakdown`    | Exact points from interactions, allergies, and contraindications       |

---

### GET /health — Service status

Check if the server is running and whether Ollama is reachable.

```bash
# Local
curl http://localhost:8000/health

# Deployed
curl https://web-production-60757.up.railway.app/health
```

**Response:**

```json
{
  "status": "ok",
  "llm_backend": "http://localhost:11434",
  "llm_model": "meditron",
  "llm_status": "available",
  "fallback_available": true,
  "cache_backend": "memory"
}
```

`llm_status` will show `"unavailable"` on the deployed version since Ollama runs locally only. `fallback_available` will always be `true` — the rule engine never goes down.

---

### GET /cache/stats — Cache status

See how many results are currently cached.

```bash
# Local
curl http://localhost:8000/cache/stats

# Deployed
curl https://web-production-60757.up.railway.app/cache/stats
```

**Response:**

```json
{
  "backend": "memory",
  "entries": 3
}
```

If you set `REDIS_URL` in your `.env`, backend will show `"redis"` instead of `"memory"`.

---

### GET /interactions/fallback — Fallback interaction rules

Returns all 22 hardcoded drug interactions used when the LLM is unavailable. Useful for auditing what the rule engine knows.

```bash
# Local
curl http://localhost:8000/interactions/fallback

# Deployed
curl https://web-production-60757.up.railway.app/interactions/fallback
```

**Response:**

```json
{
  "count": 22,
  "interactions": [
    {
      "drug_a": "warfarin",
      "drug_b": "aspirin",
      "severity": "high",
      "mechanism": "...",
      "clinical_recommendation": "...",
      "source_confidence": "high"
    },
    ...
  ]
}
```

---

### Interactive docs (Swagger UI)

FastAPI automatically generates an interactive API explorer. You can test every endpoint from the browser without Postman or curl.

```
Local:    http://localhost:8000/docs
Deployed: https://web-production-60757.up.railway.app/docs
```

Click any endpoint → **Try it out** → fill in the body → **Execute**.

---

### Validation errors (422)

The API validates every input and returns clear errors for bad data:

```bash
# Empty medicine list
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"proposed_medicines":[],"patient_history":{}}'

# Response:
{
  "detail": [
    {
      "loc": ["body", "proposed_medicines"],
      "msg": "List should have at least 1 item after validation",
      "type": "too_short"
    }
  ]
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
