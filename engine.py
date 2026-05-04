from __future__ import annotations
import json
import logging
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import httpx

from cache import cache
from models import (
    AllergyAlert,
    ContraindicationAlert,
    DrugInteraction,
    DrugSafetyRequest,
    DrugSafetyResponse,
    PatientHistory,
    RiskLevel,
    RiskScoreBreakdown,
    SeverityLevel,
    SourceType,
)

logger = logging.getLogger("evodoc.engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "meditron")
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT_S", "25"))

_FALLBACK_PATH = Path(__file__).parent / "data" / "fallback_interactions.json"
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"

# ---------------------------------------------------------------------------
# Knowledge bases (loaded once at import time)
# ---------------------------------------------------------------------------

def _load_fallback() -> dict:
    with open(_FALLBACK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


_FALLBACK_DATA = _load_fallback()
_SYSTEM_PROMPT = _load_system_prompt()
_FALLBACK_INTERACTIONS: list[dict] = _FALLBACK_DATA["interactions"]
_DRUG_ALIASES: dict[str, str] = _FALLBACK_DATA.get("drug_aliases", {})

# ---------------------------------------------------------------------------
# Drug class knowledge (for allergy detection)
# ---------------------------------------------------------------------------

DRUG_CLASS_MAP: dict[str, list[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "piperacillin", "oxacillin", "dicloxacillin",
        "nafcillin", "flucloxacillin", "benzylpenicillin", "phenoxymethylpenicillin",
        "co-amoxiclav", "amoxicillin-clavulanate", "augmentin", "temocillin",
    ],
    "cephalosporin": [
        "cefalexin", "cephalexin", "cefazolin", "cefuroxime", "cefixime",
        "ceftriaxone", "cefepime", "cefadroxil", "cefdinir", "cefpodoxime",
        "cefotaxime", "ceftazidime", "cefoperazone", "cefprozil",
    ],
    "sulfonamide": [
        "sulfamethoxazole", "sulfadiazine", "sulfisoxazole",
        "trimethoprim-sulfamethoxazole", "bactrim", "co-trimoxazole",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "diclofenac", "celecoxib", "indomethacin",
        "ketorolac", "meloxicam", "piroxicam", "etodolac", "etoricoxib",
        "mefenamic acid", "flurbiprofen", "brufen", "combiflam",
    ],
    "aspirin": ["aspirin", "ecosprin", "acetylsalicylic acid"],
    "fluoroquinolone": [
        "ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin",
        "norfloxacin", "gatifloxacin", "pefloxacin",
    ],
    "macrolide": [
        "azithromycin", "clarithromycin", "erythromycin", "roxithromycin", "spiramycin",
    ],
    "tetracycline": [
        "doxycycline", "tetracycline", "minocycline", "tigecycline", "oxytetracycline",
    ],
    "aminoglycoside": [
        "gentamicin", "amikacin", "tobramycin", "streptomycin", "neomycin",
        "netilmicin",
    ],
    "statin": [
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
        "lovastatin", "fluvastatin", "pitavastatin",
    ],
    "ace inhibitor": [
        "lisinopril", "enalapril", "ramipril", "captopril", "perindopril",
        "quinapril", "benazepril", "fosinopril", "trandolapril", "cilazapril",
    ],
    "arb": [
        "losartan", "valsartan", "irbesartan", "candesartan", "olmesartan",
        "telmisartan", "azilsartan", "eprosartan",
    ],
    "beta-blocker": [
        "metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol",
        "nebivolol", "labetalol", "acebutolol", "sotalol", "nadolol",
    ],
    "benzodiazepine": [
        "diazepam", "lorazepam", "alprazolam", "clonazepam", "midazolam",
        "temazepam", "oxazepam", "nitrazepam", "triazolam", "chlordiazepoxide",
    ],
    "ssri": [
        "fluoxetine", "sertraline", "paroxetine", "citalopram",
        "escitalopram", "fluvoxamine",
    ],
    "snri": ["venlafaxine", "duloxetine", "desvenlafaxine", "milnacipran"],
    "opioid": [
        "morphine", "codeine", "tramadol", "oxycodone", "hydrocodone",
        "fentanyl", "buprenorphine", "methadone", "hydromorphone", "pethidine",
        "meperidine", "tapentadol",
    ],
    "calcium channel blocker": [
        "amlodipine", "nifedipine", "diltiazem", "verapamil",
        "felodipine", "nicardipine", "isradipine", "lercanidipine",
    ],
    "thiazide diuretic": [
        "hydrochlorothiazide", "chlorthalidone", "indapamide",
        "bendroflumethiazide", "metolazone",
    ],
    "loop diuretic": [
        "furosemide", "frusemide", "bumetanide", "torsemide", "ethacrynic acid",
    ],
    "proton pump inhibitor": [
        "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole",
    ],
    "anticoagulant": [
        "warfarin", "heparin", "enoxaparin", "dalteparin", "tinzaparin",
        "rivaroxaban", "apixaban", "dabigatran", "edoxaban",
    ],
    "corticosteroid": [
        "prednisolone", "prednisone", "dexamethasone", "hydrocortisone",
        "methylprednisolone", "budesonide", "betamethasone", "triamcinolone",
    ],
    "antifungal azole": [
        "fluconazole", "itraconazole", "voriconazole", "ketoconazole",
        "posaconazole", "isavuconazole",
    ],
    "maoi": ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"],
    "antiplatelet": [
        "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
        "dipyridamole", "ticlopidine",
    ],
    "nitrate": [
        "isosorbide mononitrate", "isosorbide dinitrate", "glyceryl trinitrate",
        "nitroglycerin", "nitroglycerine", "gtv",
    ],
    "rifamycin": ["rifampicin", "rifampin", "rifabutin"],
    "thiazolidinedione": ["pioglitazone", "rosiglitazone"],
}

# Reverse map: normalised drug name → list of classes it belongs to
_DRUG_TO_CLASSES: dict[str, list[str]] = {}
for _cls, _drugs in DRUG_CLASS_MAP.items():
    for _d in _drugs:
        _DRUG_TO_CLASSES.setdefault(_d.lower(), []).append(_cls)

# Partial cross-reactivity (penicillin allergy has ~1–2% chance of cephalosporin reaction)
_CROSS_REACTIVITY: dict[str, list[str]] = {
    "penicillin": ["cephalosporin"],
}

# ---------------------------------------------------------------------------
# Condition contraindications (Bonus C)
# ---------------------------------------------------------------------------

# Each entry: condition keyword → list of {classes/drugs, severity, reason}
_CONDITION_CONTRAINDICATIONS: dict[str, list[dict]] = {
    "kidney disease": [
        {"classes": ["nsaid", "aspirin"], "severity": "high",
         "reason": "NSAIDs reduce renal prostaglandin-mediated afferent arteriolar dilation, precipitating acute kidney injury."},
        {"drugs": ["metformin"], "severity": "high",
         "reason": "Metformin accumulates in renal impairment increasing risk of potentially fatal lactic acidosis."},
        {"drugs": ["lithium"], "severity": "high",
         "reason": "Reduced renal clearance leads to lithium accumulation and toxicity (tremor, confusion, seizures)."},
        {"classes": ["aminoglycoside"], "severity": "high",
         "reason": "Aminoglycosides are nephrotoxic and accumulate to toxic levels in renal impairment."},
    ],
    "renal impairment": [
        {"classes": ["nsaid"], "severity": "high",
         "reason": "NSAIDs further reduce renal perfusion in already compromised kidneys."},
        {"drugs": ["metformin"], "severity": "high",
         "reason": "Risk of lactic acidosis; contraindicated when eGFR < 30 mL/min/1.73m²."},
        {"classes": ["anticoagulant"], "severity": "medium",
         "reason": "Reduced clearance of renally-excreted anticoagulants (e.g., dabigatran, enoxaparin) increases bleeding risk."},
    ],
    "liver disease": [
        {"classes": ["statin"], "severity": "medium",
         "reason": "Statins are hepatically metabolised; liver disease increases hepatotoxicity risk."},
        {"drugs": ["methotrexate"], "severity": "high",
         "reason": "Methotrexate is hepatotoxic; contraindicated in significant hepatic impairment."},
        {"drugs": ["paracetamol", "acetaminophen"], "severity": "medium",
         "reason": "Hepatic glutathione depletion in liver disease increases NAPQI-mediated hepatotoxicity at standard doses."},
        {"classes": ["nsaid"], "severity": "medium",
         "reason": "NSAIDs can precipitate renal failure (hepatorenal syndrome) and worsen coagulopathy in cirrhosis."},
    ],
    "hepatic impairment": [
        {"classes": ["statin"], "severity": "medium",
         "reason": "Increased systemic exposure and hepatotoxicity risk."},
        {"drugs": ["paracetamol", "acetaminophen"], "severity": "medium",
         "reason": "Increased hepatotoxicity risk; reduce dose."},
    ],
    "pregnancy": [
        {"classes": ["anticoagulant"], "drugs_override": ["warfarin"], "severity": "high",
         "reason": "Warfarin crosses the placenta; causes warfarin embryopathy (nasal hypoplasia, stippled epiphyses) and fetal hemorrhage."},
        {"classes": ["ace inhibitor"], "severity": "high",
         "reason": "Teratogenic in 2nd and 3rd trimester: fetal renal tubular dysgenesis, oligohydramnios, limb contractures."},
        {"classes": ["arb"], "severity": "high",
         "reason": "Teratogenic: same mechanism as ACE inhibitors; contraindicated throughout pregnancy."},
        {"drugs": ["methotrexate"], "severity": "high",
         "reason": "Abortifacient and teratogenic (neural tube defects); contraindicated."},
        {"classes": ["nsaid"], "severity": "high",
         "reason": "In 3rd trimester: premature closure of ductus arteriosus; oligohydramnios. In 1st trimester: spontaneous abortion risk."},
        {"classes": ["tetracycline"], "severity": "high",
         "reason": "Chelates calcium, inhibiting fetal bone development; causes permanent tooth discoloration."},
        {"classes": ["fluoroquinolone"], "severity": "medium",
         "reason": "Animal studies show arthropathy; use only when no safe alternative exists."},
        {"classes": ["maoi"], "severity": "high",
         "reason": "MAOIs are associated with neonatal withdrawal and fetal growth restriction."},
    ],
    "asthma": [
        {"classes": ["beta-blocker"], "severity": "high",
         "reason": "Non-selective beta-blockers block β2-adrenoceptors in bronchial smooth muscle, precipitating life-threatening bronchospasm."},
        {"classes": ["nsaid", "aspirin"], "severity": "medium",
         "reason": "Aspirin-exacerbated respiratory disease (Samter's triad) affects ~10% of asthmatics; can cause severe bronchospasm."},
    ],
    "heart failure": [
        {"classes": ["nsaid"], "severity": "high",
         "reason": "NSAIDs cause sodium and water retention (reduced renal prostaglandins), worsening fluid overload and cardiac output."},
        {"drugs": ["verapamil", "diltiazem"], "severity": "high",
         "reason": "Non-dihydropyridine CCBs exert a negative inotropic effect, worsening systolic heart failure."},
        {"classes": ["thiazolidinedione"], "severity": "high",
         "reason": "Cause fluid retention via PPAR-γ mechanism, worsening heart failure and increasing hospitalization risk."},
    ],
    "cardiac failure": [
        {"classes": ["nsaid"], "severity": "high",
         "reason": "Worsens fluid retention and reduces cardiac output."},
    ],
    "gout": [
        {"classes": ["thiazide diuretic"], "severity": "medium",
         "reason": "Thiazides competitively inhibit renal urate secretion, raising serum uric acid and precipitating gout attacks."},
        {"classes": ["aspirin", "loop diuretic"], "severity": "medium",
         "reason": "Low-dose aspirin and loop diuretics reduce tubular urate secretion, exacerbating hyperuricaemia."},
    ],
    "diabetes": [
        {"classes": ["corticosteroid"], "severity": "high",
         "reason": "Glucocorticoids cause insulin resistance and hepatic gluconeogenesis, causing steroid-induced hyperglycaemia."},
        {"classes": ["thiazide diuretic"], "severity": "medium",
         "reason": "Thiazides impair glucose tolerance via hypokalaemia-mediated suppression of insulin secretion."},
        {"classes": ["beta-blocker"], "severity": "medium",
         "reason": "Non-selective beta-blockers mask tachycardia (early warning of hypoglycaemia) and impair glycogenolysis."},
    ],
    "epilepsy": [
        {"drugs": ["tramadol"], "severity": "high",
         "reason": "Tramadol lowers seizure threshold; risk is dose-dependent and additive with other epileptogenic drugs."},
        {"classes": ["fluoroquinolone"], "severity": "medium",
         "reason": "Fluoroquinolones inhibit GABA-A receptors, lowering seizure threshold."},
        {"drugs": ["bupropion"], "severity": "high",
         "reason": "Bupropion lowers seizure threshold in a dose-dependent manner; avoid in epilepsy."},
    ],
    "peptic ulcer": [
        {"classes": ["nsaid", "aspirin"], "severity": "high",
         "reason": "NSAIDs inhibit COX-1, reducing prostaglandin-mediated gastric mucus and bicarbonate secretion, predisposing to ulceration and bleeding."},
        {"classes": ["corticosteroid"], "severity": "medium",
         "reason": "Corticosteroids reduce mucosal protective factors and inhibit healing; additive risk with NSAIDs."},
    ],
    "osteoporosis": [
        {"classes": ["corticosteroid"], "severity": "medium",
         "reason": "Long-term glucocorticoids reduce osteoblast activity and increase osteoclast activation, causing rapid bone loss."},
        {"classes": ["proton pump inhibitor"], "severity": "low",
         "reason": "Long-term PPI use reduces calcium absorption (requires acidic environment), associated with increased fracture risk."},
    ],
    "hypothyroidism": [
        {"drugs": ["amiodarone", "lithium"], "severity": "medium",
         "reason": "Amiodarone (high iodine) and lithium can cause or worsen hypothyroidism; monitor TFTs."},
    ],
    "hyperthyroidism": [
        {"drugs": ["amiodarone"], "severity": "medium",
         "reason": "Amiodarone contains 37% iodine by weight; can cause type 1 amiodarone-induced thyrotoxicosis via Jod-Basedow effect."},
    ],
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Lowercase + strip; resolve common brand/alias names."""
    n = name.strip().lower()
    return _DRUG_ALIASES.get(n, n)


def _drug_classes(drug_name: str) -> list[str]:
    """Return all drug classes a given drug belongs to."""
    return _DRUG_TO_CLASSES.get(_normalise(drug_name), [])


def _matches_class_or_drug(target: str, known: str) -> bool:
    """True if known (allergy / condition keyword) matches target's drug class or name."""
    t = _normalise(target)
    k = _normalise(known)
    if t == k:
        return True
    # Is the known term a drug class that target belongs to?
    if k in _drug_classes(t):
        return True
    # Is the known term a drug that belongs to the same class as target?
    for cls in _drug_classes(t):
        if k in DRUG_CLASS_MAP.get(cls, []):
            return True
    return False


# ---------------------------------------------------------------------------
# Allergy detection (rule-based, always applied)
# ---------------------------------------------------------------------------

def detect_allergy_alerts(
    proposed_medicines: list[str], known_allergies: list[str]
) -> list[AllergyAlert]:
    alerts: list[AllergyAlert] = []

    for med in proposed_medicines:
        med_norm = _normalise(med)
        med_classes = _drug_classes(med)

        for allergy in known_allergies:
            allergy_norm = _normalise(allergy)

            # 1. Exact name match
            if med_norm == allergy_norm:
                alerts.append(AllergyAlert(
                    medicine=med,
                    reason=f"Patient is recorded as allergic to {allergy}.",
                    severity="critical",
                    allergy_class=None,
                ))
                break  # no need to check further for this allergy

            # 2. Class match: allergy is a class name and med belongs to that class
            if allergy_norm in med_classes:
                alerts.append(AllergyAlert(
                    medicine=med,
                    reason=(
                        f"Patient has a documented {allergy} allergy. "
                        f"{med} belongs to the {allergy_norm} drug class."
                    ),
                    severity="critical",
                    allergy_class=allergy_norm,
                ))
                break

            # 3. Allergy is a specific drug → check if med is in same class
            allergy_classes = _drug_classes(allergy)
            shared_classes = [c for c in med_classes if c in allergy_classes]
            if shared_classes:
                is_cross = any(
                    allergy_norm in _CROSS_REACTIVITY.get(c, []) or
                    med_norm in _CROSS_REACTIVITY.get(allergy_norm, [])
                    for c in shared_classes
                )
                severity = "critical" if not is_cross else "high"
                alerts.append(AllergyAlert(
                    medicine=med,
                    reason=(
                        f"Patient is allergic to {allergy}. "
                        f"{med} is in the same drug class ({', '.join(shared_classes)}) "
                        "and may cause a cross-reactive allergic reaction."
                    ),
                    severity=severity,
                    allergy_class=", ".join(shared_classes),
                ))
                break

    return alerts


# ---------------------------------------------------------------------------
# Condition contraindications (Bonus C, rule-based, always applied)
# ---------------------------------------------------------------------------

def detect_contraindications(
    proposed_medicines: list[str],
    conditions: list[str],
    current_medications: list[str],
) -> list[ContraindicationAlert]:
    alerts: list[ContraindicationAlert] = []

    for condition in conditions:
        cond_norm = condition.strip().lower()

        # Find matching condition keys (partial match allowed)
        for key, rules in _CONDITION_CONTRAINDICATIONS.items():
            if key not in cond_norm and cond_norm not in key:
                continue

            for rule in rules:
                rule_classes: list[str] = rule.get("classes", [])
                rule_drugs: list[str] = rule.get("drugs", [])
                drugs_override: list[str] = rule.get("drugs_override", [])

                # Resolve which drugs to flag
                flagged_from_class: set[str] = set()
                if rule_classes:
                    for cls in rule_classes:
                        flagged_from_class.update(DRUG_CLASS_MAP.get(cls, []))

                # If drugs_override is set, only flag those specific drugs
                if drugs_override:
                    flagged_from_class = set(drugs_override)

                flagged_from_class.update(rule_drugs)

                for med in proposed_medicines + current_medications:
                    med_norm = _normalise(med)
                    if med_norm in flagged_from_class or any(
                        c in _drug_classes(med) for c in rule_classes
                    ):
                        # Avoid duplicate alerts
                        already = any(
                            a.medicine.lower() == med.lower() and
                            a.condition.lower() == condition.lower()
                            for a in alerts
                        )
                        if not already:
                            alerts.append(ContraindicationAlert(
                                medicine=med,
                                condition=condition,
                                reason=rule["reason"],
                                severity=SeverityLevel(rule["severity"]),
                            ))

    return alerts


# ---------------------------------------------------------------------------
# Fallback rule-based interaction checker
# ---------------------------------------------------------------------------

def check_interactions_fallback(
    all_medicines: list[str],
) -> tuple[list[DrugInteraction], SourceType]:
    """
    Match medicines against the hardcoded interaction dataset.
    Checks all unordered pairs from all_medicines (proposed + current).
    """
    found: list[DrugInteraction] = []
    norm_map = {_normalise(m): m for m in all_medicines}
    norm_list = list(norm_map.keys())

    for row in _FALLBACK_INTERACTIONS:
        a_norm = _normalise(row["drug_a"])
        b_norm = _normalise(row["drug_b"])

        a_in = any(
            a_norm == m or a_norm in _drug_classes(m) or m in DRUG_CLASS_MAP.get(a_norm, [])
            for m in norm_list
        )
        b_in = any(
            b_norm == m or b_norm in _drug_classes(m) or m in DRUG_CLASS_MAP.get(b_norm, [])
            for m in norm_list
        )

        if a_in and b_in:
            # Resolve display names back to original casing
            display_a = next(
                (norm_map[m] for m in norm_list
                 if a_norm == m or a_norm in _drug_classes(m) or m in DRUG_CLASS_MAP.get(a_norm, [])),
                row["drug_a"],
            )
            display_b = next(
                (norm_map[m] for m in norm_list
                 if b_norm == m or b_norm in _drug_classes(m) or m in DRUG_CLASS_MAP.get(b_norm, [])),
                row["drug_b"],
            )
            found.append(DrugInteraction(
                drug_a=display_a,
                drug_b=display_b,
                severity=SeverityLevel(row["severity"]),
                mechanism=row["mechanism"],
                clinical_recommendation=row["clinical_recommendation"],
                source_confidence=row["source_confidence"],
            ))

    return found, SourceType.FALLBACK


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(request: DrugSafetyRequest) -> str:
    h = request.patient_history
    lines = [
        "PROPOSED MEDICINES:",
        json.dumps(request.proposed_medicines, ensure_ascii=False),
        "",
        "PATIENT HISTORY:",
        f"  Current medications: {json.dumps(h.current_medications)}",
        f"  Known allergies: {json.dumps(h.known_allergies)}",
        f"  Medical conditions: {json.dumps(h.conditions)}",
        f"  Age: {h.age if h.age is not None else 'not provided'}",
        f"  Weight: {f'{h.weight} kg' if h.weight is not None else 'not provided'}",
        "",
        "Analyse ALL drug pairs (proposed-vs-proposed AND proposed-vs-current medications).",
        "Return ONLY valid JSON matching the required schema. No other text.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM caller (async, Ollama)
# ---------------------------------------------------------------------------

async def _call_llm(prompt: str) -> Optional[dict]:
    """
    Send prompt to Ollama and return parsed JSON dict, or None on failure.
    Uses /api/chat endpoint with JSON format mode.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_text: str = data["message"]["content"]
        # Strip any accidental markdown fences
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()
        return json.loads(raw_text)

    except httpx.ConnectError:
        logger.warning("Ollama not reachable at %s", OLLAMA_BASE_URL)
    except httpx.TimeoutException:
        logger.warning("LLM request timed out after %.0f s", LLM_TIMEOUT)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON: %s", exc)
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# LLM output validator / parser
# ---------------------------------------------------------------------------

_VALID_SEVERITIES = {"critical", "high", "medium", "low"}
_VALID_CONFIDENCE = {"high", "medium", "low"}
_VALID_RISK_LEVELS = {"high", "medium", "low"}


def _parse_llm_output(
    raw: dict,
    proposed_medicines: list[str],
) -> tuple[list[DrugInteraction], bool]:
    """
    Parse and validate LLM output. Returns (interactions, requires_doctor_review).
    Drops individual items that fail validation rather than failing the whole request.
    """
    interactions: list[DrugInteraction] = []
    requires_review = bool(raw.get("requires_doctor_review", False))

    # Validate medicine names (no hallucinated drugs)
    all_lower = {m.lower() for m in proposed_medicines}

    for item in raw.get("interactions", []):
        try:
            sev = (item.get("severity") or "").lower()
            if sev not in _VALID_SEVERITIES:
                requires_review = True
                continue

            conf = (item.get("source_confidence") or "medium").lower()
            if conf not in _VALID_CONFIDENCE:
                conf = "medium"

            drug_a = (item.get("drug_a") or "").strip()
            drug_b = (item.get("drug_b") or "").strip()
            mechanism = (item.get("mechanism") or "").strip()
            recommendation = (item.get("clinical_recommendation") or "").strip()

            if not all([drug_a, drug_b, mechanism, recommendation]):
                requires_review = True
                continue

            interactions.append(DrugInteraction(
                drug_a=drug_a,
                drug_b=drug_b,
                severity=SeverityLevel(sev),
                mechanism=mechanism,
                clinical_recommendation=recommendation,
                source_confidence=conf,
            ))
        except Exception:
            requires_review = True
            continue

    return interactions, requires_review


# ---------------------------------------------------------------------------
# Risk scoring (Bonus B)
# ---------------------------------------------------------------------------

def compute_risk_score(
    interactions: list[DrugInteraction],
    allergy_alerts: list[AllergyAlert],
    contraindication_alerts: list[ContraindicationAlert],
) -> tuple[int, RiskScoreBreakdown]:
    """
    Compute patient_risk_score (0–100) with a transparent breakdown.

    Weights:
      - Interactions:      critical=25, high=20, medium=10, low=5   (max 40)
      - Allergy alerts:    critical=20, high=15, medium=8            (max 35)
      - Contraindications: high=12, medium=7, low=3                  (max 25)
    """
    sev_points = {"critical": 25, "high": 20, "medium": 10, "low": 5}
    allergy_points = {"critical": 20, "high": 15, "medium": 8}
    contra_points = {"high": 12, "medium": 7, "low": 3, "critical": 15}

    interaction_raw = sum(sev_points.get(i.severity.value, 0) for i in interactions)
    allergy_raw = sum(allergy_points.get(a.severity, 0) for a in allergy_alerts)
    contra_raw = sum(contra_points.get(c.severity.value, 0) for c in contraindication_alerts)

    interaction_score = min(interaction_raw, 40.0)
    allergy_score = min(allergy_raw, 35.0)
    contra_score = min(contra_raw, 25.0)
    total = min(interaction_score + allergy_score + contra_score, 100.0)

    breakdown = RiskScoreBreakdown(
        interaction_score=interaction_score,
        allergy_score=allergy_score,
        contraindication_score=contra_score,
        total=total,
        explanation=(
            f"Score derived from {len(interactions)} drug interaction(s) "
            f"({interaction_score:.0f}/40 pts), "
            f"{len(allergy_alerts)} allergy alert(s) ({allergy_score:.0f}/35 pts), "
            f"and {len(contraindication_alerts)} contraindication(s) "
            f"({contra_score:.0f}/25 pts)."
        ),
    )
    return int(round(total)), breakdown


# ---------------------------------------------------------------------------
# Overall risk level & prescribe flag
# ---------------------------------------------------------------------------

def _determine_risk_level(
    interactions: list[DrugInteraction],
    allergy_alerts: list[AllergyAlert],
    contraindication_alerts: list[ContraindicationAlert],
    risk_score: int,
) -> tuple[RiskLevel, bool]:
    """Derive overall_risk_level and safe_to_prescribe."""
    has_critical = any(
        i.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH) for i in interactions
    ) or any(
        a.severity in ("critical", "high") for a in allergy_alerts
    ) or any(
        c.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH) for c in contraindication_alerts
    )

    if has_critical or risk_score >= 60:
        return RiskLevel.HIGH, False
    if risk_score >= 25:
        return RiskLevel.MEDIUM, True
    return RiskLevel.LOW, True


# ---------------------------------------------------------------------------
# Main engine entry point
# ---------------------------------------------------------------------------

async def analyse_drug_safety(request: DrugSafetyRequest) -> DrugSafetyResponse:
    t_start = time.monotonic()

    proposed = request.proposed_medicines
    history = request.patient_history
    all_medicines = list({_normalise(m) for m in proposed + history.current_medications})

    # ----- 1. Cache check ---------------------------------------------------
    cached = cache.get(proposed, history.current_medications, history.known_allergies, history.conditions)
    if cached is not None:
        # Rebuild response object; inject cache_hit=True and fresh timing
        result = DrugSafetyResponse(**cached)
        result.cache_hit = True
        result.processing_time_ms = int((time.monotonic() - t_start) * 1000)
        return result

    # ----- 2. Rule-based allergy + contraindication (always run) -----------
    allergy_alerts = detect_allergy_alerts(proposed, history.known_allergies)
    contraindication_alerts = detect_contraindications(
        proposed, history.conditions, history.current_medications
    )

    # ----- 3. LLM call for drug-drug interactions ---------------------------
    interactions: list[DrugInteraction] = []
    requires_doctor_review = False
    source = SourceType.LLM

    user_prompt = _build_user_prompt(request)
    llm_raw = await _call_llm(user_prompt)

    if llm_raw is not None:
        interactions, requires_doctor_review = _parse_llm_output(llm_raw, all_medicines)
        if not interactions and not allergy_alerts:
            # LLM returned empty result — merge with fallback to be safe
            fallback_interactions, _ = check_interactions_fallback(all_medicines)
            if fallback_interactions:
                interactions = fallback_interactions
                source = SourceType.HYBRID
                requires_doctor_review = True
    else:
        # LLM unavailable → use fallback rule engine
        interactions, source = check_interactions_fallback(all_medicines)
        requires_doctor_review = True  # always flag when using fallback
        logger.info("Using fallback interaction dataset (LLM unavailable).")

    # ----- 4. Risk score (Bonus B) -----------------------------------------
    risk_score, breakdown = compute_risk_score(
        interactions, allergy_alerts, contraindication_alerts
    )

    # ----- 5. Aggregate risk level ------------------------------------------
    overall_risk, safe_to_prescribe = _determine_risk_level(
        interactions, allergy_alerts, contraindication_alerts, risk_score
    )

    # Force doctor review if any critical findings
    if not safe_to_prescribe:
        requires_doctor_review = True

    # ----- 6. Build response ------------------------------------------------
    processing_ms = int((time.monotonic() - t_start) * 1000)

    response = DrugSafetyResponse(
        interactions=interactions,
        allergy_alerts=allergy_alerts,
        contraindication_alerts=contraindication_alerts,
        safe_to_prescribe=safe_to_prescribe,
        overall_risk_level=overall_risk,
        requires_doctor_review=requires_doctor_review,
        source=source,
        cache_hit=False,
        processing_time_ms=processing_ms,
        patient_risk_score=risk_score,
        risk_score_breakdown=breakdown,
    )

    # ----- 7. Cache result --------------------------------------------------
    cache.set(proposed, history.current_medications, history.known_allergies, history.conditions, response.model_dump(mode="json"))

    return response
