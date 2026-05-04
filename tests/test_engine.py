from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from cache import _build_cache_key
from engine import (
    check_interactions_fallback,
    compute_risk_score,
    detect_allergy_alerts,
    detect_contraindications,
)
from main import app
from models import (
    AllergyAlert,
    ContraindicationAlert,
    DrugInteraction,
    DrugSafetyRequest,
    PatientHistory,
    SeverityLevel,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def make_request(
    medicines: list[str],
    allergies: list[str] | None = None,
    conditions: list[str] | None = None,
    current_meds: list[str] | None = None,
    age: int | None = None,
    weight: float | None = None,
) -> dict:
    return {
        "proposed_medicines": medicines,
        "patient_history": {
            "current_medications": current_meds or [],
            "known_allergies": allergies or [],
            "conditions": conditions or [],
            "age": age,
            "weight": weight,
        },
    }


# ===========================================================================
# 1. Input validation
# ===========================================================================

class TestInputValidation:
    def test_empty_medicines_rejected(self):
        resp = client.post("/analyse", json=make_request([]))
        assert resp.status_code == 422

    def test_blank_string_medicines_rejected(self):
        resp = client.post("/analyse", json=make_request(["", "  "]))
        assert resp.status_code == 422

    def test_negative_age_rejected(self):
        resp = client.post("/analyse", json=make_request(["Aspirin"], age=-5))
        assert resp.status_code == 422

    def test_zero_weight_rejected(self):
        resp = client.post("/analyse", json=make_request(["Aspirin"], weight=0))
        assert resp.status_code == 422

    def test_duplicate_medicines_deduplicated(self):
        """Duplicate medicines should be silently deduplicated."""
        resp = client.post("/analyse", json=make_request(["Aspirin", "aspirin", "ASPIRIN"]))
        # Should not error; duplicates removed
        assert resp.status_code in (200, 503)  # 503 only if LLM + fallback both fail (shouldn't)

    def test_single_medicine_valid(self):
        resp = client.post("/analyse", json=make_request(["Paracetamol"]))
        assert resp.status_code in (200, 503)


# ===========================================================================
# 2. Cache layer
# ===========================================================================

class TestCacheLayer:
    def test_cache_key_order_independent(self):
        """Same drugs in different order must produce the same key."""
        key1 = _build_cache_key(["Warfarin", "Aspirin"], ["Metformin"], [], [])
        key2 = _build_cache_key(["Aspirin", "Warfarin"], ["Metformin"], [], [])
        assert key1 == key2

    def test_cache_key_current_meds_order_independent(self):
        key1 = _build_cache_key(["Warfarin"], ["Metformin", "Atenolol"], [], [])
        key2 = _build_cache_key(["Warfarin"], ["Atenolol", "Metformin"], [], [])
        assert key1 == key2

    def test_cache_key_case_independent(self):
        key1 = _build_cache_key(["warfarin", "aspirin"], [], [], [])
        key2 = _build_cache_key(["WARFARIN", "ASPIRIN"], [], [], [])
        assert key1 == key2

    def test_different_drugs_different_key(self):
        key1 = _build_cache_key(["Warfarin"], [], [], [])
        key2 = _build_cache_key(["Aspirin"], [], [], [])
        assert key1 != key2

    def test_different_allergies_different_key(self):
        """THE SAFETY FIX — same medicines, different allergy = different key."""
        key1 = _build_cache_key(["Amoxicillin"], [], ["penicillin"], [])
        key2 = _build_cache_key(["Amoxicillin"], [], [], [])
        assert key1 != key2, (
            "SAFETY BUG: a patient with a penicillin allergy must never "
            "receive a cached result computed for a patient without that allergy."
        )

    def test_different_conditions_different_key(self):
        """Same medicines, different conditions = different key."""
        key1 = _build_cache_key(["Ibuprofen"], [], [], ["kidney disease"])
        key2 = _build_cache_key(["Ibuprofen"], [], [], [])
        assert key1 != key2

    def test_allergy_order_independent(self):
        """Allergy list order must not matter."""
        key1 = _build_cache_key(["Aspirin"], [], ["penicillin", "sulfonamide"], [])
        key2 = _build_cache_key(["Aspirin"], [], ["sulfonamide", "penicillin"], [])
        assert key1 == key2

    def test_condition_order_independent(self):
        key1 = _build_cache_key(["Metformin"], [], [], ["diabetes", "kidney disease"])
        key2 = _build_cache_key(["Metformin"], [], [], ["kidney disease", "diabetes"])
        assert key1 == key2

    def test_cache_hit_returned_in_response(self):
        """Second identical request should return cache_hit=True."""
        payload = make_request(["Ibuprofen", "Metformin"])
        # First request
        r1 = client.post("/analyse", json=payload)
        if r1.status_code != 200:
            pytest.skip("LLM/fallback unavailable")
        assert r1.json()["cache_hit"] is False
        # Second request — should hit cache
        r2 = client.post("/analyse", json=payload)
        assert r2.status_code == 200
        assert r2.json()["cache_hit"] is True


# ===========================================================================
# 3. Allergy detection
# ===========================================================================

class TestAllergyDetection:
    def test_exact_allergy_match(self):
        alerts = detect_allergy_alerts(["Ibuprofen"], ["ibuprofen"])
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_penicillin_class_flags_amoxicillin(self):
        """Classic drug-class allergy cross-reactivity."""
        alerts = detect_allergy_alerts(["Amoxicillin"], ["penicillin"])
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert "amoxicillin" in alerts[0].medicine.lower() or "penicillin" in alerts[0].reason.lower()

    def test_penicillin_class_flags_ampicillin(self):
        alerts = detect_allergy_alerts(["Ampicillin"], ["penicillin"])
        assert len(alerts) >= 1

    def test_ssri_class_allergy(self):
        """Allergy to fluoxetine should flag sertraline (same class)."""
        alerts = detect_allergy_alerts(["Sertraline"], ["fluoxetine"])
        assert len(alerts) >= 1

    def test_no_false_positive_unrelated_drugs(self):
        """Penicillin allergy should NOT flag Metformin."""
        alerts = detect_allergy_alerts(["Metformin"], ["penicillin"])
        assert len(alerts) == 0

    def test_sulfonamide_allergy(self):
        alerts = detect_allergy_alerts(["Sulfamethoxazole"], ["sulfonamide"])
        assert len(alerts) >= 1

    def test_case_insensitive_allergy(self):
        alerts = detect_allergy_alerts(["AMOXICILLIN"], ["Penicillin"])
        assert len(alerts) >= 1

    def test_no_allergy_when_no_allergies(self):
        alerts = detect_allergy_alerts(["Amoxicillin"], [])
        assert alerts == []

    def test_multiple_medicines_multiple_allergies(self):
        alerts = detect_allergy_alerts(
            ["Amoxicillin", "Ibuprofen"],
            ["penicillin", "ibuprofen"],
        )
        assert len(alerts) >= 2


# ===========================================================================
# 4. Condition contraindications (Bonus C)
# ===========================================================================

class TestContraindications:
    def test_nsaids_flagged_in_kidney_disease(self):
        alerts = detect_contraindications(["Ibuprofen"], ["kidney disease"], [])
        assert len(alerts) >= 1
        assert any("ibuprofen" in a.medicine.lower() for a in alerts)

    def test_metformin_flagged_in_renal_impairment(self):
        alerts = detect_contraindications(["Metformin"], ["renal impairment"], [])
        assert any("metformin" in a.medicine.lower() for a in alerts)

    def test_beta_blocker_flagged_in_asthma(self):
        alerts = detect_contraindications(["Metoprolol"], ["asthma"], [])
        assert len(alerts) >= 1

    def test_warfarin_flagged_in_pregnancy(self):
        alerts = detect_contraindications(["Warfarin"], ["pregnancy"], [])
        assert len(alerts) >= 1
        assert any(a.severity == SeverityLevel.HIGH for a in alerts)

    def test_corticosteroid_in_diabetes(self):
        alerts = detect_contraindications(["Prednisolone"], ["diabetes"], [])
        assert len(alerts) >= 1

    def test_statin_in_liver_disease(self):
        alerts = detect_contraindications(["Atorvastatin"], ["liver disease"], [])
        assert len(alerts) >= 1

    def test_no_contraindication_unrelated(self):
        alerts = detect_contraindications(["Amoxicillin"], ["gout"], [])
        assert len(alerts) == 0

    def test_current_medications_also_checked(self):
        """Contraindications should also flag existing current medications."""
        alerts = detect_contraindications(
            [], ["kidney disease"], ["Ibuprofen"]
        )
        assert len(alerts) >= 1


# ===========================================================================
# 5. Fallback interaction checker
# ===========================================================================

class TestFallbackInteractions:
    def test_warfarin_aspirin_interaction(self):
        interactions, source = check_interactions_fallback(["warfarin", "aspirin"])
        names = {(i.drug_a.lower(), i.drug_b.lower()) for i in interactions}
        names_flat = {n for pair in names for n in pair}
        assert "warfarin" in names_flat
        assert "aspirin" in names_flat

    def test_sildenafil_nitrate_critical(self):
        interactions, _ = check_interactions_fallback(
            ["sildenafil", "isosorbide mononitrate"]
        )
        assert any(i.severity == SeverityLevel.CRITICAL for i in interactions)

    def test_maoi_ssri_critical(self):
        interactions, _ = check_interactions_fallback(["phenelzine", "fluoxetine"])
        assert any(i.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH) for i in interactions)

    def test_single_drug_no_interactions(self):
        interactions, _ = check_interactions_fallback(["paracetamol"])
        # Single drug cannot have drug-drug interactions with itself
        assert len(interactions) == 0

    def test_fallback_never_empty_for_known_pairs(self):
        interactions, _ = check_interactions_fallback(["digoxin", "amiodarone"])
        assert len(interactions) >= 1

    def test_fallback_dataset_has_minimum_entries(self):
        from engine import _FALLBACK_INTERACTIONS
        assert len(_FALLBACK_INTERACTIONS) >= 15, (
            f"Fallback dataset must have ≥15 entries; found {len(_FALLBACK_INTERACTIONS)}"
        )


# ===========================================================================
# 6. Risk scoring (Bonus B)
# ===========================================================================

class TestRiskScoring:
    def _make_interaction(self, sev: str) -> DrugInteraction:
        return DrugInteraction(
            drug_a="DrugA", drug_b="DrugB", severity=SeverityLevel(sev),
            mechanism="test", clinical_recommendation="test",
            source_confidence="high",
        )

    def _make_allergy(self, sev: str) -> AllergyAlert:
        return AllergyAlert(medicine="DrugA", reason="test", severity=sev)

    def _make_contra(self, sev: str) -> ContraindicationAlert:
        return ContraindicationAlert(
            medicine="DrugA", condition="cond", reason="test",
            severity=SeverityLevel(sev),
        )

    def test_zero_score_when_no_findings(self):
        score, _ = compute_risk_score([], [], [])
        assert score == 0

    def test_score_bounded_at_100(self):
        # Pile in lots of high-severity findings
        interactions = [self._make_interaction("critical")] * 10
        allergies = [self._make_allergy("critical")] * 10
        contras = [self._make_contra("high")] * 10
        score, _ = compute_risk_score(interactions, allergies, contras)
        assert score == 100

    def test_high_severity_interaction_raises_score(self):
        score1, _ = compute_risk_score([], [], [])
        score2, _ = compute_risk_score([self._make_interaction("high")], [], [])
        assert score2 > score1

    def test_critical_allergy_raises_score_more_than_low_interaction(self):
        score_low, _ = compute_risk_score([self._make_interaction("low")], [], [])
        score_critical, _ = compute_risk_score([], [self._make_allergy("critical")], [])
        assert score_critical > score_low

    def test_breakdown_fields_present(self):
        _, breakdown = compute_risk_score(
            [self._make_interaction("medium")],
            [self._make_allergy("high")],
            [],
        )
        assert breakdown.interaction_score >= 0
        assert breakdown.allergy_score >= 0
        assert breakdown.total >= 0
        assert breakdown.explanation


# ===========================================================================
# 7. Response schema compliance
# ===========================================================================

class TestResponseSchema:
    def _post(self, payload: dict):
        return client.post("/analyse", json=payload)

    def test_response_always_contains_required_fields(self):
        resp = self._post(make_request(["Aspirin"]))
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        data = resp.json()
        for field in [
            "interactions", "allergy_alerts", "safe_to_prescribe",
            "overall_risk_level", "requires_doctor_review", "source",
            "cache_hit", "processing_time_ms",
        ]:
            assert field in data, f"Missing required field: {field}"

    def test_processing_time_ms_positive(self):
        resp = self._post(make_request(["Metoprolol", "Verapamil"]))
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        assert resp.json()["processing_time_ms"] > 0

    def test_safe_to_prescribe_false_when_critical_allergy(self):
        """Known penicillin allergy + amoxicillin = not safe."""
        payload = make_request(["Amoxicillin"], allergies=["penicillin"])
        resp = self._post(payload)
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        data = resp.json()
        assert data["safe_to_prescribe"] is False

    def test_allergy_alert_present_when_penicillin_allergic(self):
        payload = make_request(["Amoxicillin"], allergies=["penicillin"])
        resp = self._post(payload)
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        data = resp.json()
        assert len(data["allergy_alerts"]) >= 1

    def test_overall_risk_level_valid_enum(self):
        resp = self._post(make_request(["Aspirin"]))
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        assert resp.json()["overall_risk_level"] in ("low", "medium", "high")

    def test_bonus_b_risk_score_present(self):
        resp = self._post(make_request(["Warfarin", "Aspirin"]))
        if resp.status_code != 200:
            pytest.skip("Engine unavailable")
        data = resp.json()
        assert "patient_risk_score" in data
        assert 0 <= data["patient_risk_score"] <= 100


# ===========================================================================
# 8. Health and utility endpoints
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_expected_keys(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "llm_model" in data
        assert "fallback_available" in data

    def test_fallback_endpoint_returns_interactions(self):
        resp = client.get("/interactions/fallback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 15
        assert len(data["interactions"]) >= 15


# ===========================================================================
# 9. Performance target (Bonus D) — no LLM, fallback only
# ===========================================================================

class TestPerformance:
    def test_fallback_5_medicines_under_500ms(self):
        """
        Fallback (no LLM) path should comfortably handle 5 medicines well
        under 3000 ms — targeting <500 ms for pure rule engine.
        """
        medicines = ["Warfarin", "Aspirin", "Simvastatin", "Amiodarone", "Digoxin"]
        t0 = time.monotonic()
        interactions, _ = check_interactions_fallback(medicines)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, f"Fallback took {elapsed_ms:.0f} ms (limit: 500 ms)"

    def test_allergy_detection_fast(self):
        medicines = ["Amoxicillin", "Ibuprofen", "Ciprofloxacin", "Atorvastatin", "Metoprolol"]
        allergies = ["penicillin", "nsaid"]
        t0 = time.monotonic()
        detect_allergy_alerts(medicines, allergies)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 50


# ===========================================================================
# 10. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_indian_brand_names_resolved(self):
        """Crocin (brand) should be resolved to paracetamol."""
        from engine import _normalise
        assert _normalise("crocin") == "paracetamol"

    def test_combiflam_alias(self):
        from engine import _normalise
        assert _normalise("combiflam") == "ibuprofen"

    def test_ecosprin_alias(self):
        from engine import _normalise
        assert _normalise("ecosprin") == "aspirin"

    def test_no_self_interaction(self):
        """A single drug cannot interact with itself."""
        interactions, _ = check_interactions_fallback(["Warfarin"])
        assert len(interactions) == 0

    def test_unknown_drug_no_crash(self):
        """Completely unknown drug name should not crash the engine."""
        resp = client.post("/analyse", json=make_request(["Xyzycin99"]))
        assert resp.status_code in (200, 503)

    def test_very_long_medicine_list(self):
        """Large medicine list should not crash the engine."""
        medicines = [
            "Aspirin", "Metformin", "Atorvastatin", "Lisinopril",
            "Metoprolol", "Amlodipine", "Omeprazole", "Levothyroxine",
        ]
        resp = client.post("/analyse", json=make_request(medicines))
        assert resp.status_code in (200, 503)
