from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cache import cache
from engine import OLLAMA_BASE_URL, LLM_MODEL, analyse_drug_safety
from models import DrugSafetyRequest, DrugSafetyResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evodoc.main")

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("EvoDoc Drug Safety Engine starting up.")
    logger.info("LLM backend: %s  Model: %s", OLLAMA_BASE_URL, LLM_MODEL)
    yield
    logger.info("EvoDoc Drug Safety Engine shutting down.")


# ---------------------------------------------------------------------------
# FastAPI instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EvoDoc Clinical Drug Safety Engine",
    description=(
        "Medical-grade drug safety assessment for Indian clinics. "
        "Returns structured interaction warnings, allergy alerts, "
        "condition contraindications, and risk scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again."},
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/analyse",
    response_model=DrugSafetyResponse,
    summary="Analyse drug safety for a patient",
    description=(
        "Accept a list of proposed medicines and a patient's medical history. "
        "Returns drug interactions, allergy alerts, condition contraindications, "
        "and a 0–100 patient risk score."
    ),
    responses={
        200: {"description": "Successful safety analysis"},
        422: {"description": "Validation error (e.g. empty medicine list, negative age)"},
        503: {"description": "Engine temporarily unavailable"},
    },
)
async def analyse(request: DrugSafetyRequest) -> DrugSafetyResponse:
    t0 = time.monotonic()
    try:
        result = await analyse_drug_safety(request)
    except Exception as exc:
        logger.error("Engine failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drug safety engine encountered an error. Please retry.",
        ) from exc

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        "Analysed %d medicine(s) in %d ms | risk=%s | cache_hit=%s | source=%s",
        len(request.proposed_medicines),
        elapsed,
        result.overall_risk_level,
        result.cache_hit,
        result.source,
    )
    return result


@app.get(
    "/health",
    summary="Health check",
    description="Returns service status and LLM connectivity.",
)
async def health():
    llm_ok = False
    llm_detail = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                llm_ok = any(LLM_MODEL in m for m in models)
                llm_detail = "available" if llm_ok else f"model '{LLM_MODEL}' not found; pull it first"
    except Exception as exc:
        llm_detail = str(exc)

    return {
        "status": "ok",
        "llm_backend": OLLAMA_BASE_URL,
        "llm_model": LLM_MODEL,
        "llm_status": llm_detail,
        "fallback_available": True,
        "cache_backend": getattr(cache, "backend", "unknown"),
    }


@app.get(
    "/cache/stats",
    summary="Cache statistics",
)
async def cache_stats():
    return cache.stats()


@app.get(
    "/interactions/fallback",
    summary="List all fallback interaction rules",
    description=(
        "Returns the full hardcoded drug interaction dataset used when the LLM is unavailable. "
        "Useful for transparency and auditing."
    ),
)
async def list_fallback_interactions():
    from engine import _FALLBACK_INTERACTIONS
    return {
        "count": len(_FALLBACK_INTERACTIONS),
        "interactions": _FALLBACK_INTERACTIONS,
    }
