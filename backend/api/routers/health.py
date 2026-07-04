"""
Health and readiness endpoints.

`/health` is a liveness check (is the process up). `/health/ready` is a
readiness check that will, as modules are added, verify that required
model artifacts are actually loaded on disk before reporting ready — useful
for Docker/Compose healthchecks and load balancers.
"""

from __future__ import annotations

from fastapi import APIRouter

from config.settings import settings

router = APIRouter()


@router.get("/health", summary="Liveness check")
def liveness() -> dict[str, str]:
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}


@router.get("/health/ready", summary="Readiness check")
def readiness() -> dict[str, object]:
    """Reports which model artifacts are currently present on disk.

    As each module (classification, segmentation, ...) is implemented, its
    expected model path is added to this dict so operators can see at a
    glance what's actually deployable versus still pending training.
    """
    checks = {
        "classification_model": (settings.MODELS_DIR / "classification" / settings.CLASSIFICATION_MODEL_NAME).exists(),
        "segmentation_model": (settings.MODELS_DIR / "segmentation" / settings.SEGMENTATION_MODEL_NAME).exists(),
        "multimodal_model": (settings.MODELS_DIR / "multimodal" / settings.MULTIMODAL_MODEL_NAME).exists(),
        "prognosis_model": (settings.MODELS_DIR / "prognosis" / settings.PROGNOSIS_MODEL_NAME).exists(),
    }
    return {"status": "ok", "models": checks}
