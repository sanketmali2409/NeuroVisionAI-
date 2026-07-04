"""
Project-specific exceptions and their FastAPI exception handlers.

Routers/services should raise these instead of generic `HTTPException`
where a domain-specific error exists, so error responses stay consistent
across the whole API and calling code can distinguish failure modes
programmatically (e.g. `except ModelNotLoadedError`).
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from config.logging_config import get_logger

logger = get_logger(__name__)


class NeuroVisionError(Exception):
    """Base class for all application-specific errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        if detail:
            self.detail = detail
        super().__init__(self.detail)


class ModelNotLoadedError(NeuroVisionError):
    """Raised when an inference request arrives before a required model
    artifact has been loaded (e.g. missing .keras file on disk)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "The requested model is not available yet."


class InvalidImageError(NeuroVisionError):
    """Raised when an uploaded file fails image validation (bad format,
    corrupted data, unsupported MIME type)."""

    status_code = status.HTTP_400_BAD_REQUEST
    detail = "The uploaded file is not a valid image."


class ResourceNotFoundError(NeuroVisionError):
    """Raised when a requested DB record (patient, report, prediction) does
    not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    detail = "The requested resource was not found."


class AuthenticationError(NeuroVisionError):
    """Raised for invalid credentials or expired/invalid JWTs."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Authentication failed."


class AuthorizationError(NeuroVisionError):
    """Raised when an authenticated user lacks the required role."""

    status_code = status.HTTP_403_FORBIDDEN
    detail = "You do not have permission to perform this action."


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every `NeuroVisionError` subclass returns a
    consistent JSON error shape: `{"error": "<ClassName>", "detail": "..."}`.
    """

    @app.exception_handler(NeuroVisionError)
    async def _handle_neurovision_error(request: Request, exc: NeuroVisionError) -> JSONResponse:
        logger.warning("%s on %s %s: %s", type(exc).__name__, request.method, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "detail": "An unexpected error occurred."},
        )
