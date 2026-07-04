"""
ASGI entrypoint.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from backend.core.app_factory import create_app

app = create_app()
