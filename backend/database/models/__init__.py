"""
Aggregates all ORM model modules so a single `import backend.database.models`
registers every table on `Base.metadata`.

As each feature module is built, its model(s) get their own file here
(e.g. `prediction.py`, `report.py`, `prognosis.py`) and are re-exported
below - keeping one import path for the rest of the app while each model
still lives in a focused, single-responsibility file.
"""

from backend.database.models.user import User
from backend.database.models.patient import Patient

__all__ = ["User", "Patient"]
