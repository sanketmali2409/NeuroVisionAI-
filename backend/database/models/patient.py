"""Patient table: demographic + clinical metadata shared by multiple modules
(multimodal fusion, prognosis, report generation)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.session import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    age: Mapped[int] = mapped_column(nullable=False)
    sex: Mapped[str] = mapped_column(String(10), nullable=False)
    symptoms: Mapped[str | None] = mapped_column(String(500), nullable=True)
    medical_history: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    tumor_size_mm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Which clinician/researcher account created this patient record.
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<Patient id={self.id} full_name={self.full_name!r} age={self.age}>"
