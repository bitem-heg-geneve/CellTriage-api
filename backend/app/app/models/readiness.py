from sqlalchemy import Column, String, Boolean, DateTime, Text

from app.db.base_class import Base


class Readiness(Base):
    """Last outcome of `python -m app.readiness` per process (id = "worker-infer"),
    read by GET /api/v1/readyz. Shared triage guard, §4."""

    id = Column(String, primary_key=True)
    ok = Column(Boolean, nullable=False, default=False)
    checked_at = Column(DateTime, nullable=False)
    error = Column(String, nullable=True)
    detail = Column(Text, nullable=True)  # JSON: model and SIBiLS measurements
