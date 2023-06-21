from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, validates
from app.db.base_class import Base
from app.models.source import Source
from datetime import datetime


class Job(Base):
    id = Column(Integer, primary_key=True, index=True)
    impaakt_ranking = Column(Boolean, default=True)
    named_entity_recognition = Column(Boolean, default=True)
    company_classification = Column(Boolean, default=True)
    status = Column(String, default="pending")
    process_time = Column(Float, default=0.0)
    job_created_at = Column(DateTime, default=datetime.now())
    process_start_at = Column(DateTime, nullable=True)
    process_end_at = Column(DateTime, nullable=True)
    sources = relationship(
        "Source",
        lazy="joined",
        cascade="all,delete-orphan",
        back_populates="job",
        uselist=True,
    )

    @validates("sources")
    def adjust_sources(self, _, s) -> Source:
        """Instantiate nestedSource object"""
        if s and isinstance(s, dict):
            return Source(**s)
