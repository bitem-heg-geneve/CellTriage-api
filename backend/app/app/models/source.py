from sqlalchemy import Column, Integer, String, ForeignKey, Float

from app.db.base_class import Base
from sqlalchemy.orm import relationship


class Source(Base):
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String)
    text = Column(String)
    impaakt_score = Column(Float, nullable=True)
    job_id = Column(Integer, ForeignKey("job.id"))
    job = relationship("Job", back_populates="sources")
    NER = relationship(
        "Entity",
        lazy="joined",
        cascade="all,delete-orphan",
        back_populates="source",
        uselist=True,
    )
    companies = relationship(
        "Company",
        lazy="joined",
        cascade="all,delete-orphan",
        back_populates="source",
        uselist=True,
    )
