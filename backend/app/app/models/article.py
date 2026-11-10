from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, Date

from app.db.base_class import Base
from sqlalchemy.orm import relationship


class Article(Base):
    id = Column(Integer, primary_key=True, index=True)
    pmid = Column(Integer, nullable=False)
    pmcid = Column(String, nullable=True)
    entrez_date = Column(String, nullable=True)
    text_source = Column(String, nullable=True)
    text = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    # Shared triage guard: what the SIBiLS fetch gave for this PMID
    # (ok | missing | empty | fetch_error) and why there is no score
    # (missing | empty | fetch_error | no_prediction). score is NULL, never 0, without text.
    text_status = Column(String, nullable=True)
    error = Column(String, nullable=True)
    job_id = Column(Integer, ForeignKey("job.id"))
    job = relationship("Job", back_populates="article_set")
