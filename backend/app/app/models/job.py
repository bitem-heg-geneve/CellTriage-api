from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, validates
from app.db.base_class import Base
from app.models.article import Article
from app.services.job_summary import article_row, summarize
from datetime import datetime


class Job(Base):
    id = Column(Integer, primary_key=True, index=True)
    use_fulltext = Column(Boolean, default=True)
    status = Column(String, default="pending")
    process_time = Column(Float, default=0.0)
    # callable, not datetime.now(): the latter was evaluated once at import and
    # froze job_created_at at container start for every job
    job_created_at = Column(DateTime, default=datetime.now)
    process_start_at = Column(DateTime, nullable=True)
    process_end_at = Column(DateTime, nullable=True)
    # Shared triage guard: sibils_fetch | no_prediction when the job failed, and
    # what the SIBiLS instance said for this job
    error = Column(String, nullable=True)
    sibils_version = Column(String, nullable=True)
    collection_version = Column(String, nullable=True)
    sibils_warning = Column(String, nullable=True)
    article_set = relationship(
        "Article",
        lazy="joined",
        back_populates="job",
        uselist=True,
    )

    @validates("article_set")
    def adjust_articles(self, _, s) -> Article:
        """Instantiate nested Article object"""
        if s and isinstance(s, dict):
            return Article(**s)
            # return

    # n_* counts for the API (schemas read them through from_attributes),
    # computed from the articles so they are never stale.
    @property
    def summary(self):
        return summarize(article_row(a) for a in self.article_set)

    @property
    def n_submitted(self) -> int:
        return self.summary.n_submitted

    @property
    def n_scored(self) -> int:
        return self.summary.n_scored

    @property
    def n_no_text(self) -> int:
        return self.summary.n_no_text

    @property
    def n_fetch_error(self) -> int:
        return self.summary.n_fetch_error

    @property
    def n_no_prediction(self) -> int:
        return self.summary.n_no_prediction
