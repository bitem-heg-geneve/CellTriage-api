from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, validates
from app.db.base_class import Base
from app.models.article import Article
from datetime import datetime


class Job(Base):
    id = Column(Integer, primary_key=True, index=True)
    use_fulltext = Column(Boolean, default=True)
    status = Column(String, default="pending")
    process_time = Column(Float, default=0.0)
    job_created_at = Column(DateTime, default=datetime.now())
    process_start_at = Column(DateTime, nullable=True)
    process_end_at = Column(DateTime, nullable=True)
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