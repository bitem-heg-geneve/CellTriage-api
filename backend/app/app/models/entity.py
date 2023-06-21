from sqlalchemy import Column, Integer, String, ForeignKey, Float

from app.db.base_class import Base
from sqlalchemy.orm import relationship


class Entity(Base):
    id = Column(Integer, primary_key=True, index=True)
    entity = Column(String)
    count = Column(Integer)
    source_id = Column(Integer, ForeignKey("source.id"))
    source = relationship("Source", back_populates="NER")
