from pydantic import BaseModel, HttpUrl, validator
from typing import Optional

from typing import Sequence
import app.models.entity as model_entity


class EntityBase(BaseModel):
    class Meta:
        orm_model = model_entity.Entity


class EntityCreate(EntityBase):
    source_id: int
    entity: str
    count: int


class EntityUpdate(EntityBase):
    entity: Optional[str] = None
    count: Optional[int] = None


# Properties shared by models stored in DB
class EntityInDBBase(EntityBase):
    entity: str
    count: int

    class Config:
        orm_mode = True


# Properties to return to client
class Entity(EntityInDBBase):
    pass


# Properties properties stored in DB
class EntityInDB(EntityInDBBase):
    id: int
