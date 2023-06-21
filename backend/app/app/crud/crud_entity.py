from typing import Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.entity import Entity
from app.schemas.entity import EntityCreate, EntityUpdate


class CRUDEntity(CRUDBase[Entity, EntityCreate, EntityUpdate]):
    pass


entity = CRUDEntity(Entity)
