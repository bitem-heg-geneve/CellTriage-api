from typing import Union

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.article import Article
from app.schemas.article import ArticleCreate, ArticleUpdate
from fastapi.encoders import jsonable_encoder
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from app.crud.base import ModelType ,CreateSchemaType

class CRUDArticle(CRUDBase[Article, ArticleCreate, ArticleUpdate]):
    pass

article = CRUDArticle(Article)
