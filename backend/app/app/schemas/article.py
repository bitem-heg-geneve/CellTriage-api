from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, List

from typing import Sequence
import app.models.article as model_article


class ArticleBase(BaseModel):
    class Meta:
        orm_model = model_article.Article


class ArticleCreate(ArticleBase):
    pmid: int


class ArticleUpdate(ArticleBase):
    pmcid: Optional[str] = None
    entrez_date: Optional[str] = None
    score: Optional[float] = None
    text_source: Optional[str] = None
    text: Optional[str] = None
    text_status: Optional[str] = None
    error: Optional[str] = None


class ArticleInDBBase(ArticleBase):
    pmid: int
    score: Optional[float] = None

    model_config = {"from_attributes": True}

    @field_validator("score")
    @classmethod
    def is_check(cls, v):
        if v is not None:
            return round(v, 2)
        return v


class Article(ArticleInDBBase):
    pmcid: Optional[str] = None
    entrez_date: Optional[str] = None
    # null when the article has no score; `error` then says why
    # (missing | empty | fetch_error | no_prediction). Never 0 for "no text".
    score: Optional[float] = None
    error: Optional[str] = None
    text_status: Optional[str] = None
    text_source: Optional[str] = None
    text: Optional[str] = None


class ArticleInDB(ArticleInDBBase):
    None
