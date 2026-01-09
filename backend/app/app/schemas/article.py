from pydantic import BaseModel, HttpUrl, validator
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


# Properties shared by models stored in DB
class ArticleInDBBase(ArticleBase):
    pmid: int
    score: Optional[float] = None

    class Config:
        orm_mode = True

    @validator("score")
    def is_check(cls, v):
        if v:
            return round(v, 2)
        else:
            return v


# Properties to return to client
class Article(ArticleInDBBase):
    pmcid: Optional[str] = None
    entrez_date: Optional[str] = None
    score: Optional[float] = None
    text_source: Optional[str] = None
    text: Optional[str] = None


# Properties properties stored in DB
class ArticleInDB(ArticleInDBBase):
    None
