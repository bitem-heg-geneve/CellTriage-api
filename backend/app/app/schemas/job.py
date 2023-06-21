from pydantic import BaseModel, HttpUrl, Field, validator

from typing import Sequence, List, Union
from datetime import datetime
from app.schemas.source import Source
from fastapi import Query
from typing import Optional, Dict
from datetime import datetime

# from app.models.source import Source
from app.schemas.source import SourceCreate
from pydantic import root_validator


class JobBase(BaseModel):
    pass


class JobCreate(JobBase):
    impaakt_ranking: bool = Field(default=True)
    named_entity_recognition: bool = Field(default=True)
    company_classification: bool = Field(default=True)
    sources: List[SourceCreate] = Field(
        example=[
            SourceCreate(url="https://www.occ.gov/news-issuances/news-releases/2011/nr-occ-2011-47c.pdf"),
            SourceCreate(
                url="https://www.pionline.com/esg/dws-sell-excessive-greenwashing-doubt-citi-analysts-say"
            ),
            SourceCreate(
                url="https://www.cnn.com/markets/fear-and-greed"
            ),
            SourceCreate(
                url="https://time.com/personal-finance/article/how-many-stocks-should-i-own/"
            ),
            SourceCreate(
                url="https://wallethub.com/answers/cc/citibank-credit-balance-refund-2140740558/"
            ),
            SourceCreate(
                url="https://www.cnn.com/2021/02/16/business/citibank-revlon-lawsuit-ruling/index.html"
            ),
            SourceCreate(
                url="https://www.businessinsider.com/citi-analysts-excessive-corporate-leverage-2018-11/"
            ),
            SourceCreate(
                url="https://en.wikipedia.org/wiki/Citibank"
            ),
            SourceCreate(
                url="https://www.propublica.org/article/citi-execs-deeply-sorry-but-dont-blame-us2"
            ),
            SourceCreate(
                url="https://www.cnbc.com/2023/01/11/citi-names-two-asset-classes-to-deploy-excess-cash-for-higher-returns-.html"
            ),
            SourceCreate(url="https://www.mckinsey.com/industries/financial-services/our-insights/global-banking-annual-review"),
        ],
        default=[],
    )


class JobUpdate(JobBase):
    status: Optional[str] = None
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None


# Properties shared by models stored in DB
class JobInDBBase(JobBase):
    id: int
    impaakt_ranking: bool = Field(default=True)
    named_entity_recognition: bool = Field(default=True)
    company_classification: bool = Field(default=True)
    status: str = Field(default="pending")
    job_created_at: datetime = Field(default=datetime.now())
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    process_time: int = 0

    class Config:
        orm_mode = True

    @root_validator
    def compute_process_time(cls, values) -> Dict:
        process_start_at = values.get("process_start_at")
        process_end_at = values.get("process_end_at")
        if process_start_at:
            if process_end_at:
                values["process_time"] = round(
                    (process_end_at - process_start_at).total_seconds(), 2
                )
            else:
                values["process_time"] = round(
                    (datetime.now() - process_start_at).total_seconds(), 2
                )
        else:
            values["process_time"] = 0.00
        return values

    # @validator("process_time")
    # def pt_check(cls, v):
    #     return round(v, 2)


# Properties to return to client
class Job(JobInDBBase):
    sources: List[Source]


# Status to return to client
class JobStatus(JobInDBBase):
    pass


# Properties properties stored in DB
class JobInDB(JobInDBBase):
    process_start_at: Optional[datetime] = None
    process_end_at: Optional[datetime] = None
    pass
