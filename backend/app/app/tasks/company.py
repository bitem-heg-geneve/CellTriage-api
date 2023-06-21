from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch, os, json, pandas as pd

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
from dataclasses import dataclass
from difflib import SequenceMatcher
from celery import shared_task
from app.db.session import SessionLocal
from app import crud
from app.schemas.company import CompanyCreate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CompanyCandidateData(torch.utils.data.dataset.Dataset):
    def __init__(self, urls_NER):
        self.urls_NER = urls_NER
        NEs_set = set()
        for url in self.urls_NER.keys():
            for NE in self.urls_NER[url].keys():
                NEs_set.add(NE)

        self.example = list(NEs_set)

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        return self.example[i]


class CompanyData(torch.utils.data.dataset.Dataset):
    def __init__(self, company_list):
        self.company_list = company_list

    def __len__(self):
        return len(self.company_list)

    def __getitem__(self, i):
        return self.company_list[i]


@dataclass
class CompanyCollator:
    def __init__(self, PATH_TO_MODEL="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)

    def __call__(self, batch):
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")


@dataclass
class CompanyCandidateCollator:
    def __init__(self, PATH_TO_MODEL="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)

    def __call__(self, batch):
        return batch, self.tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )


class CompanyScore(object):
    """
    Compute matching ratio & distances and allocate scores to company with respect
        to a list of company candidates.
        This should be used each time there are new companies or each time NEs have been run on a batch
    """

    def __init__(
        self,
        urls_NER,
        company_list,
        batch_size,
        PATH_TO_MODEL="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.batch_size = batch_size
        self.company_data = CompanyData(company_list)
        self.company_collator = CompanyCollator(PATH_TO_MODEL)
        self.sampler = torch.utils.data.sampler.SequentialSampler(self.company_data)
        self.company_dataloader = torch.utils.data.dataloader.DataLoader(
            self.company_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.company_collator,
        )

        self.company_candidate_data = CompanyCandidateData(urls_NER)
        self.company_candidate_collator = CompanyCandidateCollator()
        self.sampler = torch.utils.data.sampler.SequentialSampler(
            self.company_candidate_data
        )
        self.company_candidate_dataloader = torch.utils.data.dataloader.DataLoader(
            self.company_candidate_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.company_candidate_collator,
        )

        self.load()
        self.company2vec = []
        for batch in self.company_dataloader:
            self.company2vec.append(self.get_vec_representation(batch))
        self.company2vec = torch.cat(self.company2vec)

    def load(self, PATH_TO_MODEL="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(PATH_TO_MODEL).to(device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_vec_representation(self, batch):
        with torch.no_grad():
            batch = batch.to(device)
            tmp = self.model(**batch)
            return F.normalize(
                self.mean_pooling(tmp, batch["attention_mask"]), p=2, dim=1
            )

    def compute_distances(self, NER2vec):
        return torch.cat(
            [cos(self.company2vec, NER_vec).reshape(1, -1) for NER_vec in NER2vec]
        )

    def compute_seq_matcher_ratio(self, batch):
        seq_matcher_ratio = []
        for ne in batch:
            seq_matcher_ratio.append(
                [
                    SequenceMatcher(None, ne, e).ratio()
                    for e in self.company_dataloader.dataset
                ]
            )
        return seq_matcher_ratio

    def apply_NE_weighting_formula(self, weight):
        weight["company_distance"][weight["company_distance"] < 0] = 0
        tmp = torch.tensor(weight["company_distance"] < 0.5, dtype=int) + torch.tensor(
            weight["seq_matcher_ratio"] < 0.5, dtype=int
        )
        weight["company_distance"] += weight["seq_matcher_ratio"]
        weight["company_distance"][tmp == 2] = 0
        weight["company_distance"] /= 2
        weight["company_distance"][weight["company_distance"] < 0.5] = 0
        return weight["company_distance"]

    def apply_URL_weighting_formula(self, url_NER, scores):
        NER_count = sum(url_NER.values())
        url_score = 0
        for NE in url_NER.keys():
            url_score += scores[NE] * url_NER[NE]
        return url_score / NER_count

    def score_vector2company(self, scores, companies, top_k):
        return {
            c: s
            for s, c in sorted(zip(scores, companies), reverse=True)[:top_k]
            if s > 0
        }

    def infer(self, top_k=10):
        scores = {}
        for [text, batch] in self.company_candidate_dataloader:
            seq_matcher_ratio = self.compute_seq_matcher_ratio(text)
            company_distance = self.compute_distances(
                self.get_vec_representation(batch)
            )
            for t, m, d in zip(text, seq_matcher_ratio, company_distance):
                scores[t] = self.apply_NE_weighting_formula(
                    {
                        "seq_matcher_ratio": torch.tensor(m).to(device),
                        "company_distance": d,
                    }
                )

        url_company_scores = {}
        for url in self.company_candidate_dataloader.dataset.urls_NER.keys():
            s = self.apply_URL_weighting_formula(
                self.company_candidate_dataloader.dataset.urls_NER[url], scores
            )
            tmp = self.score_vector2company(
                s.tolist(), self.company_dataloader.dataset.company_list, top_k
            )
            if tmp:
                url_company_scores[url] = tmp

        return url_company_scores


@shared_task(
    name="torch:job_company",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_company(self, job_id, batch_size=64):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)

    COMPANY_LIST_FP = "/app/model_resources/companies.csv"
    company_list = pd.read_csv(COMPANY_LIST_FP)
    company_list = list(company_list["sorted_entity"])

    sources = [source for source in job.sources if len(source.NER) > 0]

    dummy_json = {}
    for source in sources:
        dummy_json[source.url] = {e.entity: e.count for e in source.NER}

    model = CompanyScore(
        urls_NER=dummy_json, company_list=company_list, batch_size=batch_size
    )
    infer = model.infer()

    for source in sources:
        if source.url in infer.keys():
            for e in infer[source.url].items():
                company_create = CompanyCreate(
                    source_id=source.id, company=e[0], score=e[1] * 100
                )
                crud.company.create(db=db, obj_in=company_create)

    db.close()
    return job_id
