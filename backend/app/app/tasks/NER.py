from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch, random, nltk, os

from dataclasses import dataclass
from app.db.session import SessionLocal
from app import crud
from app.schemas.entity import EntityCreate
from app import crud
from celery import shared_task


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import json


class NERData(torch.utils.data.dataset.Dataset):
    def __init__(self, json_docs, NER_max_sentences=100, NER_min_char_sentence=20):
        self.example = []
        self.sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        abbreviation = [
            "a",
            "å",
            "Ǻ",
            "Å",
            "b",
            "c",
            "d",
            "e",
            "ɛ",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "Ö",
            "Ø",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "µm",
            "abs",
            "al",
            "approx",
            "bp",
            "ca",
            "cap",
            "cf",
            "co",
            "d.p.c",
            "dr",
            "e.g",
            "et",
            "etc",
            "er",
            "eq",
            "fig",
            "figs",
            "h",
            "i.e",
            "it",
            "inc",
            "min",
            "ml",
            "mm",
            "mol",
            "ms",
            "no",
            "nt",
            "ref",
            "r.p.m",
            "sci",
            "s.d",
            "sd",
            "sec",
            "s.e.m",
            "sp",
            "ssp",
            "st",
            "supp",
            "vs",
            "wt",
        ]
        self.sentence_tokenizer._params.abbrev_types.update(abbreviation)
        random.seed(1234)
        i = 0
        for doc in json_docs:
            sentences = [
                s
                for s in self.text_preprocessing(doc["text"])
                if len(s) > NER_min_char_sentence
            ]
            if not sentences:
                continue
            # here we just sample down the number of sentences
            sentences = random.sample(
                sentences, min([len(sentences), NER_max_sentences])
            )
            for s in sentences:
                self.example.append(
                    {"input_ids": {"sentence": s, "id": i, "url": doc["url"]}}
                )
                i += 1

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        return self.example[i]

    def text_preprocessing(self, text):
        text = " ".join(text.split())
        return [t for t in self.sentence_tokenizer.tokenize(text)]


@dataclass
class NERCollator:
    def __init__(self, PATH_TO_MODEL="dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)

    def __call__(self, batch):
        tmp = self.tokenizer(
            [i["input_ids"]["sentence"] for i in batch],
            add_special_tokens=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        offset_mapping = tmp["offset_mapping"]
        from_list, to_list = [], []
        for i in range(len(offset_mapping)):
            f, t = zip(*offset_mapping[i])
            from_list.append(list(f))
            to_list.append(list(t))
        del tmp["offset_mapping"]
        tmp = self.tokenizer.pad(
            tmp, padding="max_length", max_length=512, return_tensors="pt"
        )
        offset_mapping_from = (
            torch.tensor([t + [-100] * (512 - len(t)) for t in from_list])
            .reshape([-1, 512])
            .tolist()
        )
        offset_mapping_to = (
            torch.tensor([t + [-100] * (512 - len(t)) for t in to_list])
            .reshape([-1, 512])
            .tolist()
        )
        sentences = [i["input_ids"]["sentence"] for i in batch]
        urls = [i["input_ids"]["url"] for i in batch]

        return (
            [i["input_ids"]["id"] for i in batch],
            tmp,
            offset_mapping_from,
            offset_mapping_to,
            sentences,
            urls,
        )


class NER_model(object):
    """
    NER model take a list of text and output a list of potential company NEs.
    It should be used each time new documents are added to the index
    """

    def __init__(
        self,
        json_docs,
        batch_size,
        NER_max_sentences=100,
        NER_min_char_sentence=20,
        PATH_TO_MODEL="dslim/bert-base-NER",
    ):
        self.batch_size = batch_size
        self.load(PATH_TO_MODEL)
        self.NER_data = NERData(json_docs, NER_max_sentences, NER_min_char_sentence)
        self.NER_collator = NERCollator(PATH_TO_MODEL)
        self.sampler = torch.utils.data.sampler.SequentialSampler(self.NER_data)
        self.NER_dataloader = torch.utils.data.dataloader.DataLoader(
            self.NER_data,
            batch_size=self.batch_size,
            sampler=self.sampler,
            collate_fn=self.NER_collator,
        )

    def load(self, PATH_TO_MODEL="dslim/bert-base-NER"):
        self.model = AutoModelForTokenClassification.from_pretrained(PATH_TO_MODEL).to(
            device
        )

    def infer(self):
        urls_NER, NEs_set, index_duplicate = {}, set(), set()
        for batch in self.NER_dataloader:
            with torch.no_grad():
                (
                    ids,
                    batch,
                    offset_mapping_from,
                    offset_mapping_to,
                    sentences,
                    urls,
                ) = batch
                batch.to(device)
                preds = self.model(**batch)
                preds = torch.nn.functional.softmax(preds["logits"], dim=2)
                preds = torch.argmax(preds, dim=2)

            for example_id, o, s_tokens_ids, s_attention_mask, f, t, s, url in zip(
                ids,
                preds,
                batch["input_ids"],
                batch["attention_mask"],
                offset_mapping_from,
                offset_mapping_to,
                sentences,
                urls,
            ):
                if example_id not in index_duplicate:
                    index_duplicate.add(example_id)
                else:
                    continue

                if (
                    self.model.config.label2id["B-ORG"] in o
                    or self.model.config.label2id["I-ORG"] in o
                ):
                    offset_mapping = [
                        (from_, to_)
                        for from_, to_ in zip(f, t)
                        if from_ != -100 or to_ != -100
                    ]
                    s_tokens_ids = [x for x in s_tokens_ids if x != 0]
                    s_attention_mask = [x for x in s_attention_mask if x != 0]
                    o = o[: len(s_attention_mask)]
                    s_tokens_ids = s_tokens_ids[: len(s_attention_mask)]
                    s_is_subword = []
                    for w in self.NER_collator.tokenizer.convert_ids_to_tokens(
                        s_tokens_ids
                    ):
                        if "##" in w:
                            s_is_subword.append(1)
                        else:
                            s_is_subword.append(0)

                    B_ORG = torch.nonzero(o == self.model.config.label2id["B-ORG"])
                    for i, b_org in enumerate(B_ORG):
                        tmp = []
                        max_range = B_ORG[i + 1] if len(B_ORG) > i + 1 else len(o)
                        for idx in range(b_org, max_range):
                            if (
                                o[idx]
                                in [
                                    self.model.config.label2id["B-ORG"],
                                    self.model.config.label2id["I-ORG"],
                                ]
                                or s_is_subword[idx]
                            ):
                                tmp.append(offset_mapping[idx])

                            else:
                                break

                        NE = s[tmp[0][0] : tmp[-1][1]]
                        if NE == "":
                            continue
                        if url not in urls_NER:
                            urls_NER[url] = {}
                        if NE not in urls_NER[url]:
                            urls_NER[url][NE] = 1
                        else:
                            urls_NER[url][NE] += 1

        return urls_NER


@shared_task(
    name="torch:job_ner",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_ner(self, job_id, batch_size=64):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)

    sources = [source for source in job.sources if source.text]
    dummy_json = [{"url": source.url, "text": source.text} for source in sources]
    model = NER_model(json_docs=dummy_json, batch_size=batch_size)
    infer = model.infer()
    for source in sources:
        if source.url in infer.keys():
            for e in infer[source.url].items():
                entity_create = EntityCreate(
                    source_id=source.id, entity=e[0], count=e[1]
                )
                crud.entity.create(db=db, obj_in=entity_create)

    db.close()
    return job_id
