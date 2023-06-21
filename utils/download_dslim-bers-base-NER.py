from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

DOWNLOAD_PATH = "./backend/app/model_resources/dslim-bert-base-NER/"
PATH_TO_MODEL = "dslim/bert-base-NER"

model = AutoModelForTokenClassification.from_pretrained(
    PATH_TO_MODEL, from_pretrained=True
)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL, from_pretrained=True)
classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)
classifier.save_pretrained(DOWNLOAD_PATH)
