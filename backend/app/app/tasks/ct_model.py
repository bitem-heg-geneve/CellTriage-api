import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd

# from torchmetrics import AUROC

from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import pytorch_lightning as pl
import gc


class CtDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        labels,
        text_col: str,
        tokenizer,
        max_token_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_col = text_col
        self.labels = labels
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row[self.text_col]
        labels = data_row[self.labels]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )


class CtDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        text_col,
        tokenizer,
        labels,
        # batch_size=8,
        batch_size=200,
        max_token_len=512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.text_col = text_col
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = CtDataset(
            self.train_df,
            self.labels,
            self.text_col,
            self.tokenizer,
            self.max_token_len,
        )

        self.test_dataset = CtDataset(
            self.test_df, self.labels, self.text_col, self.tokenizer, self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)


class CtModel(pl.LightningModule):
    def __init__(
        self,
        lm_model_name,
        labels,
        n_training_steps=None,
        n_warmup_steps=None,
        dropout=0.1,
    ):
        super().__init__()
        self.lm = BertModel.from_pretrained(lm_model_name, return_dict=True)
        self.fc1 = nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.lm.config.hidden_size, self.lm.config.hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.lm.config.hidden_size, len(labels))
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.outcomes = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.lm(input_ids, attention_mask=attention_mask)
        output = self.fc1(output.pooler_output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.dropout2(output)
        output = self.classifier(output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, self.outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": self.outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_training_epoch_end(self):
        labels = []
        predictions = []
        for output in self.outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(labels):
            class_roc_auc = AUROC(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
            )

        self.outcomes.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


class CtTagger(object):
    def __init__(self, chkpt_fp, labels, text_col, lm_model_name, max_token_count=512):
        self.labels = labels
        self.max_token_count = max_token_count
        self.text_col = text_col
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CtModel.load_from_checkpoint(
            chkpt_fp,
            labels=labels,
            lm_model_name=lm_model_name,
            map_location=self.device,
        )
        self.model.eval()
        self.model.freeze()
        # self.model = self.model.to(self.device)

    def __del__(self):
        del self.tokenizer
        del self.model

    def predict(self, dataframe):
        dataset = CtDataset(
            data=dataframe,
            tokenizer=self.tokenizer,
            text_col=self.text_col,
            max_token_len=self.max_token_count,
            labels=self.labels,
        )
        model = self.model.to(self.device)
        predictions, labels = [], []
        dataloader = DataLoader(dataset, batch_size=400, num_workers=4)
        for batch in dataloader:
            with torch.no_grad():
                _, prediction = model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                predictions.append(prediction.flatten().cpu())
                labels.append(batch["labels"].int().cpu())
                del _
                del prediction
                torch.cuda.empty_cache()
                gc.collect()

        # predictions = torch.stack(predictions).detach().cpu()
        # predictions = torch.stack(predictions)
        # labels = torch.stack(labels)
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return predictions, labels

        # for item in tqdm(dataset):
        #     _, prediction = self.model(
        #         item["input_ids"].unsqueeze(dim=0).to(self.device),
        #         item["attention_mask"].unsqueeze(dim=0).to(self.device),
        #     )
        #     predictions.append(prediction.flatten())
        #     labels.append(item["labels"].int())

        #     predictions = torch.stack(predictions).detach().cpu()
        #     labels = torch.stack(labels).detach().cpu()
        # return predictions, labels

        # predictions, labels = [], []
        # dataloader = DataLoader(dataset, batch_size=10, num_workers=8)
        # predictor = pl.Trainer(
        #     accelerator="auto", enable_checkpointing=False, logger=False
        # )
        # predictions = predictor.predict(self.model, dataloaders=dataloader)

        # return predictions, labels

        # predictions, labels = [], []
        # for item in tqdm(dataset):
        #     _, prediction = self.model(
        #         item["input_ids"].unsqueeze(dim=0).to(self.device),
        #         item["attention_mask"].unsqueeze(dim=0).to(self.device),
        #     )
        #     predictions.append(prediction.flatten())
        #     labels.append(item["labels"].int())

        # predictions = torch.stack(predictions).detach().cpu()
        # labels = torch.stack(labels).detach().cpu()
        # return predictions, labels
