from datetime import datetime
from typing import Optional
import pytorch_lightning as pl
from transformers import AutoConfig,AutoModelForSequenceClassification
import datasets
import torch
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MyData(torch.utils.data.Dataset):
    def __init__(self, encoding, labels):
        self.ids = encoding['input_ids']
        self.mask = encoding['attention_mask']
        self.labels= labels['input_ids']
    def __getitem__(self, idx):
      item={}
      item['input_ids'] = torch.tensor(self.ids[idx]).to(device)
      item['attention_mask'] = torch.tensor(self.mask[idx]).to(device)
      item['labels'] = torch.tensor(self.labels[idx]).to(device)
      #item={'input_ids': torch.tensor(val[idx]).to(device) for key, val in self.encoding.items()}
      #item['labels'] = torch.tensor(self.labels['input_ids'][idx]).to(device)
      return item
    def __len__(self):
        return len(self.labels)  # len(self.labels)

class GLUETransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-7,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        datadir = self.hparameters['train_dir']
        #prefix = "summarize: "
        dataset = datasets.load_dataset('json',data_files= datadir)
        train_texts, train_labels = [ each for each in dataset['train']['document']], dataset['train']['summary']
        
        encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        decodings = self.tokenizer(train_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        train_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return train_data 
    def val_dataloader(self):
        datadir = self.hparameters['val_dir']
        #prefix = "summarize: "
        dataset = load_dataset('json',data_files=datadir)
        val_texts, val_labels = [ each for each in dataset['train']['document']], dataset['train']['summary']

        encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        decodings = self.tokenizer(val_labels, truncation=True, padding=True)
        dataset_tokenized = MyData(encodings, decodings)
        print(len(dataset_tokenized))
        val_data = DataLoader(dataset_tokenized,batch_size= self.hparameters['batch_size'],collate_fn=lambda x: x,shuffle=True)
        # create a dataloader for your training data here
        return val_data