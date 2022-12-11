import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pytorch_lightning.loggers import WandbLogger

class BartModel(pl.LightningModule):
  def __init__(self, hparams):
    super(BartModel, self).__init__()
    self.hparams = hparams
    self.tokenizer = BartTokenizer.from_pretrained('bart-large')
    self.model = BartForConditionalGeneration.from_pretrained('bart-large')

  def forward(self, input_ids, attention_mask):
    return self.model(input_ids, attention_mask=attention_mask)

  def training_step(self, batch, batch_idx):
    input_ids, attention_mask = batch
    loss, logits = self(input_ids, attention_mask)

    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, attention_mask = batch
    loss, logits = self(input_ids, attention_mask)

    self.log('val_loss', loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

  @pl.data_loader
  def train_dataloader(self):
    # create a dataloader for your training data here
    pass

  @pl.data_loader
  def val_dataloader(self):
    # create a dataloader for your validation data here
    pass

hparams = {
    'max_epochs': 5,
    'batch_size': 2,
    'learning_rate': 3e-5,
    'train_dir': "./dataset/EUR-Lex/train_finetune.json",
    'val_dir': "./dataset/EUR-Lex/test_finetune.json"
}

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

model = BartModel(hparams)
wandb_logger = WandbLogger()

trainer = pl.Trainer(max_epochs=100, early_stop_callback=early_stopping, logger=wandb_logger)
trainer.fit(model)