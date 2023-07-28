from language_model import LitLSTMLanguageModel
from data import ForwardLanguageModelMultiNLIDataset, construct_embeddings, create_collate
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import RichProgressBar

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)

dataset = load_dataset("multi_nli")

MINIMUM_FREQUENCY = 5
BATCH_SIZE = 64
NUM_WORKERS = 12
LEARNING_RATE = 2e-4
MAX_EPOCHS = 20
TRAIN_DATA_LIMIT = 10_000

train_dataset = ForwardLanguageModelMultiNLIDataset(dataset["train"], MINIMUM_FREQUENCY, limit=TRAIN_DATA_LIMIT)
vocabulary = train_dataset.vocabulary
collate_fn = create_collate(vocabulary)

train_dataloader = DataLoader(train_dataset,
                              collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
validation_dataloader = DataLoader(
    ForwardLanguageModelMultiNLIDataset(dataset["validation_matched"], MINIMUM_FREQUENCY, vocabulary),
    collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(ForwardLanguageModelMultiNLIDataset(dataset["validation_mismatched"], MINIMUM_FREQUENCY, vocabulary),
                             collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

embeddings = construct_embeddings(vocabulary)

trainer = pl.Trainer(default_root_dir="forward_multinli", max_epochs=MAX_EPOCHS, callbacks=[RichProgressBar()])

model = LitLSTMLanguageModel(embeddings, learning_rate=LEARNING_RATE)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

trainer.test(model, dataloaders=test_dataloader)

trainer.save_checkpoint("forward_multinli.ckpt")

