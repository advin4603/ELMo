from sentence_classifier import LitSentenceClassifier
from data import MultiNLIDataset, construct_embeddings, create_sentiment_collate
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
import torch
import random

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
random.seed(42)

dataset = load_dataset("multi_nli")

MINIMUM_FREQUENCY = 5
BATCH_SIZE = 64
NUM_WORKERS = 12

MAX_EPOCHS = 20

train_dataset = MultiNLIDataset(dataset["train"], MINIMUM_FREQUENCY)
vocabulary = train_dataset.vocabulary

collate_fn = create_sentiment_collate(vocabulary)

train_dataloader = DataLoader(train_dataset,
                              collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
validation_dataloader = DataLoader(MultiNLIDataset(dataset["validation_matched"], 0, vocabulary),
                                   collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataset = MultiNLIDataset(dataset["validation_mismatched"], 0, vocabulary)
test_dataloader = DataLoader(test_dataset,
                             collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

embeddings = construct_embeddings(vocabulary)

LEARNING_RATE = 1e-3
HIDDEN_STATE_SIZE = 115
LSTM_CLASSIFIER_LAYERS = 1
DECODER_LAYERS = tuple()

if __name__ == "__main__":
    trainer = pl.Trainer(default_root_dir="multi_nli_classifier", max_epochs=MAX_EPOCHS, callbacks=[RichProgressBar()])
    model = LitSentenceClassifier(3, embeddings, "forward_multinli.ckpt", "backward_multinli.ckpt", stacks=2,
                                  hidden_state_size=HIDDEN_STATE_SIZE, lstm_classifier_layers=LSTM_CLASSIFIER_LAYERS,
                                  decoder_layers=DECODER_LAYERS, learning_rate=LEARNING_RATE)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    trainer.save_checkpoint("multinli_classifier.ckpt")
