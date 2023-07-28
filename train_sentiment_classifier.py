from sentence_classifier import LitSentenceClassifier
from data import SentimentScoreDataset, construct_embeddings, create_sentiment_collate
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import RichProgressBar
from datasets import load_dataset
import lightning.pytorch as pl
import torch
import random

torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
random.seed(42)

dataset = load_dataset("sst", "default")

MINIMUM_FREQUENCY = 5
BATCH_SIZE = 128
NUM_WORKERS = 12

MAX_EPOCHS = 100

train_dataset = SentimentScoreDataset(dataset["train"], MINIMUM_FREQUENCY)
vocabulary = train_dataset.vocabulary

collate_fn = create_sentiment_collate(vocabulary)

train_dataloader = DataLoader(train_dataset,
                              collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
validation_dataloader = DataLoader(SentimentScoreDataset(dataset["validation"], 0, vocabulary),
                                   collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dataset = SentimentScoreDataset(dataset["test"], 0, vocabulary)
test_dataloader = DataLoader(test_dataset,
                             collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

embeddings = construct_embeddings(vocabulary)

LEARNING_RATE = 1e-4
HIDDEN_STATE_SIZE = 300
LSTM_CLASSIFIER_LAYERS = 1
decoder_layer_count = 2
DECODER_LAYERS = (400, 350)
if __name__ == "__main__":
    trainer = pl.Trainer(default_root_dir="sentiment_classifier", max_epochs=MAX_EPOCHS, callbacks=[RichProgressBar()])
    model = LitSentenceClassifier(2, embeddings, "forward_st.ckpt", "backward_st.ckpt", stacks=2,
                                  hidden_state_size=HIDDEN_STATE_SIZE, lstm_classifier_layers=LSTM_CLASSIFIER_LAYERS,
                                  decoder_layers=DECODER_LAYERS, learning_rate=LEARNING_RATE)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    trainer.test(model, dataloaders=test_dataloader)

    trainer.save_checkpoint("sentiment_classifier.ckpt")
