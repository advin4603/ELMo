from torch import nn
import torch
import os
from elmo import ElMo
import lightning.pytorch as pl
import torch.nn.functional as F


class SentenceClassifier(nn.Module):
    def __init__(self, classes: int, embeddings_matrix: torch.tensor, forward_lm_checkpoint_path: str | os.PathLike,
                 backward_lm_checkpoint_path: str | os.PathLike, stacks: int, hidden_state_size: int,
                 lstm_classifier_layers, decoder_layers: tuple[int, ...]):
        super().__init__()
        self.elmo_layer = ElMo(embeddings_matrix, forward_lm_checkpoint_path, backward_lm_checkpoint_path, stacks)
        self.hidden_state_size = hidden_state_size
        self.lstm_classifier_layers = lstm_classifier_layers
        self.lstm_layer = nn.LSTM(self.elmo_layer.output_dimensions, self.hidden_state_size,
                                  self.lstm_classifier_layers,
                                  batch_first=True)
        self.classes = classes
        self.decoder = nn.Sequential()
        for in_dimensions, out_dimensions in zip((hidden_state_size,) + decoder_layers[:-1],
                                                 decoder_layers):
            self.decoder.append(nn.Linear(in_dimensions, out_dimensions))
            self.decoder.append(nn.ReLU())

        self.decoder.append(nn.Linear(decoder_layers[-1] if decoder_layers else hidden_state_size, classes))

    def forward(self, sentence: torch.Tensor):
        contextual_embeddings = self.elmo_layer(sentence)
        _, (_, cell_state_n) = self.lstm_layer(contextual_embeddings)
        cell_state_n = cell_state_n[-1]
        score = self.decoder(cell_state_n)
        return score


class LitSentenceClassifier(pl.LightningModule):
    def __init__(self, classes: int, embeddings_matrix: torch.tensor, forward_lm_checkpoint_path: str | os.PathLike,
                 backward_lm_checkpoint_path: str | os.PathLike, stacks: int = 2, hidden_state_size: int = 300,
                 lstm_classifier_layers=2, decoder_layers: tuple[int, ...] = (100,),
                 learning_rate: float = 0.3):
        super().__init__()
        self.sentence_classifier = SentenceClassifier(classes, embeddings_matrix, forward_lm_checkpoint_path,
                                                      backward_lm_checkpoint_path, stacks, hidden_state_size,
                                                      lstm_classifier_layers, decoder_layers)
        self.learning_rate = learning_rate

    def training_step(self, batch: tuple[torch.tensor, torch.tensor]):
        x, y = batch
        pred = self.sentence_classifier(x)
        loss = F.cross_entropy(pred, y)
        predicted_labels = pred.argmax(1)
        accuracy = (predicted_labels == y).type(torch.float).mean().item() * 100
        self.log("Train Accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("Train loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.sentence_classifier(x)
        predicted_labels = pred.argmax(1)
        accuracy = (predicted_labels == y).type(torch.float).mean().item() * 100
        self.log("Validation Accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return accuracy

    def test_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.sentence_classifier(x)
        predicted_labels = pred.argmax(1)
        accuracy = (predicted_labels == y).type(torch.float).mean().item() * 100
        self.log("Test Accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    from data import construct_embeddings, SentimentScoreDataset, START_TOKEN, END_TOKEN
    from datasets import load_dataset

    dataset = load_dataset("sst", "default")
    train_dataset = SentimentScoreDataset(dataset["train"], 5)
    vocabulary = train_dataset.vocabulary
    embeddings = construct_embeddings(vocabulary)

    s = LitSentenceClassifier(2, embeddings, "forward_st.ckpt", "backward_st.ckpt")
    sentence = [
        START_TOKEN,
        "I",
        "am",
        "a",
        "boy",
        END_TOKEN
    ]
    a = s.sentence_classifier(
        torch.tensor([[vocabulary[token] for token in sentence], [vocabulary[token] for token in sentence],
                      [vocabulary[token] for token in sentence]]))
