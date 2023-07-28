import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl


class LSTMLanguageModel(nn.Module):
    def __init__(self, embeddings_matrix: torch.tensor, stacks: int = 2):
        super().__init__()
        self.vocabulary_size, self.embedding_dimensions = embeddings_matrix.shape

        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_dimensions)
        self.embedding_layer.weight.data.copy_(embeddings_matrix)
        self.embedding_layer.weight.requires_grad = False

        self.lstm_stacks = nn.ModuleList([
            nn.LSTM(self.embedding_dimensions, self.embedding_dimensions, batch_first=True)
            for _ in range(stacks)
        ])

        self.decoder = nn.Linear(self.embedding_dimensions, self.vocabulary_size)

    def forward(self, sentence: torch.Tensor):
        embeddings = self.embedding_layer(sentence)
        lstm_out = embeddings
        for stack in self.lstm_stacks:
            lstm_out, _ = stack(lstm_out)

        return self.decoder(lstm_out)


class LitLSTMLanguageModel(pl.LightningModule):
    def __init__(self, embeddings_matrix: torch.tensor, stacks: int = 2, learning_rate: float = .3):
        super().__init__()
        self.lstm_language_model = LSTMLanguageModel(embeddings_matrix, stacks)
        self.learning_rate = learning_rate

    def training_step(self, batch: tuple[torch.tensor, torch.tensor]):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        self.log("Train Perplexity", torch.exp(loss), prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        perplexity = torch.exp(loss)
        self.log("Test Perplexity", perplexity, prog_bar=True, on_epoch=True, on_step=False)
        return perplexity

    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        x, y = batch
        pred = self.lstm_language_model(x).permute(0, 2, 1)
        loss = F.cross_entropy(pred, y)
        perplexity = torch.exp(loss)
        self.log("Validation Perplexity", perplexity, prog_bar=True, on_epoch=True, on_step=False)
        return perplexity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
