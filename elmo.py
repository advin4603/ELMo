import torch
import os
from language_model import LitLSTMLanguageModel
from torch import nn


class ElMo(nn.Module):
    def __init__(self, embeddings_matrix: torch.tensor, forward_lm_checkpoint_path: str | os.PathLike,
                 backward_lm_checkpoint_path: str | os.PathLike, stacks: int = 2):
        super().__init__()
        self.forward_lm = LitLSTMLanguageModel.load_from_checkpoint(forward_lm_checkpoint_path,
                                                                    embeddings_matrix=embeddings_matrix,
                                                                    stacks=stacks).lstm_language_model
        self.forward_lm.requires_grad_(False)
        self.backward_lm = LitLSTMLanguageModel.load_from_checkpoint(backward_lm_checkpoint_path,
                                                                     embeddings_matrix=embeddings_matrix,
                                                                     stacks=stacks).lstm_language_model
        self.backward_lm.requires_grad_(False)
        self.embedding_weights = nn.parameter.Parameter(torch.normal(torch.zeros(stacks + 1), torch.ones(stacks + 1)))
        self.embedding_weights.requires_grad = True
        self.output_dimensions = self.forward_lm.embedding_dimensions * 2

    def forward(self, sentence: torch.Tensor):
        embeddings = self.forward_lm.embedding_layer(sentence)
        forward_embeddings: list[torch.Tensor] = [embeddings]
        for forward_lstm_stack in self.forward_lm.lstm_stacks:
            lstm_out, _ = forward_lstm_stack(forward_embeddings[-1])
            forward_embeddings.append(lstm_out)

        backward_embeddings: list[torch.Tensor] = [torch.flip(embeddings, (-2,))]
        for backward_lstm_stack in self.backward_lm.lstm_stacks:
            lstm_out, _ = backward_lstm_stack(backward_embeddings[-1])
            backward_embeddings.append(lstm_out)

        for i, backward_embedding in enumerate(backward_embeddings):
            backward_embeddings[i] = torch.flip(backward_embedding, (-2,))
        embedding_list = [torch.cat((forward_embedding, backward_embedding), -1) for
                          forward_embedding, backward_embedding
                          in
                          zip(forward_embeddings, backward_embeddings)]
        contextual_embeddings = sum(
            embedding * embedding_weight for embedding, embedding_weight in
            zip(embedding_list, self.embedding_weights)
        )

        return contextual_embeddings


if __name__ == "__main__":
    from data import construct_embeddings, ForwardLanguageModelSentimentScoreDataset, START_TOKEN, END_TOKEN
    from datasets import load_dataset
    torch.manual_seed(42)

    dataset = load_dataset("sst", "default")
    train_dataset = ForwardLanguageModelSentimentScoreDataset(dataset["train"], 5)
    vocabulary = train_dataset.vocabulary
    embeddings = construct_embeddings(vocabulary)
    e = ElMo(embeddings, "forward_st.ckpt", "backward_st.ckpt")
    sentence = [[
        "i"
    ], ["the"]]
    print(e.forward(torch.tensor([[vocabulary[token] for token in tokens] for tokens in sentence])))
