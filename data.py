import torch
import torchtext.vocab as vocab
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from typing import Union
from random import shuffle
from torchtext.data import get_tokenizer

UNKNOWN_TOKEN = "<{UNKNOWN_TOKEN}>"
START_TOKEN = "<{START_TOKEN}>"
END_TOKEN = "<{END_TOKEN}>"
SEP_TOKEN = "<{SEP_TOKEN}>"
SPECIALS = [UNKNOWN_TOKEN, START_TOKEN, END_TOKEN]


def construct_embeddings(vocabulary: vocab.vocab) -> torch.tensor:
    global_embeddings = vocab.FastText(language="en", cache="fasttext_cache")
    weights_matrix = torch.zeros((len(vocabulary), global_embeddings.dim))
    mean = torch.zeros(global_embeddings.dim)
    std = torch.ones(global_embeddings.dim)
    for i, token in enumerate(vocabulary.get_itos()):
        if token in global_embeddings.stoi:
            weights_matrix[i] = global_embeddings[token]
        else:
            weights_matrix[i] = torch.normal(mean, std)

    return weights_matrix


class MultiNLIDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, minimum_frequency: int, vocabulary: vocab.Vocab | None = None,
                 limit: int = 10_000):
        self.dataset = dataset
        self.tokenizer = get_tokenizer("basic_english")
        self.premises = [[token.lower().strip() for token in self.tokenizer(premise)] for premise in dataset["premise"]]
        self.hypotheses = [[token.lower().strip() for token in self.tokenizer(hypothesis)] for hypothesis in
                           dataset["hypothesis"]]
        self.labels = dataset["label"]

        filtered_premises, filtered_hypotheses, filtered_labels = [], [], []
        for premise, hypothesis, label in zip(self.premises, self.hypotheses, self.labels):
            if label != -1:
                filtered_premises.append(premise)
                filtered_hypotheses.append(hypothesis)
                filtered_labels.append(label)

        self.premises, self.hypotheses, self.labels = filtered_premises[:limit], filtered_hypotheses[
                                                                                 :limit], filtered_labels[:limit]
        if vocabulary is None:
            self.vocabulary = vocab.build_vocab_from_iterator(self.premises + self.hypotheses,
                                                              min_freq=minimum_frequency,
                                                              specials=SPECIALS + [SEP_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

        self.encoded_premises = [[self.vocabulary[token] for token in premise] for premise in self.premises]
        self.encoded_hypotheses = [[self.vocabulary[token] for token in hypothesis] for hypothesis in self.hypotheses]

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(
            self.encoded_premises[index] + [self.vocabulary[SEP_TOKEN]] + self.encoded_hypotheses[index]), \
            torch.tensor(self.labels[index])


class ForwardLanguageModelMultiNLIDataset(MultiNLIDataset):
    def __init__(self, dataset: datasets.Dataset, minimum_frequency: int, vocabulary: vocab.Vocab | None = None,
                 limit: int = 10_000):
        super().__init__(dataset, minimum_frequency, vocabulary, limit)
        self.encoded_sentences = self.encoded_premises + self.encoded_hypotheses

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index):
        return torch.tensor(self.encoded_sentences[index][:-1]), torch.tensor(self.encoded_sentences[index][1:])


class BackwardLanguageModelMultiNLIDataset(ForwardLanguageModelMultiNLIDataset):
    def __init__(self, dataset: datasets.Dataset, minimum_frequency: int, vocabulary: vocab.Vocab | None = None,
                 limit: int = 10_000):
        super().__init__(dataset, minimum_frequency, vocabulary, limit)
        self.encoded_sentences = [encoded_sentence[::-1] for encoded_sentence in self.encoded_sentences]


class SentimentScoreDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, minimum_frequency: int, vocabulary: Union[vocab.vocab, None] = None):
        self.dataset = dataset
        self.token_sets: list[list[str]] = [tokens.lower().split("|") for tokens in self.dataset["tokens"]]
        if vocabulary is None:
            self.vocabulary = vocab.build_vocab_from_iterator(self.token_sets, min_freq=minimum_frequency,
                                                              specials=SPECIALS)
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

        self.labels = [round(score) for score in self.dataset["label"]]
        self.sentences: list[list[str]] = [[START_TOKEN] + tokens + [END_TOKEN] for tokens in
                                           self.token_sets]
        self.encoded_sentences = [
            [self.vocabulary[token] for token in sentence]
            for sentence in self.sentences
        ]
        self.data = list(zip(self.encoded_sentences, self.labels))
        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return torch.tensor(self.data[index][0]), torch.tensor(self.data[index][1])


class ForwardLanguageModelSentimentScoreDataset(Dataset):
    def __init__(self, dataset: datasets.Dataset, minimum_frequency: int, vocabulary: Union[vocab.vocab, None] = None):
        self.dataset = dataset
        self.token_sets: list[list[str]] = [tokens.lower().split("|") for tokens in self.dataset["tokens"]]
        if vocabulary is None:
            self.vocabulary = vocab.build_vocab_from_iterator(self.token_sets, min_freq=minimum_frequency,
                                                              specials=SPECIALS)
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary
        self.sentences: list[list[str]] = [[START_TOKEN] + tokens + [END_TOKEN] for tokens in
                                           self.token_sets]
        self.encoded_sentences = [
            [self.vocabulary[token] for token in sentence]
            for sentence in self.sentences
        ]

    def __len__(self):
        return len(self.encoded_sentences)

    def __getitem__(self, index: int):
        return torch.tensor(self.encoded_sentences[index][:-1]), torch.tensor(self.encoded_sentences[index][1:])


class BackwardLanguageModelSentimentScoreDataset(ForwardLanguageModelSentimentScoreDataset):
    def __init__(self, dataset: Dataset, minimum_frequency: int, vocabulary: Union[vocab.vocab, None] = None):
        super().__init__(dataset, minimum_frequency, vocabulary)
        self.sentences = [sentence[::-1] for sentence in self.sentences]
        self.encoded_sentences = [encoded_sentence[::-1] for encoded_sentence in self.encoded_sentences]


def create_sentiment_collate(vocabulary: vocab.vocab):
    def custom_collate(data: list[tuple[torch.tensor, torch.tensor]]) -> tuple[torch.tensor, torch.tensor]:
        x_list, y_list = [], []
        for x, y in data:
            x_list.append(x)
            y_list.append(y)

        return pad_sequence(x_list, batch_first=True, padding_value=vocabulary[END_TOKEN]), torch.tensor(y_list)

    return custom_collate


def create_collate(vocabulary: vocab.vocab):
    def custom_collate(data: list[tuple[torch.tensor, torch.tensor]]) -> tuple[torch.tensor, torch.tensor]:
        x_list, y_list = [], []
        for x, y in data:
            x_list.append(x)
            y_list.append(y)

        return pad_sequence(x_list, batch_first=True, padding_value=vocabulary[END_TOKEN]), \
            pad_sequence(y_list, batch_first=True, padding_value=vocabulary[END_TOKEN])

    return custom_collate
