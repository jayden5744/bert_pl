import torch.nn as nn
from torch import Tensor


class MaskedLanguageModel(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        vocab_size: int
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, bert_output: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_output))


class NextSentencePrediction(nn.Module):
    def __init__(
        self,
        d_hidden: int
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, bert_output: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_output[:, 0]))
        