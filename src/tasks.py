import torch.nn as nn
from torch import Tensor


class MaskedLanguageModeling(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, bert_ouput: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_ouput))


class NextSentencePrediction(nn.Module):
    def __init__(
        self,
        d_model: int
        ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, bert_ouput: Tensor) -> Tensor:
        return self.softmax(self.linear(bert_ouput[:, 0]))