import random
from typing import List

import torch
import sentencepiece as spm
from torch.utils.data import Dataset


class BERTPretrain_Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_len: int,
        mask_ratio: float = 0.15,
        nsp_ratio: float = 0.5,
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        pad_token: str = "[PAD]"
        ) -> None:
        super().__init__()
        self.data = self.load_txt_file(file_path)
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        self.sep_id = vocab.PieceToId(sep_token)
        self.cls_id = vocab.PieceToId(cls_token)
        self.mask_id = vocab.PieceToId(mask_token)
        self.pad_id = vocab.PieceToId(pad_token)

        self.mask_ratio = mask_ratio
        self.nsp_ratio = nsp_ratio

    def __getitem__(self, index):
        seq_1 = self.vocab.EncodeAsIds(self.data[index])
        seq_2_idx = index + 1

        if random.random() >= self.nsp_ratio:
            is_next = False
            while seq_2_idx == index + 1:
                seq_2_idx = random.randint(0, len(self.data)-1)

        else:
            is_next = True

        seq_2 = self.vocab.EncodeAsIds(self.data[seq_2_idx])

        # max sequence 분기문
        if len(seq_1) + len(seq_2) >= self.max_seq_len - 3: # -3 은 스페셜 토큰 개수
            idx = self.max_seq_len - 3 - len(seq_1)
            seq_2 = seq_2[:idx]

        target_sen = torch.tensor([1 if is_next else 0] + seq_1 + [self.sep_id] + seq_2 + [self.sep_id] + [self.pad_id] * (self.max_seq_len -3 - len(seq_1) - len(seq_2)))
        seg_emb = torch.zeros(target_sen.size(0)) 
        seg_emb[len(seq_1) + 2:] = 1 

        input_sen = torch.tensor([self.cls_id] + self.masking(seq_1) + [self.sep_id] + self.masking(seq_2) + [self.sep_id] + [self.pad_id] * (self.max_seq_len -3 - len(seq_1) - len(seq_2)))
        return input_sen, target_sen, seg_emb

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
      for x in self.data:
        yield x

    def get_vocab(self) -> spm.SentencePieceProcessor:
        return self.vocab

    def decode(self, x: str) -> str:
        return self.vocab.DecodeIds(x)

    def load_txt_file(self, file_path: str) -> List[str]:
        dataset = []
        with open(file_path, 'r') as f:
            for data in f.readlines():
                dataset.append(data.strip())
        return dataset

    def masking(self, sentence: List[int]):
        len_sen = len(sentence)

        mask_amount = round(len_sen * self.mask_ratio) # 문장 중에 총 mask ratio % 만큼 변경 예정
        for _ in range(mask_amount):
            i = random.randint(0, len - 1)

            if random.random() <= 0.8:  # 80%는 마스크
                sentence[i] = self.mask_id

            elif random.random() <= 0.5: # 대체
                j = random.randint(4, self.vocab.GetPieceSize())
                replace_token = self.vocab.IdToPiece(j)
                while sentence[i] == replace_token:  # replace token이 같으면 다른 token이 나올때까지 반복
                    j = random.randint(4, self.vocab.GetPieceSize())
                    replace_token = self.vocab.IdToPiece(j)
                sentence[i] = replace_token
        return sentence






# Todo 2 : DataModule 이라는 PL class
