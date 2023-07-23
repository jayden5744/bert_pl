import multiprocessing
import os
import os.path as osp
import shutil
import random
from typing import List

import torch
from torch import Tensor
import sentencepiece as spm
from omegaconf import DictConfig
from torch.utils.data import Dataset,  DataLoader, RandomSampler

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


def exist_file(path: str) -> bool:
    if osp.exists(path):
        return True
    return False


def create_or_load_tokenizer(
    file_path: str,
    save_path: str,
    language: str,
    vocab_size: int,
    tokenizer_type: str = "bpe",
    sep_token: str = "[SEP]",
    cls_token: str = "[CLS]",
    mask_token: str = "[MASK]",
    pad_token: str = "[PAD]"
) -> spm.SentencePieceProcessor:
    corpus_prefix = f"{language}_corpus_{vocab_size}"
    user_defined_symbols = f'{pad_token},{cls_token},{sep_token},{mask_token}'

    if tokenizer_type.strip().lower() not in ["unigram", "bpe", "char", "word"]:
        raise ValueError(
            f"param `tokenizer_type` must be one of [unigram, bpe, char, word]"
        )

    if not os.path.isdir(save_path):  # 폴더 없으면 만들어
        os.makedirs(save_path)

    model_path = osp.join(save_path, corpus_prefix + ".model")
    vocab_path = osp.join(save_path, corpus_prefix + ".vocab")

    if not exist_file(model_path) and not exist_file(vocab_path):
        model_train_cmd = f"--input={file_path} --model_prefix={corpus_prefix} --model_type={tokenizer_type} --vocab_size={vocab_size} --user_defined_symbols={user_defined_symbols}"
        spm.SentencePieceTrainer.Train(model_train_cmd)
        shutil.move(corpus_prefix + ".model", model_path)
        shutil.move(corpus_prefix + ".vocab", vocab_path)
    # model file은 있는데, vocab file이 없거나 / model_file은 없는데, vocab file이 있으면 -> Error

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def get_seg_token(seq_1: str, is_pair: bool, max_seq_len: int) -> Tensor:
    seg_token = torch.zeros(max_seq_len)
    if is_pair:
        seg_token[len(seq_1) + 2:] = 1  # + 2 는 [CLS], [SEP] Token때문이다.
    return seg_token.long()



class BERTPretrainDataset(Dataset):
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

        seg_token = get_seg_token(seq_1, True, self.max_seq_len)

        input_sen = torch.tensor([self.cls_id] + self.masking(seq_1) + [self.sep_id] + self.masking(seq_2) + [self.sep_id] + [self.pad_id] * (self.max_seq_len -3 - len(seq_1) - len(seq_2)))
        return input_sen, target_sen, seg_token

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
            i = random.randint(0, len_sen - 1)

            if random.random() <= 0.8:  # 80%는 마스크
                sentence[i] = self.mask_id

            elif random.random() <= 0.5: # 대체
                j = random.randint(4, self.vocab.GetPieceSize() - 1)
                replace_token = self.vocab.IdToPiece(j)
                while sentence[i] == replace_token:  # replace token이 같으면 다른 token이 나올때까지 반복
                    j = random.randint(4, self.vocab.GetPieceSize() - 1)
                    replace_token = self.vocab.IdToPiece(j)
                sentence[i] = self.vocab.PieceToId(replace_token)
        return sentence


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        arg_model: DictConfig,
        vocab: spm.SentencePieceProcessor,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.arg_data = arg_data
        self.arg_model = arg_model
        self.vocab = vocab
        self.max_seq_len = arg_model.max_seq_len
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # 데이터를 다운로드, split 하거나 기타 등등
        # only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        # make assignments here (train/val/test split)
        # called on every process in DDP
        self.train_dataset = BERTPretrainDataset(
            file_path=self.arg_data.train_path,
            vocab=self.vocab,
            max_seq_len=self.max_seq_len,
            mask_ratio=self.arg_model.mask_ratio,
            nsp_ratio=self.arg_model.nsp_ratio,
            sep_token=self.arg_model.sep_token,
            cls_token=self.arg_model.cls_token,
            mask_token=self.arg_model.mask_token,
            pad_token=self.arg_model.pad_token,
        )

        self.valid_dataset = BERTPretrainDataset(
            file_path=self.arg_data.valid_path,
            vocab=self.vocab,
            max_seq_len=self.max_seq_len,
            mask_ratio=self.arg_model.mask_ratio,
            nsp_ratio=self.arg_model.nsp_ratio,
            sep_token=self.arg_model.sep_token,
            cls_token=self.arg_model.cls_token,
            mask_token=self.arg_model.mask_token,
            pad_token=self.arg_model.pad_token,    
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False  # validation에서는 shuffle 하지 않는 것은 권장함
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:
        # clean up after fit or test
        # called on every process in DDP
        # setup 정반대
        return super().teardown(stage)


