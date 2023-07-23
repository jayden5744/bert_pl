from typing import Any, Dict, Tuple, List

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
import sentencepiece as spm
import lightning.pytorch as pl
from omegaconf import DictConfig

from src.model import BERT
from src.tasks import MaskedLanguageModel, NextSentencePrediction
from src.dataset import create_or_load_tokenizer


class BERTPretrainModel(pl.LightningModule):
    def __init__(self, arg: DictConfig) -> None:
        super().__init__()
        self.arg = arg 
        self.vocab = self.get_vocab()
        self.model = self.get_model()

        self.mlm_task = MaskedLanguageModel(
            d_hidden=self.arg.model.d_hidden,
            vocab_size=self.arg.data.vocab_size
            )
        self.nsp_task = NextSentencePrediction(d_hidden=self.arg.model.d_hidden)

        self.loss_function = nn.NLLLoss( # CrossEntropy loss - softmax
            ignore_index=self.vocab.PieceToId(self.arg.model.pad_token)
        )

    def _shared_eval_step(self, batch, batch_idx: int) -> Tensor:
        # validation step과 test step의 공통으로 사용되는 부분
        input_sen, target_sen, seg_token = batch
        output = self.model(input_sen, seg_token)

        mlm_output = self.mlm_task(output)
        nsp_output = self.nsp_task(output)
        return self.calculate_loss(mlm_output, target_sen, nsp_output)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        input_sen, target_sen, seg_token = batch
        output = self.model(input_sen, seg_token)

        mlm_output = self.mlm_task(output)
        nsp_output = self.nsp_task(output)

        loss, mlm_loss, nsp_loss = self.calculate_loss(mlm_output, target_sen, nsp_output)

        metrics = {"loss": loss, "mlm_loss": mlm_loss, "nsp_loss": nsp_loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        loss, mlm_loss, nsp_loss  = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "val_mlm_loss": mlm_loss, "val_nsp_loss": nsp_loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self) -> Optimizer:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param `optimizer` must be one of [Adam, AdamW].")
        return optimizer

    def on_validation_epoch_end(self):
        # validation 1 epoch 끝나고 나서 수행하게 될 로직
        pass

    def calculate_loss(self, mlm_output: Tensor, target_sen: Tensor, nsp_output: Tensor) -> Tensor:
        if self.device.type == "mps":
            # mps float64를 처리할 수 없음
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            mlm_output = mlm_output.to(device="cpu")
            nsp_output = nsp_output.to(device="cpu")
            target_sen = target_sen.to(device="cpu")

        mlm_loss = self.loss_function(mlm_output.transpose(1, 2), target_sen)
        nsp_loss = self.loss_function(nsp_output, target_sen[:, 0])
        loss = mlm_loss + nsp_loss
        return loss, mlm_loss, nsp_loss


    def get_model(self) -> nn.Module:
        params = {
            "vocab_size": self.arg.data.vocab_size,
            "d_hidden": self.arg.model.d_hidden,
            "n_heads": self.arg.model.n_heads,
            "ff_dim": self.arg.model.d_hidden * 4, # ff_dim은 d_hidden * 4이다(페이퍼)
            "n_layers": self.arg.model.n_layers,
            "max_seq_len": self.arg.model.max_seq_len,
            "dropout_rate": self.arg.model.dropout_rate,
            "padding_id": self.vocab.PieceToId(self.arg.model.pad_token),
        }
        return BERT(**params)


    def get_vocab(
        self,
    ) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        vocab = create_or_load_tokenizer(
            file_path=self.arg.data.train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.language,
            vocab_size=self.arg.data.vocab_size,
            tokenizer_type=self.arg.data.tokenizer_type,
            cls_token=self.arg.model.cls_token,
            sep_token=self.arg.model.sep_token,
            mask_token=self.arg.model.mask_token,
            pad_token=self.arg.model.pad_token
        )
        return vocab

