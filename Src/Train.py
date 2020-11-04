import torch
import numpy as np

from torch import nn
from transformers import Trainer, is_apex_available

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


if is_apex_available():
    from apex import amp


class SoftTrainer(Trainer):

    def __init__(
            self,
            model,
            args,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=None,
            prediction_loss_only=None,
            tb_writer=None,
            optimizers=None
    ):
        super().__init__(model, args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=compute_metrics,
                         prediction_loss_only=prediction_loss_only,
                         tb_writer=tb_writer,
                         optimizers=optimizers)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(**inputs)

        # TODO: I should take the second return ([1]) and apply kld loss here
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()
