import copy
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from transformers.trainer import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    scores: Optional[np.ndarray]

class CustomTrainerForSquaredGradients(Trainer):

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        test_dataset=None,
    ) -> EvalLoopOutput:
        model = self.model.to("cuda:0")
        precision_matrices = {}
        named_parameters = dict(model.named_parameters())

        for n, p in named_parameters.items():
            if p.requires_grad:
                precision_matrices[n] = variable(torch.zeros_like(p.data))

        model.eval()
        for step, inputs in tqdm(enumerate(dataloader)):
            model.zero_grad()            
            inputs = self._prepare_inputs(inputs)

            if "labels" in inputs:
                del inputs["labels"]
            
            output = model(**inputs).logits
            log_probs = F.log_softmax(output, dim=-1)

            # random sampling of output labels
            cumsums = torch.logcumsumexp(log_probs, dim=-1).exp()
            outdx = torch.searchsorted(cumsums, torch.rand(tuple(list(cumsums.shape[:-1]) + [1])).to(cumsums.device))
            outdx = outdx.squeeze(-1)

            loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), outdx.view(-1))
            loss.backward()
            for n, p in named_parameters.items():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2


        precision_matrices = {n: (p / (step+1.0)).detach() for n, p in precision_matrices.items()}
        for n, p in self.model.named_parameters():
            p.data.copy_(precision_matrices[n].data)

        output_dir = os.path.join(self.args.output_dir, "hessian")
        self.save_model(output_dir)
        return EvalLoopOutput(predictions=[[0.0]], label_ids=[0], metrics={}, num_samples=1.0, scores=None)

class CustomTrainerForObservedFisher(Trainer):

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        test_dataset=None,
    ) -> EvalLoopOutput:
        model = self.model.to("cuda:0")
        precision_matrices = {}
        named_parameters = dict(model.named_parameters())

        for n, p in named_parameters.items():
            if p.requires_grad:
                precision_matrices[n] = variable(torch.zeros_like(p.data))

        model.eval()
        for step, inputs in tqdm(enumerate(dataloader)):
            model.zero_grad()            
            inputs = self._prepare_inputs(inputs)

            loss = model(**inputs).loss

            loss.backward()
            for n, p in named_parameters.items():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: (p / (step+1.0)).detach() for n, p in precision_matrices.items()}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(precision_matrices[n].data)

        output_dir = os.path.join(self.args.output_dir, "hessian")
        self.save_model(output_dir)
        return EvalLoopOutput(predictions=[[0.0]], label_ids=[0], metrics={}, num_samples=1.0, scores=None)