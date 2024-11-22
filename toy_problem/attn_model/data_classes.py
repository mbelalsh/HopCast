from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List, Dict
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import numpy as np

@dataclass
class PICalibData:
    """Episodes available for calibration"""
    X: torch.Tensor
    Y: torch.Tensor
    X_norm: Optional[torch.Tensor] = None  
    Y_norm: Optional[torch.Tensor] = None 

@dataclass
class ValOutput:
    """VAL output for plotting"""
    y: np.ndarray
    y_low: np.ndarray
    y_high: np.ndarray

@dataclass
class MemoryData:
    """Encoded X and Error from train data for evaluation"""
    _X_enc_mem: torch.Tensor
    _Y_mem: torch.Tensor
    _X_mem:  Optional[torch.Tensor] = None

class EnsembleTrainLoader(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):

        self._X = X # [4, 24, 400, 1] [Ensembles,batch_dim,seq_len,feat]
        self._Y = Y   

    def __len__(self):
        return self._X.shape[1]

    def __getitem__(self, index):
        return self._X[:,index,:,:], self._Y[:,index,:,:]
    
class EnsembleValLoader(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):

        self._X = X # [4, 24, 400, 1] [Ensembles,batch_dim,seq_len,feat]
        self._Y = Y   

    def __len__(self):
        return self._X.shape[1]

    def __getitem__(self, index):
        return self._X[:,index,:,:], self._Y[:,index,:,:]   

class FcModel(nn.Module):

    def __init__(self, input_dim, out_dim, hidden: Tuple=(), dropout: float = 0, dropout_at_first=False,
                 dropout_after_last=False, dropout_intermediate=False, relu_after_last=False) -> None:

        nn.Module.__init__(self)    
        self._out_dim = out_dim
        if dropout is None:
            dropout = 0
        hidden_layers = []
        if len(hidden) > 0:
            for idx, layer in enumerate(hidden):
                hidden_layers.append(nn.ReLU())
                if dropout > 0 and dropout_intermediate:
                    hidden_layers.append(nn.Dropout(p=dropout))
                hidden_layers.append(nn.Linear(layer, hidden[idx+1] if idx < (len(hidden) - 1) else self._out_dim))
            stack = [nn.Linear(input_dim, hidden[0])] + hidden_layers
        else: # active
            stack = [nn.Linear(input_dim, self._out_dim)]

        if dropout > 0 and dropout_at_first:
            stack = [nn.Dropout(p=dropout)] + stack
        if relu_after_last:
            stack.append(nn.ReLU())
        if dropout > 0 and dropout_after_last:
            stack.append(nn.Dropout(p=dropout))
        self.linear_stack = nn.Sequential(*stack)

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, context, **kwargs):
        return self.linear_stack(context)    