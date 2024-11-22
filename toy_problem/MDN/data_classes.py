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
    error: Optional[torch.Tensor] = None  # only for predicted data from model
    timesteps: Optional[torch.Tensor] = None # only for predicted data from model
    X_ctx: Optional[torch.Tensor] = None

@dataclass
class ValOutput:
    """VAL output for plotting"""
    y: np.ndarray
    y_pred: np.ndarray
    y_low: np.ndarray
    y_high: np.ndarray

@dataclass
class ValOutputError:
    """VAL output for plotting"""
    error: np.ndarray
    error_low: np.ndarray
    error_high: np.ndarray    

@dataclass
class MemoryData:
    """Encoded X and Error from train data for evaluation"""
    X_ctx_true_train_enc: torch.Tensor
    error_train: torch.Tensor 
 
class TrainLoader(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, X_ctx_true: torch.Tensor, X_ctx_sim: torch.Tensor, 
                 errors: torch.Tensor):

        self._X_ctx_true = X_ctx_true
        self._X_ctx_sim = X_ctx_sim   
        self._errors = errors    

    def __len__(self):
        return self._X_ctx_true.shape[0]

    def __getitem__(self, index):
        return self._X_ctx_true[index], self._X_ctx_sim[index], self._errors[index]
    
class ValLoader(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, X_ctx_true: torch.Tensor, X_ctx_sim: torch.Tensor, 
                 errors: torch.Tensor, Y: torch.Tensor, 
                 Y_pred: torch.Tensor):

        self._X_ctx_true = X_ctx_true
        self._X_ctx_sim = X_ctx_sim   
        self._errors = errors    
        self._Y = Y
        self._Y_pred = Y_pred

    def __len__(self):
        return self._X_ctx_true.shape[0]

    def __getitem__(self, index):
        return self._X_ctx_true[index], self._X_ctx_sim[index],\
              self._errors[index], self._Y[index], self._Y_pred[index]       

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