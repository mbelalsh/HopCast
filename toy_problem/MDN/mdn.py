from enum import Enum, auto
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from typing import Dict, Tuple, List
import numpy as np

class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()

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

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components, hidden_dim, noise_type=NoiseType.DIAGONAL, fixed_noise_level=None):
        super().__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        num_sigma_channels = {
            NoiseType.DIAGONAL: dim_out * n_components,
            NoiseType.ISOTROPIC: n_components,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
        self.pi_network = FcModel(dim_in, n_components, hidden=(100,100))
        self.normal_network = FcModel(dim_in,\
                     dim_out * n_components + num_sigma_channels, hidden=(100,100))

    def forward(self, x, eps=1e-6):
        #
        # Returns
        # -------
        # log_pi: (bsz, n_components)
        # mu: (bsz, n_components, dim_out)
        # sigma: (bsz, n_components, dim_out)
        #
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            -torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def sample(self, x):

        log_pi, mu, sigma = self.forward(x) # [512,3], [512,3,1], [512,3,1] [batch,n_comp,out_dim]
        #print(torch.exp(log_pi[:10,:])) 
        #print(mu.shape)
        #print(sigma.shape)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x) # Uniform [0,1) [512,1]
        rand_pi = torch.searchsorted(cum_pi, rvs) # For random mode selection to sample
        rand_normal = torch.randn_like(mu) * sigma + mu # [512,3,1]
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1) #[512, 1]

        return samples, log_pi, mu, sigma    

class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, input_data, output_data):

        self.data_X = input_data
        self.data_y = output_data          

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]    

def prepare_loaders(x: np.ndarray, y: np.ndarray, params: Dict):

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = y[rand_perm]

    train_len = int(0.8*x.shape[0])
    val_len = x.shape[0]-train_len

    x_train = x[:train_len]
    x_val = x[-val_len:]  

    y_train = y[:train_len]
    y_val = y[-val_len:]

    
    train_loader = DataLoader(TransitionDataset(x_train,y_train),
                                batch_size=params['batch_size'], shuffle=True)
    val_set = TransitionDataset(x_val,y_val)

    sampler = SequentialSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], 
                                sampler=sampler, shuffle=False)

    return train_loader, val_loader, x_val, y_val

def prepare_val_loader(x_val: torch.Tensor, y_val: torch.Tensor):

    val_set = TransitionDataset(x_val,y_val)
    sampler = SequentialSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=256, 
                                sampler=sampler, shuffle=False)
    
    return val_loader