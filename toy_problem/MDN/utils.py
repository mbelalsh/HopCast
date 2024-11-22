import torch.nn as nn
import torch 
import numpy as np
import sys
import os 
from pathlib import Path
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import random
from copy import deepcopy
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(params):

    params['env'].seed(params['seed'])
    params['env'].action_space.seed(params['seed'] + 1)
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])        

class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)  # ob_dim or ac_dim
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)   
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
        self.stdev[self.stdev <= self.eps] = 1.0

    def reset(self):
        self.__init__(self.shape, self.clip)

    def get_torch_stdev(self):
        self.update_torch()    
        return self.torch_stdev

    def update_torch(self):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean[:x.shape[1]]) / self.stdev[:x.shape[1]]), -self.clip, self.clip)
    
    def ac_filter(self, x):
        return np.clip(((x - self.mean[-x.shape[1]:]) / self.stdev[-x.shape[1]:]), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean[:x.shape[1]]) / self.torch_stdev[:x.shape[1]]), -self.clip, self.clip)

    def ac_filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean[-x.shape[1]:]) / self.torch_stdev[-x.shape[1]:]), -self.clip, self.clip)
    
    def invert(self, x):
        return (x * self.stdev) + self.mean

    def ac_invert(self, x: np.ndarray):
        return (x * self.stdev[-x.shape[1]:]) + self.mean[-x.shape[1]:]    

    def invert_torch(self, x: torch.Tensor):
        self.update_torch()
        return (x * self.torch_stdev) + self.torch_mean

    def ac_invert_torch(self, x: torch.Tensor):
        self.update_torch()
        return (x * self.torch_stdev[-x.shape[1]:]) + self.torch_mean[-x.shape[1]:]    

def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)
