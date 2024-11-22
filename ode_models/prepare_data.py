from dataclasses import dataclass
import errno
import torch 
import numpy as np
import pandas as pd
from utils import MeanStdevFilter, prepare_data
from typing import List, Tuple, Dict
import pickle 
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataProcessor:
    def __init__(self, params):
        self.params: Dict = params
        self.input_filter: MeanStdevFilter = None
        self.input_dim: int = None
        self.output_dim: int = None
        self.output_filter: MeanStdevFilter = None

    def get_data(self, load_dpEn) -> np.ndarray:
        # read data
        data_dicts = pickle.load(open(f"{self.params['dataset_name']}.pkl", 'rb'))
        if self.params['ode_name'] == 'lorenz':
            data = np.concatenate([np.concatenate([data_dict["x"], data_dict['y'], data_dict['z'],data_dict["xdot"], data_dict['ydot'], data_dict['zdot']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)
        elif self.params['ode_name'] == 'glycolytic':    
            #data = np.concatenate([np.concatenate([data_dict["S1"], data_dict['S2'], data_dict['S3'],data_dict["S4"], data_dict['S5'], data_dict['S6'], data_dict['S7'], data_dict["dS1dt"], data_dict['dS2dt'], data_dict['dS3dt'],data_dict["dS4dt"], data_dict['dS5dt'], data_dict['dS6dt'], data_dict['dS7dt']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)
            data = np.concatenate([np.concatenate([data_dict["S1"], data_dict['S2'], data_dict['S3'],data_dict["S4"], data_dict['S5'], data_dict['S6'], data_dict['S7']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)
        elif self.params['ode_name'] == 'LVolt': 
            data = np.concatenate([np.concatenate([data_dict["x"], data_dict['y'],data_dict["xdot"], data_dict['ydot']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)
            #data = np.concatenate([np.concatenate([data_dict["x"], data_dict['y']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)  
        elif self.params['ode_name'] == 'lorenz96': 
            data = np.concatenate([np.concatenate([data_dict["X1"], data_dict['X2'],data_dict["X3"], data_dict['X4'], data_dict['X5']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0) 
        elif self.params['ode_name'] == 'FHNag':
            data = np.concatenate([np.concatenate([data_dict["v"], data_dict['w']], axis=-1)[np.newaxis,:,:] for data_dict in data_dicts], axis=0)     
        # input and output dim
        self.input_dim = data.shape[-1]
        self.output_dim = data.shape[-1]
        self.params['input_dim'] = self.input_dim
        self.params['output_dim'] = self.output_dim

        # prepare deltas for training
        X_prev = data[:,:-1,:]
        X_next = data[:,1:,:]
        delta = X_next - X_prev # output of feedforward model 

        # include 80% of trajectories for training
        total_traj = data.shape[0]
        train_sz = int(total_traj * self.params["train_val_ratio"])
        val_sz = total_traj - train_sz
        rand_perm = torch.randperm(total_traj)

        train_idx = rand_perm[:train_sz]
        val_idx = rand_perm[train_sz:]

        # train and val data
        train_X_traj = X_prev[train_idx,:,:].copy()
        train_Y_traj = X_next[train_idx,:,:].copy()
        train_delta_traj = delta[train_idx,:,:].copy()

        val_X_traj = X_prev[val_idx,:,:].copy()
        val_Y_traj = X_next[val_idx,:,:].copy()
        val_delta_traj = delta[val_idx,:,:].copy()

        train_X = train_X_traj.reshape(-1, self.input_dim).copy()
        train_delta = train_delta_traj.reshape(-1, self.output_dim).copy()

        val_X = val_X_traj.reshape(-1, self.input_dim).copy()
        val_delta = val_delta_traj.reshape(-1, self.output_dim).copy()

        # for Variance Normalized loss
        self.params['norm_loss_wghts'] = torch.FloatTensor(1 / np.var(train_delta, axis=0)).to(device) 

        if not load_dpEn:
            self.input_filter = MeanStdevFilter(self.input_dim) 
            self.output_filter = MeanStdevFilter(self.output_dim)

            self.calculate_mean_var(train_X, train_delta)
            norm_train_X, norm_val_X, norm_train_delta, norm_val_delta = \
                self.prepare_datapoints(train_X, val_X, train_delta, val_delta)
            if self.params['ode_name'] == 'LVolt' or self.params['ode_name'] == 'lorenz':
                dataset = self.prepare_dataclass(train_X_traj, train_Y_traj, val_X_traj, val_Y_traj,\
                                            norm_train_X, train_delta, norm_val_X, val_delta)
            elif self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':    
                dataset = self.prepare_dataclass(train_X_traj, train_Y_traj, val_X_traj, val_Y_traj,\
                                            norm_train_X, norm_train_delta, norm_val_X, norm_val_delta)

            return dataset
        else:
            return None
    
    def prepare_datapoints(self, train_X: np.ndarray, val_X: np.ndarray,\
                            train_delta: np.ndarray, val_delta: np.ndarray):
        
        norm_train_X = prepare_data(train_X, self.input_filter)
        norm_val_X = prepare_data(val_X, self.input_filter)

        # normalize deltas
        norm_train_delta = prepare_data(train_delta, self.output_filter)
        norm_val_delta = prepare_data(val_delta, self.output_filter)        

        return norm_train_X, norm_val_X, norm_train_delta, norm_val_delta
       
    
    def calculate_mean_var(self, input_data: np.ndarray, output_data: np.ndarray) -> None:

        total_points = input_data.shape[0]

        for i in range(total_points):
            self.input_filter.update(input_data[i,:])
        
        for i in range(total_points):
            self.output_filter.update(output_data[i,:])            

        return 
    
    def prepare_dataclass(self, train_X_traj, train_Y_traj, val_X_traj, val_Y_traj,\
                           norm_train_X, train_delta, norm_val_X, val_delta):

        dataset = PICalibData(train_X_traj=torch.Tensor(train_X_traj),
                              train_Y_traj=torch.Tensor(train_Y_traj),
                              val_X_traj=torch.Tensor(val_X_traj),
                              val_Y_traj=torch.Tensor(val_Y_traj),
                              norm_train_X=torch.Tensor(norm_train_X),
                              train_delta=torch.Tensor(train_delta),
                              norm_val_X=torch.Tensor(norm_val_X),
                              val_delta=torch.Tensor(val_delta))
        
        return dataset
    
@dataclass
class PICalibData:
    """Episodes available for calibration"""
    train_X_traj: torch.Tensor
    train_Y_traj: torch.Tensor
    val_X_traj: torch.Tensor 
    val_Y_traj: torch.Tensor
    norm_train_X: torch.Tensor 
    train_delta: torch.Tensor    
    norm_val_X: torch.Tensor 
    val_delta: torch.Tensor 

    
 
