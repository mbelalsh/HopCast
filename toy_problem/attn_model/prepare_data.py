from data_classes import PICalibData
import torch 
import numpy as np
import pandas as pd
from utils import MeanStdevFilter, prepare_data
from typing import List, Tuple
import pickle 
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataProcessor:
    def __init__(self, params):
        self.params = params
        self.input_filter = None
        self.input_dim = None
        self.output_dim = None    
        self.num_mhn_models = self.params['num_mhn_models']

    def get_data(self) -> PICalibData:
        data_state_dict = pickle.load(open(self.params['data_path'], 'rb'))

        calib_data = PICalibData(X=data_state_dict['x_data'], 
                                 Y=data_state_dict['y_data'])  
        calib_data.X = calib_data.X.reshape(-1,1)
        calib_data.Y = calib_data.Y.reshape(-1,1)
        self.input_dim = calib_data.X.shape[-1]
        self.output_dim = calib_data.Y.shape[-1]

        self.params['input_dim'] = self.input_dim
        self.params['output_dim'] = self.output_dim
         
        self.input_filter = MeanStdevFilter(self.input_dim)
        self.output_filter = MeanStdevFilter(self.output_dim)

        return calib_data
    
    def calculate_mean_var(self, _data: np.ndarray, _dim: int, _filter: MeanStdevFilter) -> None:

        _data = _data.reshape(-1, _dim)

        total_points = _data.shape[0]

        for i in range(total_points):
            _filter.update(_data[i,:])

        return 
    
    def normalize_data(self, calib_data: PICalibData) -> PICalibData:

        x_data = np.array(calib_data.X)
        y_data = np.array(calib_data.Y)
        
        self.calculate_mean_var(x_data, self.input_dim, self.input_filter)
        self.calculate_mean_var(y_data, self.output_dim, self.output_filter)

        self.params['input_filter'] = self.input_filter
        self.params['output_filter'] = self.output_filter
        
        calib_data.X_norm = torch.Tensor(prepare_data(x_data, self.input_filter))
        calib_data.Y_norm = torch.Tensor(prepare_data(y_data, self.output_filter))

        return calib_data
    
    def mix_data(self, calib_data: PICalibData) \
                                -> Tuple[PICalibData,...]:
        
        calib_data_all = [(PICalibData(None,None,None,None),
                           PICalibData(None,None,None,None)) 
                           for _ in range(self.num_mhn_models)]

        data_points = calib_data.X.shape[0]
        train_len = int(0.8*data_points)
        val_len = data_points - train_len
        seq_len = self.params['seq_len']

        for model in range(self.num_mhn_models):
            rand_perm = torch.randperm(data_points)

            rand_train = rand_perm[:train_len]
            rand_val = rand_perm[train_len:]

            extra_points_train = (train_len) % seq_len 
            extra_points_val = (val_len) % seq_len 

            calib_data_all[model][0].X = calib_data.X[rand_train,:][:-extra_points_train if extra_points_train != 0 else None,:].reshape(-1,seq_len,self.input_dim)
            calib_data_all[model][0].Y = calib_data.Y[rand_train,:][:-extra_points_train if extra_points_train != 0 else None,:].reshape(-1,seq_len,self.output_dim)
            calib_data_all[model][0].X_norm = calib_data.X_norm[rand_train,:][:-extra_points_train if extra_points_train != 0 else None,:].reshape(-1,seq_len,self.input_dim)
            calib_data_all[model][0].Y_norm = calib_data.Y_norm[rand_train,:][:-extra_points_train if extra_points_train != 0 else None,:].reshape(-1,seq_len,self.output_dim)

            calib_data_all[model][1].X = calib_data.X[rand_val,:][:-extra_points_val if extra_points_val != 0 else None,:].reshape(-1,seq_len,self.input_dim)
            calib_data_all[model][1].Y = calib_data.Y[rand_val,:][:-extra_points_val if extra_points_val != 0 else None,:].reshape(-1,seq_len,self.output_dim)
            calib_data_all[model][1].X_norm = calib_data.X_norm[rand_val,:][:-extra_points_val if extra_points_val != 0 else None,:].reshape(-1,seq_len,self.input_dim)
            calib_data_all[model][1].Y_norm = calib_data.Y_norm[rand_val,:][:-extra_points_val if extra_points_val != 0 else None,:].reshape(-1,seq_len,self.output_dim)

        return calib_data_all

    def pack_rand_data(self, calib_data_all: List[Tuple[PICalibData,...]]) -> Tuple[PICalibData]:
        """Packs the random data Tuples into a single Tuple of PICalibData object"""
        
        # PICalibData contains the normalized rand data here for all mhn models
        calib_train = PICalibData(X=None,
                                    Y=None,
                                    X_norm=None,
                                    Y_norm=None)
        calib_val = PICalibData(X=None,
                                    Y=None,
                                    X_norm=None,
                                    Y_norm=None)
        
        calib_train.X = torch.cat([calib_data_all[i][0].X.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_train.Y = torch.cat([calib_data_all[i][0].Y.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_train.X_norm = torch.cat([calib_data_all[i][0].X_norm.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_train.Y_norm = torch.cat([calib_data_all[i][0].Y_norm.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)

        calib_val.X = torch.cat([calib_data_all[i][1].X.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_val.Y = torch.cat([calib_data_all[i][1].Y.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_val.X_norm = torch.cat([calib_data_all[i][1].X_norm.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)
        calib_val.Y_norm = torch.cat([calib_data_all[i][1].Y_norm.unsqueeze(0) for i in range(self.num_mhn_models)], dim=0)

        return (calib_train, calib_val)    

    
  
