from collections import deque, namedtuple
from statistics import variance
from typing import Tuple, List, Dict

import time
import sys, os
from copy import deepcopy

from sklearn import calibration
import tree
from prepare_data import PICalibData
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from prepare_data import DataProcessor
import pandas as pd
from utils import check_or_make_folder
import pickle
import scipy.stats as stats
from model import Model, EnsembleTransitionDataset, TransitionDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Ensemble(object):
    def __init__(self, params: Dict, data_proc: DataProcessor, dataset: PICalibData):

        self.params = params
        self.input_dim = data_proc.input_dim
        self.output_dim = data_proc.output_dim
        self.models = {i: Model(input_dim=self.input_dim,
                                output_dim=self.output_dim,
                                params=params,
                                h=params['num_nodes'],
                                seed=params['seed'] + i,
                                l2_reg_multiplier=params['l2_reg_multiplier'],
                                num=i)
                       for i in range(params['num_models'])}
        #print(f"FIrst loss: {self.models[0].model.state_dict()['logvar.bias']}")
        self.num_models = params['num_models']
        self.train_val_ratio = params['train_val_ratio']
        self._model_lr = params['model_lr'] if 'model_lr' in params else 0.001
        weights = [weight for model in self.models.values() for weight in model.weights]
        if self.params['bayesian']:
            # initializing the max and min logvar to bound the predicted variance  
            self.max_logvar = torch.full((self.output_dim,), 0.5, requires_grad=True, device=device)
            self.min_logvar = torch.full((self.output_dim,), -10.0, requires_grad=True, device=device)
            weights.append({'params': [self.max_logvar]}) #  learning the max and min logvar
            weights.append({'params': [self.min_logvar]}) 
            self.set_model_logvar_limits()     
            
        self.optimizer = torch.optim.Adam(weights, lr=self._model_lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.3, verbose=False)
        self.input_filter = data_proc.input_filter
        self.output_filter = data_proc.output_filter
        self.dataset = dataset

        if self.params['bayesian']:
            self._model_id = f"{params['seed']}seed_dpEn_{params['num_nodes']}nod_{params['num_layers']}lay_{params['model_lr']}lr_{params['l2_reg_multiplier']}l2_bayesian"
        else:    
            self._model_id = f"{params['seed']}seed_dpEn_{params['num_nodes']}nod_{params['num_layers']}lay_{params['model_lr']}lr_{params['l2_reg_multiplier']}l2"

    
    def set_loaders(self):            

        ########## MIX VALIDATION AND TRAINING   

        self.seq_input_train = self.dataset.train_X_traj # [4978, 200, 11]
        self.seq_input_val = self.dataset.val_X_traj   # [262, 200, 11]

        self.seq_output_train = self.dataset.train_Y_traj # [4978, 200, 1]
        self.seq_output_val = self.dataset.val_Y_traj # [262, 200, 1]

        train_rand_perm = torch.randperm(self.dataset.norm_train_X.shape[0])
        val_rand_perm = torch.randperm(self.dataset.norm_val_X.shape[0])
        
        self.rand_input_train = self.dataset.norm_train_X[train_rand_perm] # [995600, 11]
        self.rand_input_val = self.dataset.norm_val_X[val_rand_perm] # [52400, 11]

        self.rand_output_train = self.dataset.train_delta[train_rand_perm] # [995600, 1]
        self.rand_output_val = self.dataset.val_delta[val_rand_perm] # [52400, 1]

        batch_size = self.params['batch_size']

        self.transition_loader = DataLoader(
            EnsembleTransitionDataset(self.rand_input_train, self.rand_output_train, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        
        validate_dataset = TransitionDataset(self.rand_input_val, self.rand_output_val)
        sampler = SequentialSampler(validate_dataset)
        self.validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        )

    def train_model(self, max_epochs: int = 100, save_model=False, min_model_epochs=None):
        self.current_best_losses = np.zeros(          # params['num_models'] = 7
            self.params['num_models']) + sys.maxsize  # weird hack (YLTSI), there's almost surely a better way...
        self.current_best_weights = [None] * self.params['num_models']
        val_improve = deque(maxlen=4)
        lr_lower = False
        min_model_epochs = 0 if not min_model_epochs else min_model_epochs

        ### check validation before first training epoch
        improved_any, iter_best_loss = self.check_validation_losses(self.validation_loader)
        val_improve.append(improved_any)
        best_epoch = 0
        model_idx = 0
        print('Epoch: %s, Total Loss: N/A' % (0))
        print('Validation Losses:')
        #print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
        for i in range(max_epochs):  # 1000
            t0 = time.time()
            total_loss = 0
            loss = 0
            step = 0
            # value to shuffle dataloader rows by so each model sees different data
            perm = np.random.choice(self.num_models, size=self.num_models, replace=False)
            for x_batch, y_batch in self.transition_loader:  # state_action, y

                x_batch = x_batch[:, perm]
                y_batch = y_batch[:, perm]
                step += 1
                for idx in range(self.num_models):
                    loss += self.models[idx].train_model_forward(x_batch[:, idx], y_batch[:, idx])  
                total_loss = loss.item()
                if self.params['bayesian']:
                    loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()
                loss = 0    
  
            t1 = time.time()
            print("Epoch training took {} seconds".format(t1 - t0))
            if (i + 1) % 1 == 0:
                improved_any, iter_best_loss = self.check_validation_losses(self.validation_loader)
                print('Epoch: {}, Total Loss: {}'.format(int(i + 1), float(total_loss)))
                print('Validation Losses:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(iter_best_loss)))
                print('Best Validation Losses So Far:')
                print('\t'.join('M{}: {}'.format(i, loss) for i, loss in enumerate(self.current_best_losses)))
                val_improve.append(improved_any)
                if improved_any:
                    best_epoch = (i + 1)
                    print('Improvement detected this epoch.')
                else:
                    epoch_diff = i + 1 - best_epoch
                    plural = 's' if epoch_diff > 1 else ''
                    print('No improvement detected this epoch: {} Epoch{} since last improvement.'.format(epoch_diff,plural))
                                                                                          
                if len(val_improve) > 3:
                    if not any(np.array(val_improve)[1:]):  # If no improvement in the last 5 epochs
                        # assert val_improve[0]
                        if (i >= min_model_epochs):
                            print('Validation loss stopped improving at %s epochs' % (best_epoch))
                            for model_index in self.models:
                                self.models[model_index].load_state_dict(self.current_best_weights[model_index])
                            #self._select_elites(validation_loader)
                            if save_model:
                                self._save_model()
                            return
                        elif not lr_lower:
                            self._lr_scheduler.step()
                            lr_lower = True
                            val_improve = deque(maxlen=6)
                            val_improve.append(True)
                            print("Lowering Adam Learning for fine-tuning")
                t2 = time.time() 
                print("Validation took {} seconds".format(t2 - t1))
        self._save_model()        
        #self._select_elites(validation_loader)

    def _save_model(self):
        """
        Method to save model after training is completed
        """
        print("Saving model checkpoint...")
        check_or_make_folder("./checkpoints")
        check_or_make_folder(f"./checkpoints/{self.params['dataset_name']}")
        save_dir = f"./checkpoints/{self.params['dataset_name']}/{self._model_id}"
        check_or_make_folder(save_dir)
        # Create a dictionary with pytorch objects we need to save, starting with models
        torch_state_dict = {'model_{}_state_dict'.format(i): w for i, w in enumerate(self.current_best_weights)}
        # Then add logvariance limit terms
        if self.params['bayesian']:
            torch_state_dict['logvar_min'] = self.min_logvar
            torch_state_dict['logvar_max'] = self.max_logvar
        # Save Torch files
        torch.save(torch_state_dict, save_dir + "/torch_model_weights.pt")
        print("Saving train and val data...")        
        data_state_dict = {'seq_input_train': self.seq_input_train, 
                           'seq_input_val': self.seq_input_val,
                           'seq_output_train': self.seq_output_train,
                           'seq_output_val': self.seq_output_val,
                           'rand_input_train': self.rand_input_train,
                           'rand_input_val': self.rand_input_val,
                           'rand_output_train': self.rand_output_train,
                           'rand_output_val': self.rand_output_val,
                           'input_filter': self.input_filter,
                           'output_filter': self.output_filter,
                           'best_losses': self.current_best_losses}   
        pickle.dump(data_state_dict, open(save_dir + '/model_data.pkl', 'wb'))

    def check_validation_losses(self, validation_loader):
        improved_any = False
        current_losses, current_weights = self._get_validation_losses(validation_loader, get_weights=True)
        improvements = ((self.current_best_losses - current_losses) / self.current_best_losses) > 0.01
        for i, improved in enumerate(improvements):
            if improved:
                self.current_best_losses[i] = current_losses[i]
                self.current_best_weights[i] = current_weights[i]
                improved_any = True
        return improved_any, current_losses

    def _get_validation_losses(self, validation_loader, get_weights=True):
        best_losses = []
        best_weights = []
        for model in self.models.values():
            best_losses.append(model.get_validation_loss(validation_loader))
            if get_weights:
                best_weights.append(deepcopy(model.state_dict()))
        best_losses = np.array(best_losses)
        return best_losses, best_weights   

    def set_model_logvar_limits(self):

        for model in self.models.values():
            model.model.update_logvar_limits(self.max_logvar, self.min_logvar) 

    def load_model(self):
        """loads the trained models"""
        self.model_dir = f"./checkpoints/{self.params['dataset_name']}/{self._model_id}"
        torch_state_dict = torch.load(self.model_dir + '/torch_model_weights.pt', map_location=device)
        for i in range(self.num_models):
            self.models[i].load_state_dict(torch_state_dict['model_{}_state_dict'.format(i)])
        if self.params['bayesian']:
            self.min_logvar = torch_state_dict['logvar_min']
            self.max_logvar = torch_state_dict['logvar_max']

            self.set_model_logvar_limits()

        # loading train and val data     
        data_state_dict = pickle.load(open(self.model_dir + '/model_data.pkl', 'rb'))
        self.seq_input_train = data_state_dict['seq_input_train']
        self.seq_input_val = data_state_dict['seq_input_val']
        self.seq_output_train = data_state_dict['seq_output_train']
        self.seq_output_val = data_state_dict['seq_output_val'] 
        self.rand_input_train = data_state_dict['rand_input_train']
        self.rand_input_val = data_state_dict['rand_input_val']
        self.rand_output_train = data_state_dict['rand_output_train']
        self.rand_output_val = data_state_dict['rand_output_val']
        self.input_filter = data_state_dict['input_filter']
        self.output_filter = data_state_dict['output_filter']
        self.current_best_losses = data_state_dict['best_losses']
        print(f"The best losses during training {self.current_best_losses}")

        # reinitialize the train and val loaders 
        batch_size = self.params['batch_size']
        self.timesteps = self.seq_input_train.shape[1]
        if self.params['ode_name'] == 'lorenz':
            self.traj_batch = 64
        elif self.params['ode_name'] == 'glycolytic':
            self.traj_batch = 8
        elif self.params['ode_name'] == 'LVolt':
            self.traj_batch = 8
        elif self.params['ode_name'] == 'lorenz96':    
            self.traj_batch = 8
        elif self.params['ode_name'] == 'FHNag':
            self.traj_batch = 8    

        self.transition_loader = DataLoader(
            EnsembleTransitionDataset(self.rand_input_train, self.rand_output_train, n_models=self.num_models),
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True
        )
        # TRAIN -- SEQ AND MIXED DATA BOTH CONTAIN THE SAME TRAJECTORIES
        # VALIDATION -- SEQ AND MIXED DATA BOTH CONTAIN THE SAME TRAJECTORIES
        validate_dataset = TransitionDataset(self.rand_input_val, self.rand_output_val)
        sampler = SequentialSampler(validate_dataset)
        self.validation_loader = DataLoader(
            validate_dataset,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True
        ) 

        seq_train_dataset = TransitionDataset(self.seq_input_train, self.seq_output_train)
        sampler = SequentialSampler(seq_train_dataset)
        self.seq_train_loader = DataLoader(
            seq_train_dataset,
            sampler=sampler,
            batch_size=self.traj_batch,
            pin_memory=True
        ) 

        #self.seq_input_val = torch.concatenate([self.seq_input_train, self.seq_input_val], dim=0)
        #self.seq_output_val = torch.concatenate([self.seq_output_train, self.seq_output_val], dim=0)

        seq_val_dataset = TransitionDataset(self.seq_input_val, self.seq_output_val)
        sampler = SequentialSampler(seq_val_dataset)
        self.seq_val_loader = DataLoader(
            seq_val_dataset,
            sampler=sampler,
            batch_size=self.traj_batch,
            pin_memory=True
        ) 

        return 
           



            


        
