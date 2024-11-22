import torch 
import torch.nn as nn
from data_classes import FcModel, MemoryData
from typing import Tuple, List, Optional
import math
from hflayers import Hopfield
from utils import plot_part_mhn
import pickle
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This file contains the same methods as in conformal_hopfield.py 
# but these ones will TRAIN/VALIDATE the model in batch mode.

class ConformHopfieldBatch(nn.Module):

    def __init__(self, ctx_enc_in, ctx_enc_out, params, which_out: int, model_type:str='transition') -> None:

        super(ConformHopfieldBatch, self).__init__()
        self._params = params
        self._cp_alphas = params['cp_alphas'] if model_type == 'transition' else params['nstep_cp_alphas']
        self._cp_sampling = params['cp_sampling'] if model_type == 'transition' else params['nstep_cp_sampling']
        self._cp_replacement = params['cp_replacement'] if model_type == 'transition' else params['nstep_cp_replacement']
        self._mem_data: MemoryData = None
        self._encoder = FcModel(ctx_enc_in, ctx_enc_out, hidden=(400,400,400))  
        self._hopfield_heads = self._params['num_heads'] # for now
        self._hopfield_hidden = ctx_enc_out
        self._hopfield_beta_factor = 1.0
        self._loss = nn.MSELoss()
        _alphas_list = [0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.17,0.19,0.20,0.21,0.23,0.25,0.30,0.35,0.38,0.40,0.45] #[0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20] 
        self._alphas = _alphas_list[:self._cp_alphas]  #[0.05,0.10,0.15]
        self._which_out = which_out
        self._X_ctx_true_enc_seq = []
        self._errors_samp_seq = []
        self.num_heads = self._params['num_heads']

        self._hopfield = Hopfield( # controlling the storage capacity via the dimension of the associative space
                        batch_first=True,
                        input_size=ctx_enc_out,   # R (Query) size + Alpha/Beta
                        hidden_size=self._hopfield_hidden, # head_dim
                        pattern_size=1,                                                     # Epsilon size (Values)
                        output_size=None,                                                   # Not used because no output projection
                        num_heads=self._hopfield_heads,                                     # k per interval bound
                        stored_pattern_size=ctx_enc_out,  # Stored ctx size (+ eps) (Keys)
                        pattern_projection_size=1,
                        scaling=self._hopfield_beta(ctx_enc_out),
                        # do not pre-process layer input
                        #normalize_stored_pattern=False,
                        #normalize_stored_pattern_affine=False,
                        #normalize_state_pattern=False,
                        #normalize_state_pattern_affine=False,
                        normalize_pattern_projection=False,
                        normalize_pattern_projection_affine=False,
                        # do not post-process layer output
                        disable_out_projection=True                 # To get Heads - one head per epsilon
                        )
        self._encoder.to(device)  
        self._hopfield.to(device) 

    def _hopfield_beta(self, memory_dim):
    # Default beta = Self attention beta = 1 / sqrt(key_dim)
        return self._hopfield_beta_factor / math.sqrt(memory_dim) 
    
    def forward(self, _data: List[torch.Tensor], train: bool):
        """
        Forward pass through the encoder and Hopfield network in batches
        """
        # Will not work when params['mhn_output'] == 'delta', it does 
        # work in conform_hopfield.py file. It did not work well so 
        # not implementing it for batch train/val here.

        # break_steps are not that useful here since we don't have 
        # float('inf') as end-of-sequence token anymore.

        # FOR NOW, THIS METHOD ONLY WORKS WITHOUT PADDING OF float('inf'), 
        # prepare_hopfield_data SKIPS SMALL EPISODES LESS THAN params['calib_horizon']

        X_ctx_true, X_ctx_sim, errors = _data[0].to(device), _data[1].to(device), _data[2].to(device)

        if not train:
            y, y_pred = _data[3].to(device), _data[4].to(device)

        errors = errors[:,:,self._which_out].unsqueeze(-1)
        batch_size = X_ctx_true.shape[0]
        seq_len = X_ctx_true.shape[1]
        no_of_alphas = len(self._alphas)

        # TODO: See if mask is even needed here
        X_ctx_enc_true = self._encoder(X_ctx_true.reshape(-1,X_ctx_true.shape[-1])) # stack batch sequences of context vectors
        X_ctx_enc_sim = self._encoder(X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))

        X_ctx_enc_true = X_ctx_enc_true.reshape(tuple(X_ctx_true.shape))
        X_ctx_enc_sim = X_ctx_enc_sim.reshape(tuple(X_ctx_sim.shape))

        mask = torch.diag(torch.full((X_ctx_true.shape[1],),\
                            fill_value=True, dtype=torch.bool)).to(device)
        if train:
            hopfield_res = self._hopfield((X_ctx_enc_true,X_ctx_enc_sim,errors),\
                                           association_mask=mask) # [batch_sz,seq_len,num_heads]
        
        else:
            assoc_mat = self._hopfield.get_association_matrix((X_ctx_enc_true,\
                                                                X_ctx_enc_sim,errors),\
                                                                    association_mask=mask)  
            # [1,1,300,300] --> [300,300] squeeze the head dim
            # TODO check which dim to squeeze
            scores, y_low, y_high, errors_low, errors_high = [], [], [], [], []
            for head in range(assoc_mat.shape[1]):
                selected_errors = self._process_assoc_mat(assoc_mat[:,head,:,:],\
                                                            errors)   
                # [alphas,batch_size*seq_len]   
                _errors_low, _errors_high = self._select_quantiles(selected_errors,\
                                                                    alphas=self._alphas)
                # [alphas,batch_size*seq_len,1]  
                bounds = self._calculate_bounds(_errors_low, _errors_high, y_pred) 
                _y_low, _y_high = bounds[0], bounds[1]
                #score = self._winkler_score(y, _y_low, _y_high) 

                score = self._mean_quantile_score(_errors_low, _errors_high, y_pred, y)

                # make batch first and alpha second dim
                _y_low = _y_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                _y_high = _y_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
            
                scores.append(score)
                y_low.append(_y_low.unsqueeze(0))
                y_high.append(_y_high.unsqueeze(0))

                _errors_low = _errors_low.unsqueeze(-1) # [alphas,batch_size*seq_len,1]
                _errors_high = _errors_high.unsqueeze(-1)

                _errors_low = _errors_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                _errors_high = _errors_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)                

                errors_low.append(_errors_low.unsqueeze(0)) # add head dim
                errors_high.append(_errors_high.unsqueeze(0))

            y_low = torch.cat(y_low, dim=0)
            y_high = torch.cat(y_high, dim=0)    
            scores = torch.mean(torch.Tensor(scores).to(device))

            errors_low = torch.cat(errors_low, dim=0)
            errors_high = torch.cat(errors_high, dim=0)

        if train:
            return self._loss(hopfield_res, errors.repeat(1,1,self._params['num_heads']))
        else:
            return (scores, y, y_pred, y_low, y_high, errors, errors_low, errors_high)    
        
    def _process_assoc_mat(self, assoc_mat: torch.Tensor, errors: torch.Tensor):
        """
        Samples from softmax distribution over timesteps and 
        return the corresponding errors.
        :param assoc_mat: [batches,seq_len,mem_len] 
        :param errors: [batches,mem_len,1]
        :return selected_errors: [batches*seq_len,num_samples]
        """  
        num_samples = self._cp_sampling[1]

        batch_size = assoc_mat.shape[0]
        seq_len = assoc_mat.shape[1]
        mem_len = assoc_mat.shape[-1]

        if self._cp_sampling[0] == 'topk':
            # pick topk indices
            val, idx = torch.sort(assoc_mat.reshape(-1,assoc_mat.shape[-1]), descending=True, dim=-1) 
            #print(val[40:43,:10].sum(dim=1))
            sampled_indices = idx[:,:num_samples] 
        elif self._cp_sampling[0] == 'sampling':
            sampled_indices = torch.multinomial(assoc_mat.reshape(-1,assoc_mat.shape[-1]),\
                                        num_samples=num_samples,\
                                          replacement=self._cp_replacement) # [batches*seq_len,num_samples]
        # num_samples has idx between [0,mem_len)    
        batch_idx = torch.arange(batch_size).unsqueeze(1).repeat((1,num_samples))\
                                                                .repeat_interleave(seq_len, dim=0)
        # batch_idx = sampled_indices = selected_errors =\
        #                    [batch_size*seq_len,num_samples] Note: batch_size is outer dim]
        
        selected_errors = errors.squeeze(-1)[batch_idx, sampled_indices]

        return selected_errors        
    
    def _select_quantiles(self, selected_errors: torch.Tensor, alphas: List[float]): 
        # alphas e.g. [0.05, 0.1, 0.15]
        """
        Selects quantiles from softmax samples
        :param selected_errors: [batches*seq_len,num_samples] 
        :param alphas: len(self._alphas)
        :return (q_conformal_low,q_conformal_high): ([len(self._alphas),batches*seq_len],...)         
        """  
        # selected_errors = [batch_size*seq_len,num_samples]
        q_conformal_low = torch.zeros((len(alphas),selected_errors.shape[0])).to(device)
        q_conformal_high = torch.zeros((len(alphas),selected_errors.shape[0])).to(device)

        for alpha in range(len(alphas)):
            beta = alphas[alpha] / 2
            q_conformal_low[alpha] = torch.quantile(selected_errors, beta, dim=1)
            q_conformal_high[alpha] = torch.quantile(selected_errors,\
                                                      (1 - alphas[alpha] + beta), dim=1)  

        return q_conformal_low, q_conformal_high
    
    def _calculate_bounds(self, errors_low: torch.Tensor, errors_high: torch.Tensor,\
                                y_pred: torch.Tensor, unc_prop: Optional[bool]=False):
        """
        :param errors_low: [len(self._alphas),batches*seq_len] 
        :param errors_high: [len(self._alphas),batches*seq_len]
        :param y_pred: [batches,seq_len,params['no_of_outputs']+params['reward_dim']]
        :return (y_low,y_high): ([len(self._alphas),batches*seq_len,1],...) 
        """        
        if self._params['mhn_output'] == 'y':                            

            if not unc_prop:
                y_low = errors_low.unsqueeze(2) + y_pred.reshape(-1,y_pred.shape[-1])[:,self._which_out].unsqueeze(1)
                y_high = errors_high.unsqueeze(2) + y_pred.reshape(-1,y_pred.shape[-1])[:,self._which_out].unsqueeze(1)
            else:                    
                y_low = errors_low.unsqueeze(2) + y_pred[0][:,self._which_out].unsqueeze(1)
                y_high = errors_high.unsqueeze(2) + y_pred[0][:,self._which_out].unsqueeze(1) 

        return (y_low, y_high)        
    

    def _winkler_score(self, y: torch.Tensor, y_low: torch.Tensor, y_high: torch.Tensor):
        """
        Calculates winkler score given ground truth and upper/lower bounds
           
        :param y: [batch_size,seq_len,params['no_of_outputs']+params['reward_dim']] 
        :param y_low: [len(self._alphas),batch_size*seq_len,1]
        :param y_high: [len(self._alphas),batch_size*seq_len,1]
        :return          
        """
        batch_size = y.shape[0]
        y = y.reshape(-1,y.shape[-1])[:,self._which_out].unsqueeze(-1)

        score = torch.zeros((len(self._alphas))).to(device)
        for alpha in range(len(self._alphas)): # for all quantiles
            width = (y_high[alpha]-y_low[alpha]).abs()
            undershoot = torch.lt(y, y_low[alpha]).long()
            overshoot = torch.gt(y, y_high[alpha]).long()
            alph = self._alphas[alpha]
            score[alpha] = (width + (undershoot * (y_low[alpha] - y) * 2/alph) +
                             (overshoot * (y - y_high[alpha]) * 2/alph)).sum()
            # taking mean first over batch dim to be consistent with the previous non-batch implementation
            score[alpha] = score[alpha] / batch_size

        return torch.mean(score)
    
    def _mean_quantile_score(self, errors_low: torch.Tensor, errors_high: torch.Tensor,\
                                y_pred: torch.Tensor, y: torch.Tensor):
        """
        Add the mean quantile error to prediction and substract it from ground truth
        :param errors_low: [len(self._alphas),batches*seq_len]
        :param errors_high: [len(self._alphas),batches*seq_len]
        :param y_pred: [batch_size,seq_len,y_dim]
        :param y: [batch_size,seq_len,y_dim]
        :return      
        """  
        errors = torch.cat((errors_low,errors_high), dim=0).mean(dim=0).unsqueeze(0)

        _y_pred = errors.unsqueeze(2) + y_pred.reshape(-1,y_pred.shape[-1])[:,self._which_out].unsqueeze(1)

        _y_pred = _y_pred.reshape(y.shape[0],-1)
        y_true = y[:,:,self._which_out]
        score = torch.mean((y_true-_y_pred)**2)

        return score

    
    def evaluate(self, _data: List[torch.Tensor], batch_idx: int, unc_prop: bool=False, quantile: bool=True):
        """
        Does sample-by-sample evaluation of val data with self._memory
        """
        # TODO: Check this method for unc_prop = True 
        if not unc_prop:
            X_ctx_true, X_ctx_sim, errors = _data[0].to(device), _data[1].to(device), _data[2].to(device)
            y, y_pred = _data[3].to(device), _data[4].to(device)
        else:
            X_ctx_sim, y_pred = _data[0], _data[1]  
        
        batch_size = X_ctx_sim.shape[0]
        seq_len = X_ctx_sim.shape[1]
        feat = 1 # only 1 output for now
        no_of_alphas = len(self._alphas)

        total_mem = self._mem_data.X_ctx_true_train_enc.shape[0]

        mem_len = 6
        start = 0
        # randomly pick mem_len no of examples from memory
        rand_mem = torch.randint(low=0, high=total_mem, size=(mem_len,))

        errors = errors[:,:,self._which_out].unsqueeze(-1)

        if self._params['cp_aggregate'] == 'long_seq':

            # STITCH TOGETHER MANY EPISODES AS ONE BIG EPISODE AND THEN SAMPLE
            if len(self._errors_samp_seq) == 0: 
                # _mem_data.X_ctx_true_train_enc [4978, 200, 61], 
                # _mem_data.error_train [4978, 200, 6]
                self._X_ctx_true_enc_seq = self._mem_data.X_ctx_true_train_enc[start:start+mem_len]\
                                            .reshape(-1,X_ctx_sim.shape[-1]).to(device) # [30k,19]
                self._errors_samp_seq = self._mem_data.error_train[start:start+mem_len]\
                    .reshape(-1,y_pred.shape[-1])[:,self._which_out].unsqueeze(-1).to(device) # [30k,1]

                #self._X_ctx_true_enc_seq = self._mem_data.X_ctx_true_train_enc[rand_mem]\
                #                            .reshape(-1,X_ctx_sim.shape[-1]).to(device) # [30k,19]
                #self._errors_samp_seq = self._mem_data.error_train[rand_mem]\
                #    .reshape(-1,y_pred.shape[-1])[:,self._which_out].unsqueeze(-1).to(device) # [30k,1]
  
            # batch_size no of examples for eval right after training 
            # 1 batch for uncertainty propagation unc_prop=True

            X_ctx_sim_enc = self._encoder(X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))
            # no mask for evaluation
            #mask = torch.diag(torch.full((seq_len,), fill_value=True, dtype=torch.bool)).to(device)
            # no of batches in self._errors_samp_seq and X_ctx_sim_enc should be same

            assoc_mat = self._hopfield.get_association_matrix((self._X_ctx_true_enc_seq.unsqueeze(0),\
                                                                X_ctx_sim_enc.unsqueeze(0),\
                                                                    self._errors_samp_seq.unsqueeze(0))) 
            #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[0][:,:10])
            #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[1][:,:10])

            save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"
            assoc_dict = {}
            if self.num_heads == 1:
                assoc_dict['assoc_mat'] = assoc_mat.squeeze(0).squeeze(0).reshape(X_ctx_sim.shape[0],\
                                                                X_ctx_sim.shape[1],-1).detach().cpu().numpy()
            else:
                assoc_dict['assoc_mat'] = assoc_mat.squeeze(0).reshape(self.num_heads,X_ctx_sim.shape[0],\
                                                                X_ctx_sim.shape[1],-1).detach().cpu().numpy()                    
            assoc_dict['errors_mem'] = self._errors_samp_seq.reshape(mem_len,-1).detach().cpu().numpy()
            #if batch_idx == 0: 
                #pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{batch_idx}_TrainValseq.pkl', 'wb'))
            pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{batch_idx}.pkl', 'wb'))

            y_low, y_high, errors_low, errors_high = [], [], [], []
            for head in range(assoc_mat.shape[1]):     
                selected_errors = self._process_assoc_mat(assoc_mat[:,head,:,:],\
                                                            self._errors_samp_seq.unsqueeze(0))
                #print(torch.min(selected_errors.reshape(4,200,-1)[1,150,:]))
                #print(torch.max(selected_errors.reshape(4,200,-1)[1,150,:]))
                
                _errors_low, _errors_high = self._select_quantiles(selected_errors,\
                                                                    alphas=self._alphas)  # [no_of_alphas,seq_len]
                _bounds = self._calculate_bounds(_errors_low, _errors_high, y_pred)
                _y_low, _y_high = _bounds[0], _bounds[1] 

                _y_low = _y_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                _y_high = _y_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)

                y_low.append(_y_low.unsqueeze(0))
                y_high.append(_y_high.unsqueeze(0))      

                _errors_low = _errors_low.unsqueeze(-1) # [alphas,batch_size*seq_len,1]
                _errors_high = _errors_high.unsqueeze(-1)

                _errors_low = _errors_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                _errors_high = _errors_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)                

                errors_low.append(_errors_low.unsqueeze(0)) # add head dim
                errors_high.append(_errors_high.unsqueeze(0))

        y_low = torch.cat(y_low, axis=0)  # [num_heads,batch_sz,alphas,seq_len,1]
        y_high = torch.cat(y_high, axis=0) 

        errors_low = torch.cat(errors_low, dim=0)
        errors_high = torch.cat(errors_high, dim=0)

        return (y, y_pred, y_low, y_high, errors, errors_low, errors_high)
