from tkinter import Y
import torch 
import torch.nn as nn
from data_classes import FcModel, MemoryData
from typing import Tuple, List, Optional, Dict
import math
from hflayers import Hopfield
from utils import plot_part_mhn, MeanStdevFilter
import pickle
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This file contains the same methods as in conformal_hopfield.py 
# but these ones will TRAIN/VALIDATE the model in batch mode.

class ConformHopfieldBatchSameEnc(nn.Module):

    def __init__(self, ctx_enc_in, ctx_enc_out, params) -> None:

        super(ConformHopfieldBatchSameEnc, self).__init__()
        self._params = params
        self._cp_alphas = params['cp_alphas'] 
        self._cp_sampling = params['cp_sampling']
        self._cp_replacement = params['cp_replacement']
        self.mhn_models = params['num_mhn_models']
        self._mem_data: Dict[int,MemoryData] = {i: None for i in range(self.mhn_models)}
        self._use_base_enc = self._params['use_base_enc']
        self.ctx_enc_out = ctx_enc_out
        self.input_filter: MeanStdevFilter = None
        self.output_filter: MeanStdevFilter = None

        self.heads = nn.ModuleDict({str(i): FcModel(ctx_enc_in,ctx_enc_out, hidden=(100,100,100)) # (400,400) (200,200) (100,100,100)
                                    for i in range(self.mhn_models)})                 
        self._hopfield_heads = self._params['num_heads'] # for now
        self._hopfield_hidden = ctx_enc_out
        self._hopfield_beta_factor = 1.0
        self._loss = nn.MSELoss()
        _alphas_list = [0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.17,0.19,0.20,0.21,0.28,0.35,0.40,0.45] #[0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20] 
        self._alphas = _alphas_list[:self._cp_alphas]  #[0.05,0.10,0.15]
        self._X_mem: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}
        self._X_enc_mem: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}
        self._Y_mem: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}

        self._hopfield = nn.ModuleDict({str(i): Hopfield( # controlling the storage capacity via the dimension of the associative space
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
                            pattern_projection_as_static=True,
                            disable_out_projection=True, # To get Heads - one head per epsilon
                            ) for i in range(self.mhn_models)})

        for i in range(self.mhn_models): 
            self._hopfield[str(i)].to(device)
            self.heads[str(i)].to(device) 

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

        # X_ctx_true_m: [4, 6, 5000, 20]
        X_d, Y_d = _data[0].to(device), _data[1].to(device)

        batch_size = X_d.shape[0]
        seq_len = X_d.shape[2]
        no_of_alphas = len(self._alphas)

        # TODO: See if mask is even needed here
        #mask = torch.diag(torch.full((seq_len,),\
        #                    fill_value=True, dtype=torch.bool)).to(device)
        if train:
            losses = torch.zeros((self.mhn_models), requires_grad=True).to(device)
            for hop_id in range(len(self._hopfield)):
                self._which_out = 0

                X_m, Y_m = X_d[:,hop_id,:,:], Y_d[:,hop_id,:,:]

                X_m_enc = self.heads[str(hop_id)](X_m.reshape(-1,X_m.shape[-1]))

                X_m_enc = X_m_enc.reshape(batch_size,seq_len,-1)
                hopfield_res, attn_wghts, _, _ = self._hopfield[str(hop_id)]((X_m_enc,X_m_enc,Y_m))#,\
                                            #association_mask=mask) # [batch_sz,seq_len,num_heads]
                losses[hop_id] = self._loss(hopfield_res,\
                                            Y_m.repeat(1,1,self._params['num_heads']))      
        else:
            scores_out, y_low_out, y_high_out = [], [], [] # cat along output dim (no. of attn models)
            for hop_id in range(len(self._hopfield)):

                self._which_out = 0

                X_m, Y_m = X_d[:,hop_id,:,:], Y_d[:,hop_id,:,:]

                X_m_enc = self.heads[str(hop_id)](X_m.reshape(-1,X_m.shape[-1]))

                X_m_enc = X_m_enc.reshape(batch_size,seq_len,-1)

                assoc_mat = self._hopfield[str(hop_id)].get_association_matrix((X_m_enc,\
                                                     X_m_enc,Y_m))#, association_mask=mask)  
                # [1,1,300,300] --> [300,300] squeeze the head dim
                # TODO check which dim to squeeze
                scores, y_low, y_high = [], [], [] # cat along head dim
                for head in range(assoc_mat.shape[1]):
                    selected_y = self._process_assoc_mat(assoc_mat[:,head,:,:], Y_m)   
                    # [alphas,batch_size*seq_len]   
                    _y_low, _y_high = self._select_quantiles(selected_y,\
                                                                        alphas=self._alphas)
                    # [alphas,batch_size*seq_len,1]  
                    #score = self._winkler_score(y, _y_low, _y_high) 
                    score = self._mean_quantile_score(_y_low, _y_high, Y_m)

                    # make batch first and alpha second dim
                    _y_low = _y_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                    _y_high = _y_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                
                    scores.append(score)
                    y_low.append(_y_low.unsqueeze(0)) # add head dim
                    y_high.append(_y_high.unsqueeze(0))

                y_low = torch.cat(y_low, dim=0)
                y_high = torch.cat(y_high, dim=0) # concatenate along head dim   
                scores = torch.mean(torch.Tensor(scores).to(device))

                # APPEND EACH OUTPUT RESULTS FOR CAT ALONG OUT DIM
                scores_out.append(scores) # DON'T TAKE MEAN ACROSS OUTPUTS, KEEP EACH OUTPUT SCORE SEPARATE
                y_low_out.append(y_low)
                y_high_out.append(y_high)

            scores_out = torch.Tensor(scores_out)
            y_low_out = torch.cat(y_low_out, dim=-1)   # cat along output dim
            y_high_out = torch.cat(y_high_out, dim=-1) 

        if train:
            return losses
        else:
            y = Y_d[:,:,:,self._which_out].permute(0,2,1) # [batch,model_no,seq_len] -> [batch,seq_len,model_no]

            return (scores_out, y, y_low_out, y_high_out)    
        
    def _process_assoc_mat(self, assoc_mat: torch.Tensor, Y_m: torch.Tensor):
        """
        :param assoc_mat: [batches,seq_len,mem_len] 
        :param Y_m: [batches,mem_len,1]
        :return selected_y: [batches*seq_len,num_samples]
        Samples from softmax distribution over timesteps and 
        return the corresponding Y.
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
        
        selected_y = Y_m.squeeze(-1)[batch_idx, sampled_indices]

        return selected_y
    
    def _select_quantiles(self, selected_y: torch.Tensor, alphas: List[float]): 
        # alphas e.g. [0.05, 0.1, 0.15]
        """
        :param selected_y: [batches*seq_len,num_samples] 
        :param alphas: len(self._alphas)
        :return (q_conformal_low,q_conformal_high): ([len(self._alphas),batches*seq_len],...)         
        """  
        # selected_errors = [batch_size*seq_len,num_samples]
        q_conformal_low = torch.zeros((len(alphas),selected_y.shape[0])).to(device)
        q_conformal_high = torch.zeros((len(alphas),selected_y.shape[0])).to(device)

        for alpha in range(len(alphas)):
            beta = alphas[alpha] / 2
            q_conformal_low[alpha] = torch.quantile(selected_y, beta, dim=1)
            q_conformal_high[alpha] = torch.quantile(selected_y,\
                                                      (1 - alphas[alpha] + beta), dim=1)  

        return q_conformal_low, q_conformal_high        

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
    
    def _mean_quantile_score(self, y_low: torch.Tensor, y_high: torch.Tensor,\
                                y: torch.Tensor):
        """
        Add the mean quantile error to prediction and substract it from ground truth
        :param y_low: [len(self._alphas),batches*seq_len]
        :param y_high: [len(self._alphas),batches*seq_len]
        :param y: [batch_size,seq_len,y_dim]
        :return      
        """  
        y = y[:,:,self._which_out]
        y_pred = torch.cat((y_low,y_high), dim=0).mean(dim=0).unsqueeze(0)

        y_pred = y_pred.reshape(y.shape[0],-1)
        score = torch.mean((y-y_pred)**2)

        return score

    
    def evaluate(self, _data: List[torch.Tensor], batch_idx: int):
        """
        Does sample-by-sample evaluation of val data with self._mem_data
        """
        X_d, Y_d = _data[0].to(device), _data[1].to(device)
        
        batch_size = X_d.shape[0]
        seq_len = X_d.shape[2]
        feat = X_d.shape[-1] # only 1 output for now
        no_of_alphas = len(self._alphas)

        total_mem = self._mem_data[0]._X_enc_mem.shape[0]

        mem_len = 100 # BE CAREFUL MIGHT END UP PUSHING INF VALUES SINCE THE REMAINING VALUES ARE float('inf) Check encode_train
        start = 0
        # randomly pick mem_len no of examples from memory
        rand_mem = torch.randint(low=0, high=total_mem, size=(mem_len,))

        if self._params['cp_aggregate'] == 'long_seq':

            # STITCH TOGETHER MANY EPISODES AS ONE BIG EPISODE AND THEN SAMPLE
            for m in range(self.mhn_models):
                if len(self._Y_mem[m]) == 0: 
                    # _mem_data.X_ctx_true_train_enc [4978, 200, 61], 
                    # _mem_data.error_train [4978, 200, 6]

                    self._X_mem[m] = self._mem_data[m]._X_mem[start:start+mem_len]\
                                                .reshape(-1,feat).to(device)
                    self._X_enc_mem[m] = self._mem_data[m]._X_enc_mem[start:start+mem_len]\
                                                .reshape(-1,self.ctx_enc_out).to(device) # [30k,19]
                    self._Y_mem[m] = self._mem_data[m]._Y_mem[start:start+mem_len]\
                        .reshape(-1,Y_d.shape[-1]).to(device) # [30k,6]

            # no mask for evaluation
            #mask = torch.diag(torch.full((seq_len,), fill_value=True, dtype=torch.bool)).to(device)
            # no of batches in self._Y_mem and X_ctx_sim_enc should be same
            scores_out, y_low_out, y_high_out = [], [], [] # cat along output dim (no. of attn models)
            
            for hop_id in range(len(self._hopfield)):

                self._which_out = 0

                X_m, Y_m = X_d[:,hop_id,:,:], Y_d[:,hop_id,:,:]

                X_m_enc = self.heads[str(hop_id)](X_m.reshape(-1,X_m.shape[-1]))   
                
                assoc_mat = self._hopfield[str(hop_id)].get_association_matrix((self._X_enc_mem[hop_id].unsqueeze(0),\
                                                                    X_m_enc.unsqueeze(0),\
                                self._Y_mem[hop_id][:,self._which_out].unsqueeze(0).unsqueeze(-1))) 
                #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[0][:,:10])
                #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[1][:,:10])

                save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self._params['num_mhn_models']}encHeads_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['seq_len']}seqlen_{self._params['num_heads']}heads"
                assoc_dict = {}

                if self._params['ood_pred']:
                    assoc_dict['X_m'] = X_m.detach().cpu().numpy()
                    assoc_dict['Y_m'] = Y_m.detach().cpu().numpy()                    
                else:    
                    assoc_dict['X_m'] = self.input_filter.invert_torch(X_m.reshape(-1,1)).reshape(tuple(X_m.shape)).detach().cpu().numpy()
                    assoc_dict['Y_m'] = self.output_filter.invert_torch(Y_m.reshape(-1,1)).reshape(tuple(Y_m.shape)).detach().cpu().numpy()
                assoc_dict['assoc_mat'] = assoc_mat.squeeze(0).squeeze(0).reshape(X_m.shape[0],\
                                                                X_m.shape[1],-1).detach().cpu().numpy()
                assoc_dict['x_mem'] = self.input_filter.invert_torch(self._X_mem[hop_id]).reshape(mem_len,-1,X_m.shape[-1]).detach().cpu().numpy()
                assoc_dict['y_mem'] = self.output_filter.invert_torch(self._Y_mem[hop_id]).reshape(mem_len,-1,Y_m.shape[-1]).detach().cpu().numpy()
                #if batch_idx == 0: 
                    #pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{batch_idx}_TrainValseq.pkl', 'wb'))
                if self._params['ood_pred']:
                    pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{hop_id}out_OOD.pkl', 'wb'))
                else:    
                    pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{hop_id}out_{batch_idx}bs.pkl', 'wb'))

                y_low, y_high, errors_low, errors_high = [], [], [], []
                for head in range(assoc_mat.shape[1]):     
                    selected_y = self._process_assoc_mat(assoc_mat[:,head,:,:],\
                                                                self._Y_mem[hop_id][:,self._which_out].unsqueeze(0).unsqueeze(-1))
                    
                    _y_low, _y_high = self._select_quantiles(selected_y,\
                                                                        alphas=self._alphas)  # [no_of_alphas,seq_len]

                    _y_low = _y_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                    _y_high = _y_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)

                    y_low.append(_y_low.unsqueeze(0))
                    y_high.append(_y_high.unsqueeze(0))      

                y_low = torch.cat(y_low, axis=0)  # [num_heads,batch_sz,alphas,seq_len,1]
                y_high = torch.cat(y_high, axis=0) 

                # APPEND EACH OUTPUT RESULTS FOR CAT ALONG OUT DIM
                y_low_out.append(y_low)
                y_high_out.append(y_high)

            y_low_out = torch.cat(y_low_out, dim=-1)   
            y_high_out = torch.cat(y_high_out, dim=-1) 
             
        y = Y_d[:,:,:,self._which_out].permute(0,2,1) # [batch,model_no,seq_len] -> [batch,seq_len,model_no]

        return (y, y_low_out, y_high_out)
