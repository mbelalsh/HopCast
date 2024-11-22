from os import error
from statistics import variance
from matplotlib import axes
from sympy import true
import torch 
import torch.nn as nn
from data_classes import FcModel, MemoryData
from typing import Tuple, List, Optional, Dict
import math
from hflayers import Hopfield
from utils import plot_part_mhn, MeanStdevFilter
import torch.nn.functional as F
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
        self.ctx_enc_in = ctx_enc_in
        self.ctx_enc_out = ctx_enc_out
        self._which_out = params['out']
        if self._use_base_enc:
            self._encoder = FcModel(ctx_enc_in, ctx_enc_out, hidden=(600,600)) 
            self.heads = nn.ModuleDict({str(i): FcModel(ctx_enc_out,ctx_enc_out, hidden=(200,200)) # (400,400) (200,200)
                                        for i in range(self.mhn_models)}) 
        else: 
            self.heads = nn.ModuleDict({str(i): FcModel(ctx_enc_in,ctx_enc_out, hidden=(100,)) # (100,) (400,400) (200,200) (400,400,400)
                                        for i in range(self.mhn_models)})                 
        self._hopfield_heads = self._params['num_heads'] # for now
        self._hopfield_hidden = ctx_enc_out
        self._hopfield_beta_factor = 1.0
        self._loss = nn.MSELoss()
        _alphas_list = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45] #[0.05,0.06,0.08,0.1,0.12,0.14,0.15,0.17,0.19,0.20,0.21] #[0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20] 
        self._alphas = _alphas_list[:self._cp_alphas]  #[0.05,0.10,0.15]
        self._wink_alphas = self._alphas[0] # PI-width and Winkler score for 90% PI
        self._X_ctx_true_seq: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}
        self._X_ctx_true_enc_seq: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}
        self._errors_samp_seq: Dict[int, torch.Tensor] = {i: [] for i in range(self.mhn_models)}
        self.input_filter: MeanStdevFilter = self._params['input_filter']

        self._hopfield: Dict[str, Hopfield] = nn.ModuleDict({str(i): Hopfield( # controlling the storage capacity via the dimension of the associative space
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
                            disable_out_projection=True                 # To get Heads - one head per epsilon
                            ) for i in range(self.mhn_models)})
        if self._use_base_enc:
            self._encoder.to(device) 
        for i in range(self.mhn_models): 
            self._hopfield[str(i)].to(device)
            self.heads[str(i)].to(device) 

    def _hopfield_beta(self, memory_dim):
        if self._params['memory_dim']:
            memory_dim = self._params['memory_dim']
        print("*********** Temprature ***********")
        print(f"The temperature is: {memory_dim}")   
    # Default beta = Self attention beta = 1 / sqrt(key_dim)
        return self._hopfield_beta_factor / math.sqrt(memory_dim) 
    
    def forward(self, _data: List[torch.Tensor], train: bool, epoch: int, batch_id: int):
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
        X_ctx_true_m, X_ctx_sim_m, errors_m = _data[0].to(device), _data[1].to(device), _data[2].to(device)

        if not train:
            y_m, y_pred_m = _data[3].to(device), _data[4].to(device)

        batch_size = X_ctx_true_m.shape[0]
        seq_len = X_ctx_true_m.shape[2]
        no_of_alphas = len(self._alphas)

        # TODO: See if mask is even needed here
        mask = torch.diag(torch.full((seq_len,),\
                           fill_value=True, dtype=torch.bool)).to(device)
        #mask = None
        if train:
            losses = torch.zeros((self.mhn_models), requires_grad=True).to(device)
            for hop_id in range(len(self._hopfield)):
                #self._which_out = 0

                X_ctx_true, X_ctx_sim, errors = X_ctx_true_m[:,hop_id,:,:],\
                                                    X_ctx_sim_m[:,hop_id,:,:],\
                                                        errors_m[:,hop_id,:,:]

                if self._use_base_enc:
                    X_ctx_enc_true = self._encoder(X_ctx_true.reshape(-1,X_ctx_true.shape[-1])) # stack batch sequences of context vectors
                    X_ctx_enc_sim = self._encoder(X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))

                    X_ctx_enc_head_true = self.heads[str(hop_id)](X_ctx_enc_true)
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_enc_sim)
                else:
                    X_ctx_enc_head_true = self.heads[str(hop_id)](X_ctx_true.reshape(-1,X_ctx_true.shape[-1]))
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))                        

                X_ctx_enc_head_true = X_ctx_enc_head_true.reshape(batch_size,seq_len,-1)
                X_ctx_enc_head_sim = X_ctx_enc_head_sim.reshape(batch_size,seq_len,-1)
                #print(errors[:,:15,self._which_out])

                hopfield_res, attn_wghts, _, _ = self._hopfield[str(hop_id)]((X_ctx_enc_head_true,X_ctx_enc_head_sim,\
                                                errors[:,:,self._which_out].unsqueeze(-1)),\
                                            association_mask=mask) # [batch_sz,seq_len,num_heads]
                #if epoch == 5 and batch_id == 0:
                #    print("************ Train ************")
                #    print(f"Hop: {hopfield_res[0,:25,:].squeeze(-1)}")
                #    #if hop_id == 0:
                #    print(f"True: {errors[0,:25,self._which_out]}")
                #    print(f"predicted: {torch.sum(attn_wghts[0,:25,:]*errors[0,:,self._which_out], dim=1)}")
                    #print(attn_wghts[0,0,:])    
                    
                losses[hop_id] = self._loss(hopfield_res,\
                                   errors[:,:,self._which_out].unsqueeze(-1).repeat(1,1,self._params['num_heads']))    
   
        else:
            scores_out, y_low_out, y_high_out, errors_low_out,\
                  errors_high_out = [], [], [], [], [] # cat along output dim (no. of attn models)
            for hop_id in range(len(self._hopfield)):

                #self._which_out = 0

                X_ctx_true, X_ctx_sim, errors = X_ctx_true_m[:,hop_id,:,:],\
                                                    X_ctx_sim_m[:,hop_id,:,:],\
                                                        errors_m[:,hop_id,:,:]
                y, y_pred = y_m[:,hop_id,:,:], y_pred_m[:,hop_id,:,:]

                if self._use_base_enc:
                    X_ctx_enc_true = self._encoder(X_ctx_true.reshape(-1,X_ctx_true.shape[-1])) # stack batch sequences of context vectors
                    X_ctx_enc_sim = self._encoder(X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))

                    X_ctx_enc_head_true = self.heads[str(hop_id)](X_ctx_enc_true)
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_enc_sim)
                else:
                    X_ctx_enc_head_true = self.heads[str(hop_id)](X_ctx_true.reshape(-1,X_ctx_true.shape[-1]))
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))                        

                X_ctx_enc_head_true = X_ctx_enc_head_true.reshape(batch_size,seq_len,-1)
                X_ctx_enc_head_sim = X_ctx_enc_head_sim.reshape(batch_size,seq_len,-1)

                assoc_mat = self._hopfield[str(hop_id)].get_association_matrix((X_ctx_enc_head_true,\
                                                     X_ctx_enc_head_sim,errors[:,:,self._which_out].unsqueeze(-1)),\
                                                                        association_mask=mask)  
                #if epoch == 5 and batch_id == 0:
                #    print("************ Validation ************")
                #    print(assoc_mat.squeeze(1)[0,4,:])
                #    print(f"predicted: {torch.sum(assoc_mat.squeeze(1)[0,4,:]*errors[0,:,self._which_out])}")  
                #    print(errors[0,4,self._which_out])
        
                # [1,1,300,300] --> [300,300] squeeze the head dim
                # TODO check which dim to squeeze
                scores, y_low, y_high, errors_low, errors_high = [], [], [], [], [] # cat along head dim
                for head in range(assoc_mat.shape[1]):
                    selected_errors, _, _, _ = self._process_assoc_mat(assoc_mat[:,head,:,:],\
                                                                errors[:,:,self._which_out].unsqueeze(-1))   
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
                    y_low.append(_y_low.unsqueeze(0)) # add head dim
                    y_high.append(_y_high.unsqueeze(0))

                    _errors_low = _errors_low.unsqueeze(-1) # [alphas,batch_size*seq_len,1]
                    _errors_high = _errors_high.unsqueeze(-1)

                    _errors_low = _errors_low.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)
                    _errors_high = _errors_high.reshape(no_of_alphas,batch_size,seq_len,1).permute(1,0,2,3)                

                    errors_low.append(_errors_low.unsqueeze(0)) # add head dim
                    errors_high.append(_errors_high.unsqueeze(0))

                y_low = torch.cat(y_low, dim=0)
                y_high = torch.cat(y_high, dim=0) # concatenate along head dim   
                scores = torch.mean(torch.Tensor(scores).to(device))

                errors_low = torch.cat(errors_low, dim=0)
                errors_high = torch.cat(errors_high, dim=0)

                # APPEND EACH OUTPUT RESULTS FOR CAT ALONG OUT DIM
                scores_out.append(scores) # DON'T TAKE MEAN ACROSS OUTPUTS, KEEP EACH OUTPUT SCORE SEPARATE
                y_low_out.append(y_low)
                y_high_out.append(y_high)
                errors_low_out.append(errors_low)
                errors_high_out.append(errors_high)

            scores_out = torch.Tensor(scores_out)
            y_low_out = torch.cat(y_low_out, dim=-1)   # cat along output dim
            y_high_out = torch.cat(y_high_out, dim=-1) 
            errors_low_out = torch.cat(errors_low_out, dim=-1) 
            errors_high_out = torch.cat(errors_high_out, dim=-1)              

        if train:
            return losses
        else:
            y = y_m[:,:,:,self._which_out].permute(0,2,1) # [batch,model_no,seq_len] -> [batch,seq_len,model_no]
            y_pred = y_pred_m[:,:,:,self._which_out].permute(0,2,1)
            errors = errors_m[:,:,:,self._which_out].permute(0,2,1)

            return (scores_out, y, y_pred, y_low_out, y_high_out, errors, errors_low_out, errors_high_out)    
        
    def _process_assoc_mat(self, assoc_mat: torch.Tensor, errors: torch.Tensor):
        """
        :param assoc_mat: [batches,seq_len,mem_len] 
        :param errors: [batches,mem_len,1]
        :return selected_errors: [batches*seq_len,num_samples]
        Samples from softmax distribution over timesteps and 
        return the corresponding errors.
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
            sampled_weights = [assoc_mat.reshape(-1,assoc_mat.shape[-1])[samp][sampled_indices[samp]].unsqueeze(0)\
                    for samp in range(sampled_indices.shape[0])]
            sampled_weights = torch.cat(sampled_weights, dim=0) 
            sampled_weights = sampled_weights / torch.sum(sampled_weights,dim=-1).unsqueeze(-1)
        # num_samples has idx between [0,mem_len)    
        batch_idx = torch.arange(batch_size).unsqueeze(1).repeat((1,num_samples))\
                                                                .repeat_interleave(seq_len, dim=0)
        # batch_idx = sampled_indices = selected_errors =\
        #                    [batch_size*seq_len,num_samples] Note: batch_size is outer dim]
        
        selected_errors = errors.squeeze(-1)[batch_idx, sampled_indices]
        expected_errors = torch.multiply(sampled_weights, selected_errors).sum(dim=1).unsqueeze(-1)
        variance_errors = torch.multiply(sampled_weights, (selected_errors - expected_errors)**2).sum(dim=1)
        variance_errors = variance_errors.unsqueeze(-1)

        return selected_errors, expected_errors, variance_errors, sampled_weights
    
    def nll(self, selected_errors: torch.Tensor, expected_errors: torch.Tensor,\
             variance_errors: torch.Tensor, true_errors: torch.Tensor):
        """Calculate the negative log-likelihood of the sampled errors """
        true_errors = true_errors.reshape(-1,true_errors.shape[-1]) # stack all batches 

        nll = 0.5 * torch.log(2 * torch.pi * variance_errors) + 0.5 * ((true_errors - expected_errors) ** 2 / variance_errors)
   
        return nll.mean()
    
    def mse(self, true_errors: torch.Tensor, expected_errors: torch.Tensor):
        """Calculate the mean squared error of the expected value of sampled errors """
        true_errors = true_errors.reshape(-1,true_errors.shape[-1]) # stack all batches

        return ((true_errors - expected_errors) ** 2).mean()
    
    def _select_quantiles(self, selected_errors: torch.Tensor, alphas: List[float]): 
        # alphas e.g. [0.05, 0.1, 0.15]
        """
        :param selected_errors: [batches*seq_len,num_samples] 
        :param alphas: len(self._alphas)
        :return (q_conformal_low,q_conformal_high): ([len(self._alphas),batches*seq_len],...)         
        """  
        # selected_errors = [batch_size*seq_len,num_samples]
        q_conformal_low = torch.zeros((len(alphas),selected_errors.shape[0])).to(device)
        q_conformal_high = torch.zeros((len(alphas),selected_errors.shape[0])).to(device)

        for alpha in range(len(alphas)):
            #beta = alphas[alpha] / 2
            beta = alphas[alpha]
            q_conformal_low[alpha] = torch.quantile(selected_errors, beta, dim=1)
            q_conformal_high[alpha] = torch.quantile(selected_errors,\
                                                      (1 - alphas[alpha]), dim=1)  #+beta

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
           
        :param y: [batch_size,seq_len,1] 
        :param y_low: [len(self._alphas),batch_size*seq_len]
        :param y_high: [len(self._alphas),batch_size*seq_len]
        :return          
        """
        y = y.reshape(-1,y.shape[-1])  # [1192,1]
        y_low = y_low.unsqueeze(-1) # [9, 1192, 1]
        y_high = y_high.unsqueeze(-1) # [9, 1192, 1]
        batch_size = y.shape[0]    

        score = torch.zeros((len(self._alphas))).to(device)
        pi_width = torch.zeros((len(self._alphas))).to(device)

        for alpha in range(len(self._alphas)): # for all quantiles
            width = (y_high[alpha]-y_low[alpha]).abs()
            undershoot = torch.lt(y, y_low[alpha]).long()
            overshoot = torch.gt(y, y_high[alpha]).long()
            alph = self._alphas[alpha]
            score[alpha] = (width + (undershoot * (y_low[alpha] - y) * 2/alph) +
                             (overshoot * (y - y_high[alpha]) * 2/alph)).sum()
            # taking mean first over batch dim to be consistent with the previous non-batch implementation
            score[alpha] = score[alpha] / batch_size
            pi_width[alpha] = width.sum() / batch_size

        return score, pi_width # Don't mean across alphas, KEEP EACH ALPHA SEPARATED
    
    def calibration_score(self, error_low: torch.Tensor, error_high: torch.Tensor,\
                           true_error: torch.Tensor):
        """
        :param error_low: [len(self._alphas),batch_size*seq_len]
        :param error_high:[len(self._alphas),batch_size*seq_len]
        :param true_error:[batch_size,seq_len,1]
        Check if the expected frequency matches the observed frequency
        """
        true_error = true_error.reshape(-1,) # [batch_size*seq_len,1]
        total_points = true_error.shape[0]
        undershoots = torch.zeros(len(self._alphas)).to(device)
        overshoots = torch.zeros(len(self._alphas)).to(device)

        for alpha in range(len(self._alphas)): # for all quantiles
            
            undershoot = torch.lt(true_error, error_low[alpha]).long()
            overshoot = torch.gt(true_error, error_high[alpha]).long()

            undershoots[alpha] = torch.where(undershoot == 1)[0].shape[0]
            overshoots[alpha] = torch.where(overshoot == 1)[0].shape[0]
        
        bounded_errors = total_points - (undershoots+overshoots)
        bounded_errors = bounded_errors / total_points # len(self._alphas)

        return bounded_errors 

    
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

    def moving_average_with_padding_torch(self, data, window_size=15):
        pad_width = window_size // 2  # Padding amount to keep length the same
        
        padded_data = F.pad(data.unsqueeze(1), (pad_width, pad_width), mode='replicate')
        kernel = torch.ones(1, 1, window_size).to(data.device) / window_size  # Kernel for 1D convolution

        smoothed_data = F.conv1d(padded_data, kernel, groups=1)
        smoothed_data = smoothed_data.squeeze(1)
        
        return smoothed_data   

    
    def evaluate(self, _data: List[torch.Tensor], batch_idx: int, unc_prop: bool=False):
        """
        Does sample-by-sample evaluation of val data with self._mem_data
        """
        # TODO: Check this method for unc_prop = True 
        if not unc_prop:
            X_ctx_true_m, X_ctx_sim_m, errors_m = _data[0].to(device), _data[1].to(device), _data[2].to(device)
            y_m, y_pred_m = _data[3].to(device), _data[4].to(device)
        else:
            X_ctx_sim_m, y_pred_m = _data[0], _data[1]  
        
        batch_size = X_ctx_sim_m.shape[0] # DEBUG THIS -- CHANGE IT TO SHAPE[1]
        seq_len = X_ctx_sim_m.shape[2]
        feat = 1 # only 1 output for now
        no_of_alphas = len(self._alphas)

        total_mem = self._mem_data[0].X_ctx_true_train_enc.shape[0]

        mem_len = 100
        start = 0
        # randomly pick mem_len no of examples from memory
        rand_mem = torch.randint(low=0, high=total_mem, size=(mem_len,))

        if self._params['cp_aggregate'] == 'long_seq':

            # STITCH TOGETHER MANY EPISODES AS ONE BIG EPISODE AND THEN SAMPLE
            for m in range(self.mhn_models):
                if len(self._errors_samp_seq[m]) == 0: 
                    # _mem_data.X_ctx_true_train_enc [4978, 200, 61], 
                    # _mem_data.error_train [4978, 200, 6]
                    self._X_ctx_true_seq[m] = self._mem_data[m].X_ctx_true_train[start:start+mem_len]\
                            .reshape(-1,self.ctx_enc_in).to(device) # [30k,19]
                    self._X_ctx_true_enc_seq[m] = self._mem_data[m].X_ctx_true_train_enc[start:start+mem_len]\
                                                .reshape(-1,self.ctx_enc_out).to(device) # [30k,19]
                    self._errors_samp_seq[m] = self._mem_data[m].error_train[start:start+mem_len]\
                        .reshape(-1,y_pred_m.shape[-1]).to(device) # [30k,6]
            if batch_idx == 0:
                print(f"The memory length is: {len(self._X_ctx_true_enc_seq[0])}")            

            # no mask for evaluation
            #mask = torch.diag(torch.full((seq_len,), fill_value=True, dtype=torch.bool)).to(device)
            # no of batches in self._errors_samp_seq and X_ctx_sim_enc should be same
            scores_out, y_low_out, y_high_out, errors_low_out,\
                  errors_high_out, calib_scores_out, mses_out, nlls_out,\
                        wink_score_out, pi_width_out = [], [], [], [], [], [], [], [], [], [] # cat along output dim (no. of attn models)
            
            for hop_id in range(len(self._hopfield)):

                #self._which_out = 0

                X_ctx_true, X_ctx_sim, errors = X_ctx_true_m[:,hop_id,:,:],\
                                                    X_ctx_sim_m[:,hop_id,:,:],\
                                                        errors_m[:,hop_id,:,self._which_out].unsqueeze(-1)
                y, y_pred = y_m[:,hop_id,:,:], y_pred_m[:,hop_id,:,:]

                if self._use_base_enc:
                    X_ctx_sim_enc = self._encoder(X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_sim_enc)
                else:
                    X_ctx_enc_head_sim = self.heads[str(hop_id)](X_ctx_sim.reshape(-1,X_ctx_sim.shape[-1]))   
                
                assoc_mat = self._hopfield[str(hop_id)].get_association_matrix((self._X_ctx_true_enc_seq[hop_id].unsqueeze(0),\
                                                                    X_ctx_enc_head_sim.unsqueeze(0),\
                                self._errors_samp_seq[hop_id][:,self._which_out].unsqueeze(0).unsqueeze(-1))) 
                #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[0][:,:10])
                #print(assoc_mat.squeeze(0).squeeze(0)[40:60,:].sort(descending=True)[1][:,:10])

                save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['num_mhn_models']}encHeads_{self._params['seq_len']}seqlen_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads_{self._params['out']}out"
                assoc_dict = {}
                assoc_dict['assoc_mat'] = assoc_mat.squeeze(0).squeeze(0).reshape(X_ctx_sim.shape[0],\
                                                                X_ctx_sim.shape[1],-1).detach().cpu().numpy()
                assoc_dict['errors_mem'] = self._errors_samp_seq[hop_id].reshape(mem_len,-1,y_pred.shape[-1]).detach().cpu().numpy()
                if self._params['data_type'] == 'synthetic':
                    assoc_dict['y_pred'] = y_pred.detach().cpu().numpy()
                    assoc_dict['y_true'] = y.detach().cpu().numpy()
                    assoc_dict['x_mem'] = self.input_filter.invert_torch(self._X_ctx_true_seq[m]).detach().cpu().numpy()

                #if batch_idx == 0: 
                    #pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{batch_idx}_TrainValseq.pkl', 'wb'))
                pickle.dump(assoc_dict, open(save_dir + f'/assoc_data_{hop_id}out_{batch_idx}bs.pkl', 'wb'))

                y_low, y_high, errors_low, errors_high, calib_scores,\
                      mses, nlls, wink_scores, pi_widths = [], [], [], [], [], [], [], [], []
                for head in range(assoc_mat.shape[1]):     
                    selected_errors, expected_errors, variance_errors, sampled_weights\
                          = self._process_assoc_mat(assoc_mat[:,head,:,:],\
                                self._errors_samp_seq[hop_id][:,self._which_out].unsqueeze(0).unsqueeze(-1))
                    
                    _errors_low, _errors_high = self._select_quantiles(selected_errors,\
                                                                        alphas=self._alphas)  # [no_of_alphas,seq_len]

                    # all three below unnormalized
                    y_expected_temp = y_pred[:,:,self._which_out] + expected_errors.reshape(batch_size,-1) # [batch_size,seq_len]
                    y_low_temp = y_pred[:,:,self._which_out] + _errors_low.reshape(len(self._alphas),batch_size,-1) # [alphas,batch_size,seq_len]
                    y_high_temp = y_pred[:,:,self._which_out] + _errors_high.reshape(len(self._alphas),batch_size,-1) # [alphas,batch_size,seq_len]
                    selected_y = y_pred[:,:,self._which_out].unsqueeze(-1) + selected_errors.reshape(batch_size,seq_len,-1) # [batch_sz,seq_len,num_samples]
                    variance_y = torch.multiply(sampled_weights, (selected_y.reshape(batch_size*seq_len,-1) - y_expected_temp.reshape(batch_size*seq_len,-1))**2).sum(dim=1)

                    """
                    if hop_id == 0:
                        temp_dict = {'y_expected_temp': y_expected_temp.detach().cpu().numpy(),\
                                      'y_low_temp': y_low_temp.detach().cpu().numpy(),\
                                          'y_high_temp': y_high_temp.detach().cpu().numpy()}
                        pickle.dump(temp_dict, open(save_dir + f'/temp_data_{hop_id}out_{batch_idx}bs.pkl', 'wb'))
                        sys.exit()
                    """    
                    """
                    if not self._params['data_type'] == 'synthetic':
                        nll = self.nll(selected_errors, expected_errors, variance_errors, errors)
                        mse = self.mse(errors, expected_errors)
                
                        nlls.append(nll.unsqueeze(0)) # TODO: RECHECK nll method
                        mses.append(mse.unsqueeze(0)) # TODO: RECHECK mse method
                        calib_score = self.calibration_score(_errors_low, _errors_high, errors)
                        calib_scores.append(calib_score.unsqueeze(0))

                        # Winkler scores and PI-widths on unnormalized errors
                        wink_score, pi_width = self._winkler_score(errors, _errors_low, _errors_high) 
                        wink_scores.append(wink_score.unsqueeze(0))
                        pi_widths.append(pi_width.unsqueeze(0))
                    else:   
                    """
                    #if hop_id == 3:
                    #    print(variance_errors)
                    #    sys.exit()

                    nll = self.nll(selected_y.reshape(batch_size*seq_len,-1), \
                                y_expected_temp.reshape(batch_size*seq_len,1), \
                                    variance_y.unsqueeze(-1), y[:,:,self._which_out].unsqueeze(-1)) 
                    # Moving average won't impact nll
                    # Don't use moving average for spacecraft
                    y_expected_temp = self.moving_average_with_padding_torch(y_expected_temp)
                    y_low_temp = self.moving_average_with_padding_torch(y_low_temp.reshape(len(self._alphas)*batch_size,-1)).reshape(len(self._alphas),batch_size,-1)
                    y_high_temp = self.moving_average_with_padding_torch(y_high_temp.reshape(len(self._alphas)*batch_size,-1)).reshape(len(self._alphas),batch_size,-1)
                    mse = self.mse(y[:,:,self._which_out].unsqueeze(-1), y_expected_temp.reshape(batch_size*seq_len,1))       
                    nlls.append(nll.unsqueeze(0)) # TODO: RECHECK nll method
                    mses.append(mse.unsqueeze(0)) # TODO: RECHECK mse method
                    calib_score = self.calibration_score(y_low_temp.reshape(len(self._alphas),-1),\
                                                        y_high_temp.reshape(len(self._alphas),-1),\
                                                        y[:,:,self._which_out].unsqueeze(-1))
                    calib_scores.append(calib_score.unsqueeze(0))                    

                    # Winkler scores and PI-widths on unnormalized y
                    wink_score, pi_width = self._winkler_score(y[:,:,self._which_out].unsqueeze(-1),\
                                                                y_low_temp.reshape(len(self._alphas),-1), \
                                                            y_high_temp.reshape(len(self._alphas),-1))
                    wink_scores.append(wink_score.unsqueeze(0))
                    pi_widths.append(pi_width.unsqueeze(0))
                    ################################################
                    
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

                calib_scores = torch.cat(calib_scores, dim=0) # [num_heads,len(self._alphas)]
                mses = torch.cat(mses, dim=0) # [num_heads,1]
                nlls = torch.cat(nlls, dim=0) # [num_heads,1]
                wink_scores = torch.cat(wink_scores, dim=0) # [num_heads,len(self._alphas)]
                pi_widths = torch.cat(pi_widths, dim=0) # [num_heads,len(self._alphas)]

                # APPEND EACH OUTPUT RESULTS FOR CAT ALONG OUT DIM
                y_low_out.append(y_low)
                y_high_out.append(y_high)
                errors_low_out.append(errors_low)
                errors_high_out.append(errors_high)
                calib_scores_out.append(calib_scores.unsqueeze(-1))
                mses_out.append(mses.unsqueeze(-1))
                nlls_out.append(nlls.unsqueeze(-1))
                wink_score_out.append(wink_scores.unsqueeze(-1))
                pi_width_out.append(pi_widths.unsqueeze(-1))

            y_low_out = torch.cat(y_low_out, dim=-1)   
            y_high_out = torch.cat(y_high_out, dim=-1) 
            errors_low_out = torch.cat(errors_low_out, dim=-1) 
            errors_high_out = torch.cat(errors_high_out, dim=-1) 
            calib_scores_out = torch.cat(calib_scores_out, dim=-1) # [num_heads,len(self._alphas),num_mhn_models]
            mses_out = torch.cat(mses_out, dim=-1) # [num_heads,1,num_mhn_models]
            nlls_out = torch.cat(nlls_out, dim=-1) # [num_heads,1,num_mhn_models]
            wink_score_out = torch.cat(wink_score_out, dim=-1) # [num_heads,len(self._alphas),num_mhn_models]
            pi_width_out = torch.cat(pi_width_out, dim=-1) # [num_heads,len(self._alphas),num_mhn_models]
             
        y = y_m[:,:,:,self._which_out].permute(0,2,1) # [batch,model_no,seq_len] -> [batch,seq_len,model_no]
        y_pred = y_pred_m[:,:,:,self._which_out].permute(0,2,1)
        errors = errors_m[:,:,:,self._which_out].permute(0,2,1)

        return (y, y_pred, y_low_out, y_high_out, errors, errors_low_out, errors_high_out, calib_scores_out, mses_out, nlls_out, wink_score_out, pi_width_out)
