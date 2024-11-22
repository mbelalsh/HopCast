from typing import Tuple, List, Dict
from data_classes import PICalibData
from utils import get_positional_encoding
import torch
import time
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BuildContext:
    """
    This class contains methods to prepare different types of contexts
    for MHN.
    """
    def __init__(self, params: Dict) -> None:
        self.params = params
        self.past_ts_ctxt = self.params['past_ts_ctxt']
        self.past_feat_ctxt = self.params['past_feat_ctxt']
        self.past_pred_ctxt = self.params['past_pred_ctxt']

    def add_init_cond(self, calib_data: Tuple[PICalibData]):
        """
        Add initial condition to the context vector
        """
        seq_len = calib_data[0].X_ctx.shape[1]
        # CAUTION: The X entry in tuple should always contain the original system state
        assert calib_data[0].X.shape[2] == self.params['state_dim']

        calib_data[0].X_ctx = torch.cat((calib_data[0].X[:,0,:].unsqueeze(1).repeat(1,seq_len,1),\
                                          calib_data[0].X_ctx), dim=2)
        calib_data[1].X_ctx = torch.cat((calib_data[1].X[:,0,:].unsqueeze(1).repeat(1,seq_len,1), \
                                         calib_data[1].X_ctx), dim=2)
        calib_data[2].X_ctx = torch.cat((calib_data[2].X[:,0,:].unsqueeze(1).repeat(1,seq_len,1), \
                                         calib_data[2].X_ctx), dim=2)
        calib_data[3].X_ctx = torch.cat((calib_data[3].X[:,0,:].unsqueeze(1).repeat(1,seq_len,1), \
                                         calib_data[3].X_ctx), dim=2)

        return calib_data

    def add_timestep(self, calib_data: Tuple[PICalibData]): 
        """
        Prepare the context features for MHN. This method adds time-step [1,2,3,....,]
        to the end of the original state and action input. 
        """       
        train_ts = calib_data[2].timesteps.repeat(calib_data[2].X.shape[0],1).unsqueeze(2)
        val_ts = calib_data[3].timesteps.repeat(calib_data[3].X.shape[0],1).unsqueeze(2)

        calib_data[0].X_ctx = torch.cat((calib_data[0].X_ctx,train_ts), dim=2)
        calib_data[1].X_ctx = torch.cat((calib_data[1].X_ctx,val_ts), dim=2)
        calib_data[2].X_ctx = torch.cat((calib_data[2].X_ctx,train_ts), dim=2)
        calib_data[3].X_ctx = torch.cat((calib_data[3].X_ctx,val_ts), dim=2)

        return calib_data
    
    def add_k_timestep(self, calib_data: Tuple[PICalibData]):
        
        train_traj = calib_data[0].X.shape[0]
        val_traj = calib_data[1].X.shape[0]

        ts = calib_data[2].timesteps

        padding = torch.zeros((self.past_ts_ctxt-1))
        padded_t = torch.cat((padding, ts))
        win_padded_t = padded_t.unfold(0, self.past_ts_ctxt, step=1)

        train_ctxt = win_padded_t.repeat(train_traj,1,1) # [no_of_episodes,seq_len,self.past_ts_ctxt]
        val_ctxt = win_padded_t.repeat(val_traj,1,1)

        calib_data[0].X_ctx = torch.cat((calib_data[0].X_ctx,train_ctxt), dim=2)
        calib_data[1].X_ctx = torch.cat((calib_data[1].X_ctx,val_ctxt), dim=2)
        calib_data[2].X_ctx = torch.cat((calib_data[2].X_ctx,train_ctxt), dim=2)
        calib_data[3].X_ctx = torch.cat((calib_data[3].X_ctx,val_ctxt), dim=2)

        return calib_data

    def add_k_feat(self, calib_data: Tuple[PICalibData]):
        """Adding previous k inputs to the input"""
        train_len = calib_data[0].X.shape[0]
        val_len = calib_data[1].X.shape[0]
        seq_len = calib_data[0].X.shape[1]
        input_dim = calib_data[0].X.shape[-1]

        train_true_X = calib_data[0].X
        val_true_X = calib_data[1].X
        train_sim_X = calib_data[2].X
        val_sim_X = calib_data[3].X

        # includes current feat vector as in 5 means past 4 feat vectors plus current
        past_k_feat = self.past_feat_ctxt
        padding_train = torch.zeros((train_len,past_k_feat-1,input_dim))
        padding_val = torch.zeros((val_len,past_k_feat-1,input_dim))

        train_true_X = torch.cat([padding_train,train_true_X], dim=1)
        val_true_X = torch.cat([padding_val,val_true_X], dim=1)
        train_sim_X = torch.cat([padding_train,train_sim_X], dim=1)
        val_sim_X = torch.cat([padding_val,val_sim_X], dim=1)

        train_true_X = train_true_X.unfold(1,past_k_feat,1).permute(0,1,3,2).reshape(train_len,seq_len,-1)
        val_true_X = val_true_X.unfold(1,past_k_feat,1).permute(0,1,3,2).reshape(val_len,seq_len,-1)
        train_sim_X = train_sim_X.unfold(1,past_k_feat,1).permute(0,1,3,2).reshape(train_len,seq_len,-1)
        val_sim_X = val_sim_X.unfold(1,past_k_feat,1).permute(0,1,3,2).reshape(val_len,seq_len,-1)

        calib_data[0].X_ctx = train_true_X
        calib_data[1].X_ctx = val_true_X
        calib_data[2].X_ctx = train_sim_X
        calib_data[3].X_ctx = val_sim_X

        return calib_data
    
    def add_k_pred(self, calib_data: Tuple[PICalibData], which_out: int):
        """Add past k predictions as context to the input"""

        train_len = calib_data[0].X.shape[0]
        val_len = calib_data[1].X.shape[0]

        train_true_X = calib_data[0].X[:,:,which_out].unsqueeze(-1)
        val_true_X = calib_data[1].X[:,:,which_out].unsqueeze(-1)
        train_sim_X = calib_data[2].X[:,:,which_out].unsqueeze(-1)
        val_sim_X = calib_data[3].X[:,:,which_out].unsqueeze(-1)

        # includes the current prediction as in 5 means past 4 pred plus current one
        past_k_pred = self.past_pred_ctxt
        padding_train = torch.zeros((train_len,past_k_pred-1,1))
        padding_val = torch.zeros((val_len,past_k_pred-1,1))

        train_true_X = torch.cat([padding_train,train_true_X], dim=1)
        val_true_X = torch.cat([padding_val,val_true_X], dim=1)
        train_sim_X = torch.cat([padding_train,train_sim_X], dim=1)
        val_sim_X = torch.cat([padding_val,val_sim_X], dim=1)

        train_true_X = train_true_X.unfold(1,past_k_pred,1).permute(0,1,3,2).squeeze(-1)[:,:,:-1] # skip the current pred (keep past only)
        val_true_X = val_true_X.unfold(1,past_k_pred,1).permute(0,1,3,2).squeeze(-1)[:,:,:-1]
        train_sim_X = train_sim_X.unfold(1,past_k_pred,1).permute(0,1,3,2).squeeze(-1)[:,:,:-1]
        val_sim_X = val_sim_X.unfold(1,past_k_pred,1).permute(0,1,3,2).squeeze(-1)[:,:,:-1]

        train_true_X = torch.cat((train_true_X,calib_data[0].X), dim=-1)
        val_true_X = torch.cat((val_true_X,calib_data[1].X), dim=-1)
        train_sim_X = torch.cat((train_sim_X,calib_data[2].X), dim=-1)
        val_sim_X = torch.cat((val_sim_X,calib_data[3].X), dim=-1)

        calib_data[0].X_ctx = train_true_X
        calib_data[1].X_ctx = val_true_X
        calib_data[2].X_ctx = train_sim_X
        calib_data[3].X_ctx = val_sim_X

        return calib_data
    
    def just_ts_ctxt(self, calib_data: Tuple[PICalibData]):
        """Include just past k timesteps as context"""

        train_traj = calib_data[0].X.shape[0]
        val_traj = calib_data[1].X.shape[0]

        ts = calib_data[2].timesteps

        padding = torch.zeros((self.past_ts_ctxt-1))
        padded_t = torch.cat((padding, ts))
        win_padded_t = padded_t.unfold(0, self.past_ts_ctxt, step=1)

        train_ctxt = win_padded_t.repeat(train_traj,1,1) # [no_of_episodes,seq_len,self.past_ts_ctxt]
        val_ctxt = win_padded_t.repeat(val_traj,1,1)

        calib_data[0].X_ctx = train_ctxt.clone()
        calib_data[1].X_ctx = val_ctxt.clone()
        calib_data[2].X_ctx = train_ctxt.clone()
        calib_data[3].X_ctx = val_ctxt.clone()

        return calib_data
    
    def positional_encoding(self, calib_data: Tuple[PICalibData]):
        """Add sinusoidal positional encoding to the input"""

        seq_len = calib_data[2].X_ctx.shape[1]
        d_model = calib_data[2].X_ctx.shape[2]

        pos_enc = torch.from_numpy(get_positional_encoding(seq_len, d_model))

        calib_data[0].X_ctx = calib_data[0].X_ctx + pos_enc.unsqueeze(0)
        calib_data[1].X_ctx = calib_data[1].X_ctx + pos_enc.unsqueeze(0)
        calib_data[2].X_ctx = calib_data[2].X_ctx + pos_enc.unsqueeze(0)
        calib_data[3].X_ctx = calib_data[3].X_ctx + pos_enc.unsqueeze(0)

        return calib_data
