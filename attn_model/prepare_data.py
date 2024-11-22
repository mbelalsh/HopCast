from data_classes import PICalibData
import torch 
import numpy as np
import pandas as pd
from utils import MeanStdevFilter, prepare_data
from build_ctxt import BuildContext
from typing import List, Tuple
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataProcessor:
    def __init__(self, params):
        self.params = params
        self.input_filter = None
        self.input_dim = None
        self.output_dim = None
        self.calib_horizon = params['calib_horizon']
        self.past_ts_ctxt = self.params['past_ts_ctxt']
        self.past_feat_ctxt = self.params['past_feat_ctxt']
        self.past_pred_ctxt = self.params['past_pred_ctxt']
        self.init_cond_ctxt = self.params['init_cond_ctxt']
        self._build_ctxt = BuildContext(self.params)
        self.params['ctxt_dim'] = 0
        if self.past_ts_ctxt > 0: 
            self.params['ctxt_dim'] = self.past_ts_ctxt # input_dim will be added in get_data method   
        if self.past_pred_ctxt > 0:
            # exclude the current timestep of the prediction context
           self.params['ctxt_dim'] += (self.past_pred_ctxt - 1)  
        self.pos_enc = params['pos_enc']       
        self._add_mirrorless_enc = self.params['mirrorless_enc']

    def get_data(self) -> np.ndarray:
        df = pd.read_csv(self.params['data_path'])

        #df['group'] = df.index // self.calib_horizon # repeats every 0 to 1440
        if self.params['data_type']:
            grouped = df.groupby('horizon') 
        else:
            grouped = df.groupby('pred_dt') 

        if self.params['ode_name'] == 'lorenz' or self.params['ode_name'] == 'force_model':
            pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
            gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
        elif self.params['ode_name'] == 'glycolytic':    
            pred_cols = ['S1_pred','S2_pred','S3_pred','S4_pred','S5_pred','S6_pred','S7_pred']
            gr_cols = ['S1_gr','S2_gr','S3_gr','S4_gr','S5_gr','S6_gr','S7_gr'] 
        elif self.params['ode_name'] == 'LVolt':    
            pred_cols = ['x_pred','y_pred','x_dot_pred','y_dot_pred']
            gr_cols = ['x_gr','y_gr','x_dot_gr','y_dot_gr']    
        elif self.params['ode_name'] == 'lorenz96':
            pred_cols = ['x1_pred','x2_pred','x3_pred','x4_pred','x5_pred']
            gr_cols = ['x1_gr','x2_gr','x3_gr','x4_gr','x5_gr']    
        elif self.params['ode_name'] == 'FHNag':
            pred_cols = ['v_pred','w_pred']
            gr_cols = ['v_gr','w_gr']                                              
        group_arrs = []

        for _, group in grouped:
            group_arr = group[pred_cols+gr_cols].values
            group_arrs.append(np.expand_dims(group_arr, axis=0))

        data = np.concatenate(group_arrs, axis=0)
        if self.params['data_type']:
            data = data.transpose(1,0,2)
        print(f"The data shape is: {data.shape}")    
        
        self.input_dim = len(pred_cols) # Using this to store norm data in PICalibData
        self.output_dim = len(pred_cols)

        self.params['ctxt_dim'] += self.past_feat_ctxt*self.input_dim # input_dim+ctxt_dim
        if self.init_cond_ctxt:
            self.params['ctxt_dim'] += self.input_dim
        if self._add_mirrorless_enc:
            self.params['ctxt_dim'] += 4
        self.ctxt_dim = self.params['ctxt_dim']
        self.params['input_dim'] = self.input_dim
        self.params['output_dim'] = self.output_dim
        self.input_filter = MeanStdevFilter(self.ctxt_dim) 

        return data
    
    def data_tuples(self, data: np.ndarray) -> Tuple[PICalibData]:

        sim_data = data[:,:,:self.input_dim]
        gr_tr_data = data[:,:,self.input_dim:]

        # dropping one point from seq due to shifting (calib_horizon-1) is seq_len now
        # TODO: Check this step below for ode_data (VALID or not)
        gr_input_data = gr_tr_data[:,:-1,:] 
        gr_output_data = gr_tr_data[:,1:,:]
        self.params['norm_loss_wghts'] = \
            torch.FloatTensor(1 / np.var(np.array(gr_output_data.reshape(-1,self.output_dim)), axis=0)).to(device)
        #[4.0810e-08, 1.3870e-06, 3.9746e-08, 3.6635e-02, 1.2449e+00, 3.5690e-02]

        sim_input_data = sim_data[:,:-1,:]
        sim_output_data = sim_data[:,1:,:]

        # get normalized ground truth and simulated input data
        #self.calculate_mean_var(gr_input_data)
        #input_data = self.normalize_data(gr_input_data, sim_input_data)

        #gr_input_filter, sim_input_filter = input_data
        if self.params['data_type']:
            train_len = int(0.80*gr_input_data.shape[0])
        else:
            train_len = int(0.95*gr_input_data.shape[0])    
        val_len = gr_input_data.shape[0] - train_len - 1 # TODO: remove -1 later
        #print(train_len)
        #print(gr_output_data[-val_len,:,0])
        #print(val_len)
        #print((gr_output_data[-val_len:]-sim_output_data[-val_len:])[0,:20,0])
        #sys.exit()        
        print(f"Training seq. are: {train_len}")
        print(f"Validation seq. are: {val_len}")

        # PICalibData contains the un-normalized data here for all cases 
        calib_train_true = PICalibData(X=torch.Tensor(gr_input_data[:train_len]),
                                       Y=torch.Tensor(gr_output_data[:train_len]))
        
        calib_val_true = PICalibData(X=torch.Tensor(gr_input_data[-val_len:]),
                                    Y=torch.Tensor(gr_output_data[-val_len:]))
        
        calib_train_sim = PICalibData(X=torch.Tensor(sim_input_data[:train_len]),
                                       Y=torch.Tensor(sim_output_data[:train_len]),
                                       timesteps=torch.arange(1,calib_train_true.X.shape[1]+1),
                                       error=torch.Tensor(gr_output_data[:train_len]-sim_output_data[:train_len]))
        
        calib_val_sim = PICalibData(X=torch.Tensor(sim_input_data[-val_len:]),
                                    Y=torch.Tensor(sim_output_data[-val_len:]),
                                    timesteps=torch.arange(1,calib_val_true.X.shape[1]+1),
                                    error=torch.Tensor(gr_output_data[-val_len:]-sim_output_data[-val_len:]))
        
        return (calib_train_true, calib_val_true, calib_train_sim, calib_val_sim)
    
    def stack_seq(self, calib_data: Tuple[PICalibData]):
        """Stack n-sequences together for Improved (?) training"""

        # TODO: make this more general for n-sequences
        train_len = calib_data[0].X_ctx.shape[0]
        seq_len = calib_data[0].X_ctx.shape[1] # same for train and val 
        val_len = calib_data[1].X_ctx.shape[0]

        calib_data[0].X_ctx = calib_data[0].X_ctx.reshape(train_len//2, seq_len*2, -1)
        calib_data[0].Y = calib_data[0].Y.reshape(train_len//2, seq_len*2, -1)
        calib_data[0].X = calib_data[0].X.reshape(train_len//2, seq_len*2, -1)

        calib_data[1].X_ctx = calib_data[1].X_ctx.reshape(val_len//2, seq_len*2, -1)
        calib_data[1].Y = calib_data[1].Y.reshape(val_len//2, seq_len*2, -1)
        calib_data[1].X = calib_data[1].X.reshape(val_len//2, seq_len*2, -1)

        calib_data[2].X_ctx = calib_data[2].X_ctx.reshape(train_len//2, seq_len*2, -1)
        calib_data[2].Y = calib_data[2].Y.reshape(train_len//2, seq_len*2, -1)
        calib_data[2].error = calib_data[2].error.reshape(train_len//2, seq_len*2, -1) 
        calib_data[2].X = calib_data[2].X.reshape(train_len//2, seq_len*2, -1) 

        calib_data[3].X_ctx = calib_data[3].X_ctx.reshape(val_len//2, seq_len*2, -1)
        calib_data[3].Y = calib_data[3].Y.reshape(val_len//2, seq_len*2, -1)
        calib_data[3].error = calib_data[3].error.reshape(val_len//2, seq_len*2, -1)
        calib_data[3].X = calib_data[3].X.reshape(val_len//2, seq_len*2, -1)

        return calib_data
    
    def normalize_calib_data(self, calib_data: Tuple[PICalibData]):

        train_len = calib_data[0].X_ctx.shape[0]
        val_len = calib_data[1].X_ctx.shape[0]

        gr_input_ctxt = torch.cat([calib_data[0].X_ctx, calib_data[1].X_ctx], dim=0)
        gr_input_ctxt = np.array(gr_input_ctxt)

        sim_input_ctxt = torch.cat([calib_data[2].X_ctx, calib_data[3].X_ctx], dim=0)
        sim_input_ctxt = np.array(sim_input_ctxt)
        
        self.calculate_mean_var(gr_input_ctxt, self.ctxt_dim)
        gr_ctxt_filter, sim_ctxt_filter = self.normalize_data(gr_input_ctxt, sim_input_ctxt, self.ctxt_dim)

        gr_ctxt_filter = torch.Tensor(gr_ctxt_filter)
        sim_ctxt_filter = torch.Tensor(sim_ctxt_filter)

        # saving X_ctxt as normalized 
        calib_data[0].X_ctx, calib_data[1].X_ctx = gr_ctxt_filter[:train_len], gr_ctxt_filter[-val_len:]
        calib_data[2].X_ctx, calib_data[3].X_ctx = sim_ctxt_filter[:train_len], sim_ctxt_filter[-val_len:]

        # saving X as normalized 
        calib_data[0].X, calib_data[1].X = gr_ctxt_filter[:train_len,:,:self.input_dim], gr_ctxt_filter[-val_len:,:,:self.input_dim]
        calib_data[2].X, calib_data[3].X = sim_ctxt_filter[:train_len,:,:self.input_dim], sim_ctxt_filter[-val_len:,:,:self.input_dim]     

        return calib_data  
    
    def calculate_mean_var(self, gr_input_data: np.ndarray, input_dim: int) -> None:

        gr_input_data = gr_input_data.reshape(-1, input_dim)

        total_points = gr_input_data.shape[0]

        for i in range(total_points):
            self.input_filter.update(gr_input_data[i,:])

        self.params['input_filter'] = self.input_filter  

        return 
    
    def normalize_data(self, gr_input_data: np.ndarray, \
                       sim_input_data: np.ndarray, input_dim: int) -> Tuple[np.ndarray]:
        
        gr_input_data = gr_input_data.reshape(-1, input_dim)
        sim_input_data = sim_input_data.reshape(-1, input_dim)

        gr_input_filter = prepare_data(gr_input_data, self.input_filter)
        sim_input_filter = prepare_data(sim_input_data, self.input_filter)

        gr_input_filter = gr_input_filter.reshape(-1, self.calib_horizon-1, input_dim)
        sim_input_filter = sim_input_filter.reshape(-1, self.calib_horizon-1, input_dim)

        return (gr_input_filter, sim_input_filter)
    
    def no_ctxt(self, calib_data: Tuple[PICalibData]):
        """
        In case of no context, simply copy the calib_data[0].X to
        calib_data[0].X_ctxt 
        """
        calib_data[0].X_ctx = calib_data[0].X.clone()
        calib_data[1].X_ctx = calib_data[1].X.clone()
        calib_data[2].X_ctx = calib_data[2].X.clone()
        calib_data[3].X_ctx = calib_data[3].X.clone()

        return calib_data
    
    def build_context(self, calib_data: Tuple[PICalibData], just_ts_ctxt: bool):

        if not just_ts_ctxt:
            # CAUTION: calib_data gets modified in-place so be careful sending it multiple times to self._build_ctxt
            # CAUTION: DON'T MODIFY X ENTRY OF PICalibData TUPLE AS IT IS USEFUL FOR ADDING INITIAL CONDITION CONTEXT
            if self.past_feat_ctxt > 1:
                assert self.past_pred_ctxt == 0, "Current implementation does not support both contexts simultaneouly"
                calib_data = self._build_ctxt.add_k_feat(calib_data)
            else:
                print("Past feat vectors will NOT be used as context!")  

            if self.past_pred_ctxt > 0:    
                assert self.past_feat_ctxt == 1, "Current implementation does not support both contexts simultaneouly"    
                calib_data = self._build_ctxt.add_k_pred(calib_data, which_out=0)

            if calib_data[0].X_ctx is None: # True if self.past_feat_ctxt == 1
                calib_data = self.no_ctxt(calib_data)
                print("X has been copied to X_ctxt!")

            if self.past_ts_ctxt == 1:
                calib_data = self._build_ctxt.add_timestep(calib_data)
            elif self.past_ts_ctxt > 1: 
                calib_data = self._build_ctxt.add_k_timestep(calib_data)
            else: 
                print("The past timesteps will NOT be used as context!")   

            if self.init_cond_ctxt:
                calib_data = self._build_ctxt.add_init_cond(calib_data)
            else:
                print("Initial condition won't be used as a context!")  

            if self._add_mirrorless_enc:
                calib_data = self._build_ctxt.add_mirrorless_encoding(calib_data)

            if self.pos_enc:
                calib_data = self._build_ctxt.positional_encoding(calib_data)
            else:
                print("Positional encodings are NOT used!")    
        else:
            calib_data = self._build_ctxt.just_ts_ctxt(calib_data) 
            print("Only time context will be used!")       
            if self.init_cond_ctxt:
                calib_data = self._build_ctxt.add_init_cond(calib_data)

        print(f"The context length is: {calib_data[0].X_ctx.shape[-1]}")    

        return calib_data    
