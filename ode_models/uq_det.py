import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from train import Ensemble
from scores import Scores
from scipy import stats
import pandas as pd
import pickle
import sys, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UncPropagate():

    def __init__(self, params: Dict, ensemble: Ensemble):
        self.params = params
        self.ens = ensemble
        self._alphas = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
        self.score = Scores()
        self.rand_models = self.params['rand_models']

    def z_scores(self):
        """Calculates the z-scores for given set of quantiles in a standard normal"""
        lower_alphas = np.array(self._alphas)
        upper_alphas = 1 - lower_alphas

        lower_z_score = torch.Tensor(stats.norm.ppf(lower_alphas)).to(device).unsqueeze(-1).unsqueeze(-1)
        upper_z_score = torch.Tensor(stats.norm.ppf(upper_alphas)).to(device).unsqueeze(-1).unsqueeze(-1)

        return lower_z_score, upper_z_score

    def propagate(self, run: int):  
        """Select how many steps you want predict ahead"""
        self.run = run
        rand_models = torch.randperm(self.ens.num_models).detach().cpu().numpy().tolist()
        self.temp_dict = {}
        
        for i in range(len(rand_models)):
            self.temp_dict[i] = self.ens.models[rand_models[i]]
           
        if self.params['one_step']:
            return self.one_step()
        else:
            return self.multi_step()

    def multi_step(self): 
        """Do multi-step predictions autoregressively with the trained model"""     

        self.no_of_episodes = len(self.ens.seq_input_val)

        all_pred_unnorm_y = torch.zeros((self.no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((self.no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

        for idx, example in enumerate(self.ens.seq_val_loader):

            if example[0].shape[0] < self.ens.traj_batch:
                start_idx = end_idx
                end_idx = start_idx + example[0].shape[0]
            else:
                start_idx = idx*example[0].shape[0] 
                end_idx = (idx+1)*example[0].shape[0]

            timesteps = example[0].shape[1]
            unnorm_true_y = example[1]
       
            all_true_unnorm_y[start_idx:end_idx] = unnorm_true_y.to(device)
            for step in range(timesteps):
  
                if step == 0:
                    unnorm_x = example[0][:,step,:].to(device)
                    norm_x = self.ens.input_filter.filter_torch(unnorm_x)
                    norm_x_m = norm_x.repeat(self.rand_models,1,1) 
                    unnorm_x_m = unnorm_x.repeat(self.rand_models,1,1)   
                models_pred_norm = []   
                models_pred_unnorm = [] 

                for model in range(self.params['rand_models']):

                    #self.ens.models[model].eval()
                    self.temp_dict[model].eval()
                    with torch.no_grad():
                        #delta = self.ens.models[model](norm_x_m[model])
                        delta = self.temp_dict[model](norm_x_m[model])
                        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':
                            delta = self.ens.output_filter.invert_torch(delta)
                        pred_unnorm_y = unnorm_x_m[model] + delta
                        all_pred_unnorm_y[start_idx:end_idx,model,step,:] = pred_unnorm_y
                        pred_norm_y = self.ens.input_filter.filter_torch(pred_unnorm_y)
                        models_pred_norm.append(pred_norm_y.unsqueeze(0))
                        models_pred_unnorm.append(pred_unnorm_y.unsqueeze(0))          

                norm_x_m = torch.concatenate(models_pred_norm, dim=0)
                unnorm_x_m = torch.concatenate(models_pred_unnorm, dim=0)
        print(f"Models used for inference are: {model+1}")        

        self.save_scores(all_pred_unnorm_y, all_true_unnorm_y)        

        return  
    
    def save_scores(self, all_pred_unnorm_y: torch.Tensor, all_true_unnorm_y: torch.Tensor):
        """Calculate calibration and other scores and save to CSV file."""

        mean_pred_unnorm_y = all_pred_unnorm_y.mean(dim=1) # [batch_sz,sq_len,outs]
        var_pred_unnorm_y = all_pred_unnorm_y.var(dim=1) # [batch_sz,sq_len,outs]
        lower_z_scores, upper_z_scores = self.z_scores()
        # CALCULATE QUANTILES
        unnorm_mu, unnorm_var, unnorm_upper_mu, unnorm_lower_mu = \
                            self.select_quantiles(mean_pred_unnorm_y, var_pred_unnorm_y,\
                                                                lower_z_scores, upper_z_scores) 

        scores = self.score.get_scores(mean_pred_unnorm_y, var_pred_unnorm_y,\
                                        unnorm_lower_mu, unnorm_upper_mu, all_true_unnorm_y)

        # save all scores to dict    
        scores_dict = {'calib_scores': scores[0],
                       'wink_scores': scores[1],
                       'pi_widths': scores[2],
                       'mses': scores[3],
                       'nlls': scores[4]
                       }          

        #pickle.dump(scores_dict, open(self.ens.model_dir + f"/scores_calib_{self.params['one_step']}onestep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        mu_quant_pred_unnorm_y = np.concatenate([unnorm_mu[np.newaxis,:,:,:],\
                                                 unnorm_upper_mu,unnorm_lower_mu], axis=0)
        data_dict = {'all_pred_unnorm_y': all_pred_unnorm_y.detach().cpu().numpy(),
                     'all_true_unnorm_y': all_true_unnorm_y.detach().cpu().numpy(),
                     'mu_quant_pred_unnorm_y': mu_quant_pred_unnorm_y}       
         
        #pickle.dump(data_dict, open(self.ens.model_dir + f"/val_propagate_{self.params['one_step']}onestep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        errors = all_pred_unnorm_y - all_true_unnorm_y.repeat(self.rand_models,1,1,1).permute(1,0,2,3)
        errors_dict = {"errors": errors.detach().cpu().numpy()}
        #pickle.dump(errors_dict, open(self.ens.model_dir + f"/errors_{self.params['one_step']}onestep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        horizon = torch.Tensor(np.linspace(0, self.ens.timesteps-1, self.ens.timesteps)).repeat(self.no_of_episodes,1).numpy()
        
        model_error = 0
        pd_pred_unnorm_y = all_pred_unnorm_y[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_true_unnorm_y = all_true_unnorm_y.reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_errors = errors[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()

        if self.params['ode_name'] == 'lorenz':
            pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
            gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
            resid_cols = ['resid_x','resid_y','resid_z','resid_x_dot','resid_y_dot','resid_z_dot']
        elif self.params['ode_name'] == 'glycolytic':
            pred_cols = ['S1_pred','S2_pred','S3_pred','S4_pred','S5_pred','S6_pred','S7_pred']
            gr_cols = ['S1_gr','S2_gr','S3_gr','S4_gr','S5_gr','S6_gr','S7_gr']
            resid_cols = ['resid_S1','resid_S2','resid_S3','resid_S4','resid_S5','resid_S6','resid_S7']    
        elif self.params['ode_name'] == 'LVolt':
            pred_cols = ['x_pred','y_pred','x_dot_pred','y_dot_pred']
            gr_cols = ['x_gr','y_gr','x_dot_gr','y_dot_gr']   
            resid_cols = ['resid_x','resid_y','resid_x_dot','resid_y_dot']      
        elif self.params['ode_name'] == 'lorenz96':
            pred_cols = ['x1_pred','x2_pred','x3_pred','x4_pred','x5_pred']
            gr_cols = ['x1_gr','x2_gr','x3_gr','x4_gr','x5_gr']   
            resid_cols = ['resid_x1','resid_x2','resid_x3','resid_x4','resid_x5'] 
        elif self.params['ode_name'] == 'FHNag':
            pred_cols = ['v_pred','w_pred']
            gr_cols = ['v_gr','w_gr']   
            resid_cols = ['resid_v','resid_w']                                            

        df = pd.DataFrame({
                    'horizon': horizon.reshape(-1,),
                    pred_cols[0]: pd_pred_unnorm_y[:,0],
                    pred_cols[1]: pd_pred_unnorm_y[:,1],
                    #pred_cols[2]: pd_pred_unnorm_y[:,2],
                    #pred_cols[3]: pd_pred_unnorm_y[:,3],
                    #pred_cols[4]: pd_pred_unnorm_y[:,4],
                    #pred_cols[5]: pd_pred_unnorm_y[:,5],
                    #pred_cols[6]: pd_pred_unnorm_y[:,6],
                    gr_cols[0]: pd_true_unnorm_y[:,0],
                    gr_cols[1]: pd_true_unnorm_y[:,1],
                    #gr_cols[2]: pd_true_unnorm_y[:,2],
                    #gr_cols[3]: pd_true_unnorm_y[:,3],
                    #gr_cols[4]: pd_true_unnorm_y[:,4],
                    #gr_cols[5]: pd_true_unnorm_y[:,5],
                    #gr_cols[6]: pd_true_unnorm_y[:,6],
                    resid_cols[0]: pd_errors[:,0],
                    resid_cols[1]: pd_errors[:,1],
                    #resid_cols[2]: pd_errors[:,2],
                    #resid_cols[3]: pd_errors[:,3],
                    #resid_cols[4]: pd_errors[:,4],
                    #resid_cols[5]: pd_errors[:,5],
                    #resid_cols[6]: pd_errors[:,6],
                })

        if not self.params['one_step']:
            # Save DataFrame to CSV file
            if os.path.exists(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv"):
                print("Error file already exists!")
            else:    
                print('Error file Saved')
                df.to_csv(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv", index=False)

        return          

    def select_quantiles(self, unnorm_mu:torch.Tensor, unnorm_var:torch.Tensor,\
                     lower_z_scores: torch.Tensor, upper_z_scores: torch.Tensor):
        """
        unnorm_mu: [batch_size*seq_len,dp_outs] normalized predictions in tensor
        unnorm_var: [batch_size*seq_len,dp_outs] predicted logvar in tensor
        lower_z_scores: [self._alphas,1,1]
        upper_z_scores: [self._alphas,1,1]
        """
        batch_sz = unnorm_mu.shape[0]
        seq_len = unnorm_mu.shape[1]
        outs = unnorm_mu.shape[-1]

        unnorm_mu = unnorm_mu.reshape(batch_sz*seq_len,outs)
        unnorm_var = unnorm_var.reshape(batch_sz*seq_len,outs)       

        unnorm_upper_mu =  unnorm_mu.unsqueeze(0) + torch.mul(unnorm_var.sqrt().unsqueeze(0), upper_z_scores) # [alphas,examples,dp_outs]
        unnorm_lower_mu =  unnorm_mu.unsqueeze(0) - torch.mul(unnorm_var.sqrt().unsqueeze(0), upper_z_scores) # upper is fine since we are subtracting 
    
        unnorm_mu = unnorm_mu.reshape(batch_sz,seq_len,outs).detach().cpu().numpy() # [examples,dp_outs]
        unnorm_var = unnorm_var.reshape(batch_sz,seq_len,outs).detach().cpu().numpy()
        unnorm_upper_mu = unnorm_upper_mu.reshape(-1,batch_sz,seq_len,outs).detach().cpu().numpy() # [alphas,examples,dp_outs]
        unnorm_lower_mu = unnorm_lower_mu.reshape(-1,batch_sz,seq_len,outs).detach().cpu().numpy() # [alphas,examples,dp_outs]

        return unnorm_mu, unnorm_var, unnorm_upper_mu, unnorm_lower_mu 
    
    def one_step(self):
        """Calculate the one-step calibration of deterministic and probabilistic ensembles"""
            
        self.no_of_episodes = len(self.ens.seq_input_val)

        all_pred_unnorm_y = torch.zeros((self.no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((self.no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

        for idx, example in enumerate(self.ens.seq_val_loader):
            if example[0].shape[0] < self.ens.traj_batch:
                start_idx = end_idx
                end_idx = start_idx + example[0].shape[0]
            else:
                start_idx = idx*example[0].shape[0] 
                end_idx = (idx+1)*example[0].shape[0]

            timesteps = example[0].shape[1]
            unnorm_true_y = example[1]  

            all_true_unnorm_y[start_idx:end_idx] = unnorm_true_y.to(device)
            for step in range(timesteps):
  
                unnorm_x = example[0][:,step,:].to(device)
                norm_x = self.ens.input_filter.filter_torch(unnorm_x)
                norm_x_m = norm_x.repeat(self.rand_models,1,1) 
                unnorm_x_m = unnorm_x.repeat(self.rand_models,1,1)   

                for model in range(self.params['rand_models']):

                    #self.ens.models[model].eval()
                    self.temp_dict[model].eval()
                    with torch.no_grad():
                        #delta = self.ens.models[model](norm_x_m[model])
                        delta = self.temp_dict[model](norm_x_m[model])
                        pred_unnorm_y = unnorm_x_m[model] + delta
                        all_pred_unnorm_y[start_idx:end_idx,model,step,:] = pred_unnorm_y

        print(f"Models used for inference are: {model+1}")          

        self.save_scores(all_pred_unnorm_y, all_true_unnorm_y)  

        return                



            


        
