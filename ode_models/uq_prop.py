from typing import Tuple, List, Dict

import time
import sys
import time
import numpy as np
import torch
import pandas as pd
from torch.distributions import Normal
import pickle
import scipy.stats as stats
from train import Ensemble
from scores import Scores
import scipy.stats as stats
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UncPropDeepEns():

    def __init__(self, params: Dict, ensemble: Ensemble):
        self.params = params
        self.ens = ensemble
        self.n_particles = params['n_particles']
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
        """Select appropriate method for uncertainty propagation"""
        self.run = run
        rand_models = torch.randperm(self.ens.num_models).detach().cpu().numpy().tolist()
        self.temp_dict = {}        
        
        for i in range(len(rand_models)):
            self.temp_dict[i] = self.ens.models[rand_models[i]]        

        if self.params['one_step']:
            return self.one_step()
        else:
            if self.params['uq_method'] == 'expectation':
                return self._propagate_expectation()
            elif self.params['uq_method'] == 'trajectory_sampling':
                return self._trajectory_sampling()
            elif self.params['uq_method'] == 'moment_matching':
                return self._moment_matching()   

        return      

    def _propagate_expectation(self):    
        """Expectation method of propagation from the following paper: https://arxiv.org/abs/1805.12114"""  

        no_of_episodes = len(self.ens.seq_input_val)

        all_pred_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_var_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

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
                        # get pred, logvar
                        #delta_mu, delta_logvar = self.ens.models[model](norm_x_m[model]).chunk(2, dim=1)
                        delta_mu, delta_logvar = self.temp_dict[model](norm_x_m[model]).chunk(2, dim=1)
                        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':
                            delta_mu = self.ens.output_filter.invert_torch(delta_mu)
                        pred_unnorm_y = unnorm_x_m[model] + delta_mu
                        all_pred_unnorm_y[start_idx:end_idx,model,step,:] = pred_unnorm_y
                        # since input to model unnorm_x_m[model] is constant the variance of it is just the 
                        # variance of delta delta_logvar.exp()
                        #unnorm_var = torch.add(delta_logvar.exp(), torch.Tensor(np.square(self.ens.input_filter.stdev)).to(device))
                        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':
                            all_var_unnorm_y[start_idx:end_idx,model,step,:] = torch.square(torch.Tensor(self.ens.output_filter.stdev)).unsqueeze(0).to(device)*delta_logvar.exp() # logvar -> var
                        elif self.params['ode_name'] == 'LVolt' or self.params['ode_name'] == 'lorenz':
                            all_var_unnorm_y[start_idx:end_idx,model,step,:] = delta_logvar.exp() # logvar -> var   
                        pred_norm_y = self.ens.input_filter.filter_torch(pred_unnorm_y)
                        models_pred_norm.append(pred_norm_y.unsqueeze(0))
                        models_pred_unnorm.append(pred_unnorm_y.unsqueeze(0))         

                norm_x_m = torch.concatenate(models_pred_norm, dim=0)
                unnorm_x_m = torch.concatenate(models_pred_unnorm, dim=0)

        print(f"Models used for inference are: {model+1}")         
        self.save_scores(all_pred_unnorm_y, all_var_unnorm_y, \
                                    all_true_unnorm_y, no_of_episodes)       

        return       
    
    def save_scores(self, all_pred_unnorm_y: torch.Tensor, all_var_unnorm_y: torch.Tensor,\
                     all_true_unnorm_y: torch.Tensor, no_of_episodes: int):
        """Calculate calibration scores and other scores for ensemble models"""

        mean_pred_unnorm_y = all_pred_unnorm_y.mean(dim=1) 
        var_pred_unnorm_y = all_var_unnorm_y.mean(dim=1) + all_pred_unnorm_y.var(dim=1) # mean of var + var of mean
        #negative_values = var_pred_unnorm_y[var_pred_unnorm_y < 0]
        #print(negative_values)
        lower_z_scores, upper_z_scores = self.z_scores()
        # CALCULATE QUANTILES
        unnorm_mu, unnorm_var, unnorm_upper_mu, unnorm_lower_mu = \
                            self.select_quantiles(mean_pred_unnorm_y, var_pred_unnorm_y,\
                                                                lower_z_scores, upper_z_scores) 
        scores = self.score.get_scores(mean_pred_unnorm_y, var_pred_unnorm_y, unnorm_lower_mu, unnorm_upper_mu, all_true_unnorm_y)

        # save all scores to dict    
        scores_dict = {'calib_scores': scores[0],
                       'wink_scores': scores[1],
                       'pi_widths': scores[2],
                       'mses': scores[3],
                       'nlls': scores[4]
                       }     

        if self.params['one_step']:
            pickle.dump(scores_dict, open(self.ens.model_dir + f"/scores_calib_{self.params['one_step']}onestep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        else:    
            pickle.dump(scores_dict, open(self.ens.model_dir + f"/scores_calib_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        mu_quant_pred_unnorm_y = np.concatenate([unnorm_mu[np.newaxis,:,:,:],\
                                                 unnorm_upper_mu,unnorm_lower_mu], axis=0)
        data_dict = {'all_pred_unnorm_y': all_pred_unnorm_y.detach().cpu().numpy(),
                     'all_true_unnorm_y': all_true_unnorm_y.detach().cpu().numpy(),
                     'mu_quant_pred_unnorm_y': mu_quant_pred_unnorm_y}       

        if self.params['one_step']:
             pickle.dump(data_dict, open(self.ens.model_dir + f"/val_propagate_{self.params['one_step']}onestep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        else:     
            pickle.dump(data_dict, open(self.ens.model_dir + f"/val_propagate_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        errors = all_pred_unnorm_y - all_true_unnorm_y.repeat(self.rand_models,1,1,1).permute(1,0,2,3)
        errors_dict = {"errors": errors.detach().cpu().numpy()}
        if self.params['one_step']:
           pickle.dump(errors_dict, open(self.ens.model_dir + f"/errors_{self.params['one_step']}oenstep_{self.run}run_{self.params['rand_models']}models.pkl", 'wb')) 
        else:   
            pickle.dump(errors_dict, open(self.ens.model_dir + f"/errors_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        horizon = torch.Tensor(np.linspace(0, self.ens.timesteps-1, self.ens.timesteps)).repeat(no_of_episodes,1).numpy()
        
        model_error = 0
        pd_pred_unnorm_y = all_pred_unnorm_y[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_true_unnorm_y = all_true_unnorm_y.reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_errors = errors[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()

        pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
        gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
        resid_cols = ['resid_x','resid_y','resid_z','resid_x_dot','resid_y_dot','resid_z_dot']

        df = pd.DataFrame({
                    'horizon': horizon.reshape(-1,),
                    pred_cols[0]: pd_pred_unnorm_y[:,0],
                    pred_cols[1]: pd_pred_unnorm_y[:,1],
                    #pred_cols[2]: pd_pred_unnorm_y[:,2],
                    #pred_cols[3]: pd_pred_unnorm_y[:,3],
                    #pred_cols[4]: pd_pred_unnorm_y[:,4],
                    #pred_cols[5]: pd_pred_unnorm_y[:,5],
                    gr_cols[0]: pd_true_unnorm_y[:,0],
                    gr_cols[1]: pd_true_unnorm_y[:,1],
                    #gr_cols[2]: pd_true_unnorm_y[:,2],
                    #gr_cols[3]: pd_true_unnorm_y[:,3],
                    #gr_cols[4]: pd_true_unnorm_y[:,4],
                    #gr_cols[5]: pd_true_unnorm_y[:,5],
                    resid_cols[0]: pd_errors[:,0],
                    resid_cols[1]: pd_errors[:,1],
                    #resid_cols[2]: pd_errors[:,2],
                    #resid_cols[3]: pd_errors[:,3],
                    #resid_cols[4]: pd_errors[:,4],
                    #resid_cols[5]: pd_errors[:,5],
                })

        if not self.params['one_step']:
            # Save DataFrame to CSV file
            if os.path.exists(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv"):
                print("Error file already exists!")
            else:                
                print("Error file saved!")
                df.to_csv(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv", index=False)
        return               

    def select_quantiles(self, unnorm_mu:torch.Tensor, unnorm_var:torch.Tensor,\
                     lower_z_scores: torch.Tensor, upper_z_scores: torch.Tensor):
        """
        mu: [batch_size*seq_len,dp_outs] normalized predictions in tensor
        var: [batch_size*seq_len,dp_outs] predicted logvar in tensor
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
    
    def _trajectory_sampling(self):  
        """TS-Inf from the following paper: https://arxiv.org/abs/1805.12114"""  

        no_of_episodes = len(self.ens.seq_input_val)

        all_pred_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.n_particles,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_var_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.n_particles,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

        for idx, example in enumerate(self.ens.seq_val_loader):

            batch_sz = example[0].shape[0]
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
                    norm_x_m = norm_x.repeat(self.rand_models,self.n_particles,1,1).permute(0,2,1,3) # [models,part,batch_sz,input_dim] -> [models,batch_sz,part,input_dim]
                    unnorm_x_m = unnorm_x.repeat(self.rand_models,self.n_particles,1,1).permute(0,2,1,3)

                models_pred_norm = []   
                models_pred_unnorm = [] 

                for model in range(self.params['rand_models']):

                    #self.ens.models[model].eval()
                    self.temp_dict[model].eval()
                    with torch.no_grad():
                        # get pred, logvar
                        #delta_mu, delta_logvar = self.ens.models[model](norm_x_m[model].reshape(batch_sz*self.n_particles,-1)).chunk(2, dim=1)
                        delta_mu, delta_logvar = self.temp_dict[model](norm_x_m[model].reshape(batch_sz*self.n_particles,-1)).chunk(2, dim=1)
                 
                        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':
                            delta_mu = self.ens.output_filter.invert_torch(delta_mu)
                            delta_var = torch.square(torch.Tensor(self.ens.output_filter.stdev)).unsqueeze(0).to(device)*delta_logvar.exp()
                        elif self.params['ode_name'] == 'LVolt' or self.params['ode_name'] == 'lorenz':
                            delta_var = delta_logvar.exp()
                        delta_sample = Normal(delta_mu, delta_var.sqrt()).sample()
                        pred_unnorm_y = unnorm_x_m[model].reshape(batch_sz*self.n_particles,-1) + delta_sample
                        all_pred_unnorm_y[start_idx:end_idx,model,:,step,:] = pred_unnorm_y.reshape(batch_sz,self.n_particles,-1)
                        all_var_unnorm_y[start_idx:end_idx,model,:,step,:] = delta_var.reshape(batch_sz,self.n_particles,-1) # logvar -> var
                        pred_norm_y = self.ens.input_filter.filter_torch(pred_unnorm_y)
                        models_pred_norm.append(pred_norm_y.unsqueeze(0))
                        models_pred_unnorm.append(pred_unnorm_y.unsqueeze(0))        

                norm_x_m = torch.concatenate(models_pred_norm, dim=0)
                unnorm_x_m = torch.concatenate(models_pred_unnorm, dim=0)

        print(f"Models used for inference are: {model+1}")  
        mean_pred_unnorm_y = all_pred_unnorm_y.mean(dim=2)  
        var_pred_unnorm_y = all_pred_unnorm_y.var(dim=2) 

        _mean_pred_unnorm_y = mean_pred_unnorm_y.mean(dim=1) 
        _var_pred_unnorm_y = var_pred_unnorm_y.mean(dim=1) + mean_pred_unnorm_y.var(dim=1) # mean of var + var of mean
        lower_z_scores, upper_z_scores = self.z_scores()
        # CALCULATE QUANTILES
        unnorm_mu, unnorm_var, unnorm_upper_mu, unnorm_lower_mu = \
                            self.select_quantiles(_mean_pred_unnorm_y, _var_pred_unnorm_y,\
                                                                lower_z_scores, upper_z_scores) 
        scores = self.score.get_scores(_mean_pred_unnorm_y, _var_pred_unnorm_y,\
                                        unnorm_lower_mu, unnorm_upper_mu, all_true_unnorm_y)

        # save all scores to dict    
        scores_dict = {'calib_scores': scores[0],
                       'wink_scores': scores[1],
                       'pi_widths': scores[2],
                       'mses': scores[3],
                       'nlls': scores[4]
                       }          

        pickle.dump(scores_dict, open(self.ens.model_dir + f"/scores_calib_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        mu_quant_pred_unnorm_y = np.concatenate([unnorm_mu[np.newaxis,:,:,:],\
                                                 unnorm_upper_mu,unnorm_lower_mu], axis=0)
        data_dict = {'all_pred_unnorm_y': mean_pred_unnorm_y.detach().cpu().numpy(),
                     'all_true_unnorm_y': all_true_unnorm_y.detach().cpu().numpy(),
                     'mu_quant_pred_unnorm_y': mu_quant_pred_unnorm_y}     
        part_spread = {'all_pred_unnorm_y': all_pred_unnorm_y.detach().cpu().numpy()}  
         
        pickle.dump(data_dict, open(self.ens.model_dir + f"/val_propagate_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        pickle.dump(part_spread, open(self.ens.model_dir + f"/part_spread_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        errors = mean_pred_unnorm_y - all_true_unnorm_y.repeat(self.rand_models,1,1,1).permute(1,0,2,3)
        errors_dict = {"errors": errors.detach().cpu().numpy()}
        pickle.dump(errors_dict, open(self.ens.model_dir + f"/errors_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        horizon = torch.Tensor(np.linspace(0, self.ens.timesteps-1, self.ens.timesteps)).repeat(no_of_episodes,1).numpy()
        
        model_error = 0
        pd_pred_unnorm_y = mean_pred_unnorm_y[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_true_unnorm_y = all_true_unnorm_y.reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_errors = errors[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()

        pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
        gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
        resid_cols = ['resid_x','resid_y','resid_z','resid_x_dot','resid_y_dot','resid_z_dot']

        df = pd.DataFrame({
                    'horizon': horizon.reshape(-1,),
                    pred_cols[0]: pd_pred_unnorm_y[:,0],
                    pred_cols[1]: pd_pred_unnorm_y[:,1],
                    #pred_cols[2]: pd_pred_unnorm_y[:,2],
                    #pred_cols[3]: pd_pred_unnorm_y[:,3],
                    #pred_cols[4]: pd_pred_unnorm_y[:,4],
                    #pred_cols[5]: pd_pred_unnorm_y[:,5],
                    gr_cols[0]: pd_true_unnorm_y[:,0],
                    gr_cols[1]: pd_true_unnorm_y[:,1],
                    #gr_cols[2]: pd_true_unnorm_y[:,2],
                    #gr_cols[3]: pd_true_unnorm_y[:,3],
                    #gr_cols[4]: pd_true_unnorm_y[:,4],
                    #gr_cols[5]: pd_true_unnorm_y[:,5],
                    resid_cols[0]: pd_errors[:,0],
                    resid_cols[1]: pd_errors[:,1],
                    #resid_cols[2]: pd_errors[:,2],
                    #resid_cols[3]: pd_errors[:,3],
                    #resid_cols[4]: pd_errors[:,4],
                    #resid_cols[5]: pd_errors[:,5],
                })
    
        # Save DataFrame to CSV file
        df.to_csv(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv", index=False)

        return 

    def _moment_matching(self):  
        """Moment Matching from the following paper: https://arxiv.org/abs/1805.12114"""  

        no_of_episodes = len(self.ens.seq_input_val)
        # quantiles to be sampled at t-th timestep of multi-step predictions
        samp_quant = torch.Tensor(np.linspace(0, 1.0, self.rand_models*self.n_particles + 2)[1:-1]).to(device)

        all_pred_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.n_particles,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_var_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.n_particles,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

        for idx, example in enumerate(self.ens.seq_val_loader):

            batch_sz = example[0].shape[0]
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
                    norm_x_m = norm_x.repeat(self.rand_models,self.n_particles,1,1).permute(0,2,1,3) # [models,part,batch_sz,input_dim] -> [models,batch_sz,part,input_dim]
                    unnorm_x_m = unnorm_x.repeat(self.rand_models,self.n_particles,1,1).permute(0,2,1,3)

                models_pred_norm = []   
                models_pred_unnorm = [] 

                for model in range(self.params['rand_models']):

                    #self.ens.models[model].eval()
                    self.temp_dict[model].eval()
                    with torch.no_grad():
                        # get pred, logvar
                        # TODO: Unnormalize delta and logvar
                        #delta_mu, delta_logvar = self.ens.models[model](norm_x_m[model].reshape(batch_sz*self.n_particles,-1)).chunk(2, dim=1)
                        delta_mu, delta_logvar = self.temp_dict[model](norm_x_m[model].reshape(batch_sz*self.n_particles,-1)).chunk(2, dim=1)
                        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':
                            delta_mu = self.ens.output_filter.invert_torch(delta_mu)
                            delta_var = torch.square(torch.Tensor(self.ens.output_filter.stdev)).unsqueeze(0).to(device)*delta_logvar.exp()
                        elif self.params['ode_name'] == 'LVolt' or self.params['ode_name'] == 'lorenz':
                            delta_var = delta_logvar.exp()                        
                        delta_sample = Normal(delta_mu, delta_var.sqrt()).sample()
                        pred_unnorm_y = unnorm_x_m[model].reshape(batch_sz*self.n_particles,-1) + delta_sample
                        #all_pred_unnorm_y[start_idx:end_idx,model,:,step,:] = pred_unnorm_y.reshape(batch_sz,self.n_particles,-1)
                        #all_var_unnorm_y[start_idx:end_idx,model,:,step,:] = delta_logvar.exp().reshape(batch_sz,self.n_particles,-1) # logvar -> var
                        pred_norm_y = self.ens.input_filter.filter_torch(pred_unnorm_y)
                        models_pred_norm.append(pred_norm_y.unsqueeze(0))
                        models_pred_unnorm.append(pred_unnorm_y.unsqueeze(0))         

                norm_x_m = torch.concatenate(models_pred_norm, dim=0) # [models,batch_sz*part,input_dim]
                unnorm_x_m_ = torch.concatenate(models_pred_unnorm, dim=0) 
                #print(unnorm_x_m_.reshape(self.rand_models,batch_sz,self.n_particles,-1)[:,0,:5,0])

                norm_x_m = norm_x_m.reshape(self.rand_models,batch_sz,self.n_particles,-1) # [models,batch_sz,part,input_dim]
                norm_x_m = norm_x_m.permute(0,2,1,3) # [models,part,batch_sz,input_dim]
                norm_x_m = norm_x_m.reshape(self.rand_models*self.n_particles,batch_sz,-1) # [models*part,batch_sz,input_dim]
                # do moment matching with respect to bootstraps and particles
                # predictions from all models and particles will be re-cast to Gaussian
                
                mean_t = norm_x_m.mean(dim=0) # take mean w/r/t models and particles # [batch_sz,input_dim]
                var_t = norm_x_m.var(dim=0)  # take var w/r/t models and particles # [batch_sz,input_dim]
                # Take samples from inverse CDF Gaussian at t-th timestep
                gaussian_at_t = Normal(mean_t, var_t.sqrt()) 
                #norm_x_samp = gaussian_at_t.icdf(samp_quant.unsqueeze(-1).unsqueeze(-1).repeat(1,\
                #                    mean_t.shape[0],mean_t.shape[1]))                          # [models*part,batch_sz,input_dim]         
                norm_x_samp = gaussian_at_t.sample((self.rand_models*self.n_particles,))           
                norm_x_samp = norm_x_samp.reshape(self.rand_models,self.n_particles,batch_sz,-1) # [models,part,batch_sz,input_dim]
                norm_x_samp = norm_x_samp.permute(0,2,1,3)                                          # [models,batch_sz,part,input_dim]
                norm_x_m = norm_x_samp.reshape(self.rand_models,batch_sz*self.n_particles,-1) # [models,batch_sz*part,input_dim]

                unnorm_x_m = self.ens.input_filter.invert_torch(norm_x_m.reshape(norm_x_m.shape[0]*norm_x_m.shape[1],-1)) # [model*batch_sz*part,input_dim]
                unnorm_x_m = unnorm_x_m.reshape(self.rand_models,batch_sz*self.n_particles,-1) # [models,batch_sz*part,input_dim]
                all_pred_unnorm_y[start_idx:end_idx,:,:,step,:] = unnorm_x_m.reshape(self.rand_models,batch_sz,self.n_particles,-1).permute(1,0,2,3)
                #print(unnorm_x_m.reshape(self.rand_models,batch_sz,self.n_particles,-1)[:,0,:5,0])
                #sys.exit()
        print(f"Models used for inference are: {model+1}")     

        mean_pred_unnorm_y = all_pred_unnorm_y.mean(dim=2) # mean w/r/t the particles
        var_pred_unnorm_y = all_pred_unnorm_y.var(dim=2) # variance w/r/t the particles

        _mean_pred_unnorm_y = mean_pred_unnorm_y.mean(dim=1) # mean w/r/t models
        _var_pred_unnorm_y = var_pred_unnorm_y.mean(dim=1) + mean_pred_unnorm_y.var(dim=1) # mean of var + var of mean
        lower_z_scores, upper_z_scores = self.z_scores()

        # CALCULATE QUANTILES
        unnorm_mu, unnorm_var, unnorm_upper_mu, unnorm_lower_mu = \
                            self.select_quantiles(_mean_pred_unnorm_y, _var_pred_unnorm_y,\
                                                                lower_z_scores, upper_z_scores) 
        scores = self.score.get_scores(_mean_pred_unnorm_y, _var_pred_unnorm_y,\
                                        unnorm_lower_mu, unnorm_upper_mu, all_true_unnorm_y)

        # save all scores to dict    
        scores_dict = {'calib_scores': scores[0],
                       'wink_scores': scores[1],
                       'pi_widths': scores[2],
                       'mses': scores[3],
                       'nlls': scores[4]
                       }  
        pickle.dump(scores_dict, open(self.ens.model_dir + f"/scores_calib_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        mu_quant_pred_unnorm_y = np.concatenate([unnorm_mu[np.newaxis,:,:,:],\
                                                 unnorm_upper_mu,unnorm_lower_mu], axis=0)
        data_dict = {'all_pred_unnorm_y': mean_pred_unnorm_y.detach().cpu().numpy(),
                     'all_true_unnorm_y': all_true_unnorm_y.detach().cpu().numpy(),
                     'mu_quant_pred_unnorm_y': mu_quant_pred_unnorm_y}     
        part_spread = {'all_pred_unnorm_y': all_pred_unnorm_y.detach().cpu().numpy()}  
         
        pickle.dump(data_dict, open(self.ens.model_dir + f"/val_propagate_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        pickle.dump(part_spread, open(self.ens.model_dir + f"/part_spread_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))

        errors = mean_pred_unnorm_y - all_true_unnorm_y.repeat(self.rand_models,1,1,1).permute(1,0,2,3)
        errors_dict = {"errors": errors.detach().cpu().numpy()}
        pickle.dump(errors_dict, open(self.ens.model_dir + f"/errors_{self.params['uq_method']}_{self.run}run_{self.params['rand_models']}models.pkl", 'wb'))
        horizon = torch.Tensor(np.linspace(0, self.ens.timesteps-1, self.ens.timesteps)).repeat(no_of_episodes,1).numpy()
        
        model_error = 0
        pd_pred_unnorm_y = mean_pred_unnorm_y[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_true_unnorm_y = all_true_unnorm_y.reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        pd_errors = errors[:,model_error,:,:].reshape(-1,self.ens.output_dim).detach().cpu().numpy()
        """
        pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
        gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
        resid_cols = ['resid_x','resid_y','resid_z','resid_x_dot','resid_y_dot','resid_z_dot']

        df = pd.DataFrame({
                    'horizon': horizon.reshape(-1,),
                    pred_cols[0]: pd_pred_unnorm_y[:,0],
                    pred_cols[1]: pd_pred_unnorm_y[:,1],
                    #pred_cols[2]: pd_pred_unnorm_y[:,2],
                    #pred_cols[3]: pd_pred_unnorm_y[:,3],
                    #pred_cols[4]: pd_pred_unnorm_y[:,4],
                    #pred_cols[5]: pd_pred_unnorm_y[:,5],
                    gr_cols[0]: pd_true_unnorm_y[:,0],
                    gr_cols[1]: pd_true_unnorm_y[:,1],
                    gr_cols[2]: pd_true_unnorm_y[:,2],
                    gr_cols[3]: pd_true_unnorm_y[:,3],
                    gr_cols[4]: pd_true_unnorm_y[:,4],
                    gr_cols[5]: pd_true_unnorm_y[:,5],
                    resid_cols[0]: pd_errors[:,0],
                    resid_cols[1]: pd_errors[:,1],
                    resid_cols[2]: pd_errors[:,2],
                    resid_cols[3]: pd_errors[:,3],
                    resid_cols[4]: pd_errors[:,4],
                    resid_cols[5]: pd_errors[:,5],
                })
    
        # Save DataFrame to CSV file
        df.to_csv(self.ens.model_dir + f"/{self.params['dataset_name']}_errors.csv", index=False)
        """

        return
        
    def one_step(self):
        """Calculate the one-step calibration of probabilistic ensembles"""
            
        no_of_episodes = len(self.ens.seq_input_val)

        all_pred_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_var_unnorm_y = torch.zeros((no_of_episodes,self.rand_models,self.ens.timesteps,self.ens.output_dim)).to(device)
        all_true_unnorm_y = torch.zeros((no_of_episodes,self.ens.timesteps,self.ens.output_dim)).to(device)

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
                        # get pred, logvar
                        #delta_mu, delta_logvar = self.ens.models[model](norm_x_m[model]).chunk(2, dim=1)
                        delta_mu, delta_logvar = self.temp_dict[model](norm_x_m[model]).chunk(2, dim=1)
                        pred_unnorm_y = unnorm_x_m[model] + delta_mu
                        all_pred_unnorm_y[start_idx:end_idx,model,step,:] = pred_unnorm_y
                        # unnormalize variance 
                        unnorm_var = torch.add(delta_logvar.exp(), torch.Tensor(np.square(self.ens.input_filter.stdev)).to(device))
                        all_var_unnorm_y[start_idx:end_idx,model,step,:] = unnorm_var # logvar -> var   

        print(f"Models used for inference are: {model+1}")                             

        self.save_scores(all_pred_unnorm_y, all_var_unnorm_y, \
                                    all_true_unnorm_y, no_of_episodes)   

        return     
