import torch.nn as nn
import torch 
import numpy as np
import sys
import os 
from pathlib import Path
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import random
from copy import deepcopy
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(params):

    np.random.seed(params['seed'])
    random.seed(params['seed'])
    torch.manual_seed(params['seed'])        

class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)  # ob_dim or ac_dim
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)   
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
        self.stdev[self.stdev <= self.eps] = 1.0

    def reset(self):
        self.__init__(self.shape, self.clip)

    def get_torch_stdev(self):
        self.update_torch()    
        return self.torch_stdev

    def update_torch(self):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean[:x.shape[1]]) / self.stdev[:x.shape[1]]), -self.clip, self.clip)
    
    def ac_filter(self, x):
        return np.clip(((x - self.mean[-x.shape[1]:]) / self.stdev[-x.shape[1]:]), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean[:x.shape[1]]) / self.torch_stdev[:x.shape[1]]), -self.clip, self.clip)

    def ac_filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean[-x.shape[1]:]) / self.torch_stdev[-x.shape[1]:]), -self.clip, self.clip)
    
    def invert(self, x):
        return (x * self.stdev) + self.mean

    def ac_invert(self, x: np.ndarray):
        return (x * self.stdev[-x.shape[1]:]) + self.mean[-x.shape[1]:]    

    def invert_torch(self, x: torch.Tensor):
        self.update_torch()
        return (x * self.torch_stdev) + self.torch_mean

    def ac_invert_torch(self, x: torch.Tensor):
        self.update_torch()
        return (x * self.torch_stdev[-x.shape[1]:]) + self.torch_mean[-x.shape[1]:]    

def get_exp_weights(decay_rate: float, num_timesteps: int):

    timesteps = np.arange(num_timesteps)
    return np.exp(-decay_rate * timesteps)


class MultiStepMSELoss(nn.Module):

    def __init__(self):
        super(MultiStepMSELoss, self).__init__()
        self.loss_weights = torch.FloatTensor(get_exp_weights(decay_rate=0.9, num_timesteps=50)).to(device)

    def forward(self, y_pred, delta_batch, break_steps, params):

        losses = torch.zeros((y_pred.shape[0])).to(device)
        for i in range(y_pred.shape[0]):
            y_pred_sample = y_pred[i,:break_steps[i]+1,:].reshape(-1,y_pred.shape[2])
            delta_sample = delta_batch[i,:break_steps[i]+1,:].reshape(-1,delta_batch.shape[2])
            #losses[i] = torch.mean((y_pred_sample - delta_sample)**2)
            # exponentially weighted loss (Normalize or Not? Unnormalized yet)
            #losses[i] = (self.loss_weights[:break_steps[i]+1]*torch.mean((y_pred_sample - delta_sample)**2, dim=1)).mean()
            # var-weighted loss (Decided not to normalize weights)
            losses[i] = torch.mean(params['norm_loss_wghts']*torch.mean((y_pred_sample - delta_sample)**2, dim=0))

        return losses.mean()

class VarNormMSELoss(nn.Module):

    def __init__(self, params):
        super(VarNormMSELoss, self).__init__()
        self.params = params

    def forward(self, y_pred, delta_batch):

        loss = self.params['norm_loss_wghts']*torch.mean((y_pred-delta_batch)**2, dim=0)

        return loss.mean()    

class GaussianMSELoss(nn.Module):

    def __init__(self, params):
        super(GaussianMSELoss, self).__init__()
        self.params = params

    def forward(self, mu_logvar, target, logvar_loss = True):
        mu, logvar = mu_logvar.chunk(2, dim=1)
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu)**2 * inv_var).mean()
        else:
            return ((target - mu)**2).mean()

class MultivariateGaussianLoss(nn.Module):

    def __init__(self):
        super(MultivariateGaussianLoss, self).__init__()

    def chol_covariance(self, lower_triangular, pred):
        batch_size, output_dim = lower_triangular.shape
        covariance = torch.zeros(batch_size, pred.shape[1], pred.shape[1], device=lower_triangular.device)
        idx = torch.tril_indices(pred.shape[1], pred.shape[1])
        covariance[:, idx[0], idx[1]] = lower_triangular

        return covariance 

    def chol_covariance2(self, lower_triangular):

        diag_ele = F.softplus(lower_triangular.diagonal(dim1=-2, dim2=-1))
        diag_mat = torch.diag_embed(diag_ele)
        covariance = torch.tril(lower_triangular, diagonal=-1)
        covariance = covariance + diag_mat
        covariance = covariance @ covariance.transpose(-2, -1)

        return covariance


    def forward(self, mean, y_true):
        pred, logvar = mean[:,:y_true.shape[1]], mean[:,y_true.shape[1]:]
        #covariance = self.chol_covariance(logvar.exp(), pred)
        #m = torch.distributions.multivariate_normal.MultivariateNormal(pred, scale_tril=covariance)
        lower_trian = self.chol_covariance(logvar.exp(), pred)
        covariance = self.chol_covariance2(lower_trian)
        m = torch.distributions.multivariate_normal.MultivariateNormal(pred, covariance_matrix=covariance)
        return -m.log_prob(y_true).mean()


def prepare_data(input_data:np.ndarray, input_filter:MeanStdevFilter, is_action:bool=False):

    if is_action:
        input_filtered = input_filter.ac_filter(input_data)
    else:    
        input_filtered = input_filter.filter(input_data)
    
    return input_filtered 

def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)

def plot_part(mu_part: np.ndarray, ground_truth: np.ndarray, no_of_outputs:int, fig_name: str):

    no_of_inputs = np.linspace(0, stop=(mu_part.shape[1]-1), num=mu_part.shape[1], dtype=int)
    #labels = [r"$\Theta$[rad]", r"$\dot{\Theta}$[rad/s]", r"$x$[m]", r"$\dot{x}$[m/s]"]
    labels = [r"height[m]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]",\
              r"$v[m/s]$", r"$v[m/s]$", r"$\dot{\Theta}$[rad/s]", r"$\dot{\Theta}$[rad/s]",\
                r"$\dot{\Theta}$[rad/s]",  r"$\dot{\Theta}$[rad/s]", "rewards"]
    fig = plt.figure() 

    gs = fig.add_gridspec(no_of_outputs, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(13*no_of_outputs)
    fig.set_figwidth(30)

    plt.rc('font', size=41)          # controls default text sizes
    plt.rc('axes', titlesize=41)     # fontsize of the axes title
    plt.rc('axes', labelsize=63)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=46)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=46)    # fontsize of the tick labels
    plt.rc('legend', fontsize=47)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    for i in range(no_of_outputs):

        ax[i].fill_between(no_of_inputs, np.quantile(mu_part[:,:,i], 0.025, axis=0).reshape(-1,),\
                                    np.quantile(mu_part[:,:,i], 0.975, axis=0).reshape(-1,), alpha=0.3)
        r1, = ax[i].plot(no_of_inputs, ground_truth[:,i].reshape(-1,), "r", marker='*', markersize=4)
        for j in range(mu_part.shape[0]):
            k1, = ax[i].plot(no_of_inputs, mu_part[j,:,i].reshape(-1,), "k", markersize=6)
        ax[i].grid(True)
        ax[i].set_ylabel(labels[i])
        ax[i].spines['top'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5) # set_visible(False)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)

    ax[no_of_outputs-1].set_xlabel('time [s]')
    ax[0].legend((r1, k1), ('Ground Truth', 'Predictions'),\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
    #plt.show()    

def plot_part_mhn(mu_part: np.ndarray, ground_truth: np.ndarray, no_of_outputs:int, fig_name: str):

    no_of_inputs = np.linspace(0, stop=(mu_part.shape[2]-1), num=mu_part.shape[2], dtype=int)
    #labels = [r"$\Theta$[rad]", r"$\dot{\Theta}$[rad/s]", r"$x$[m]", r"$\dot{x}$[m/s]"]
    labels = [r"height[m]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]",\
              r"$v[m/s]$", r"$v[m/s]$", r"$\dot{\Theta}$[rad/s]", r"$\dot{\Theta}$[rad/s]",\
                r"$\dot{\Theta}$[rad/s]",  r"$\dot{\Theta}$[rad/s]", "rewards"]
    fig = plt.figure() 

    gs = fig.add_gridspec(no_of_outputs, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(13*no_of_outputs)
    fig.set_figwidth(30) # 70

    plt.rc('font', size=41)          # controls default text sizes
    plt.rc('axes', titlesize=41)     # fontsize of the axes title
    plt.rc('axes', labelsize=63)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=46)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=46)    # fontsize of the tick labels
    plt.rc('legend', fontsize=47)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    #for i in range(no_of_outputs):
    #    if i == 1: # hack remove it later
    #        continue
        #ax[i].fill_between(no_of_inputs, lower_mu[i,:].reshape(-1,), upper_mu[i,:].reshape(-1,), alpha=0.3)
    for head in range(mu_part.shape[0]):

        r1, = ax.plot(no_of_inputs, ground_truth[:,:].reshape(-1,), "r", marker='*', markersize=4)
        k1, = ax.plot(no_of_inputs, mu_part[head,0,:,:].reshape(-1,), "k", marker='*', markersize=6)
        colors = ["g", "b"]
        no_of_alphas = (mu_part[head].shape[0]-1) // 2
        color_id = 0
        for j in range(1,mu_part[head].shape[0]):
            k1, = ax.plot(no_of_inputs, mu_part[head,j,:,:].reshape(-1,), colors[color_id], markersize=6)
            if (j%no_of_alphas) == 0:
                color_id += 1
    ax.grid(True)
    ax.set_ylabel(labels[9])
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5) # set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    ax.set_xlabel('time [s]')
    ax.legend(('Ground Truth', 'Prediction', 'Samples'),\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()      

def plot_part_mhn_many(mu_part: np.ndarray, ground_truth: np.ndarray, no_of_outputs:int, fig_name: str):

    no_of_inputs = np.linspace(0, stop=(mu_part.shape[2]-1), num=mu_part.shape[2], dtype=int)
    #labels = [r"$\Theta$[rad]", r"$\dot{\Theta}$[rad/s]", r"$x$[m]", r"$\dot{x}$[m/s]"]
    labels = [r"height[m]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]",\
              r"$v[m/s]$", r"$v[m/s]$", r"$\dot{\Theta}$[rad/s]", r"$\dot{\Theta}$[rad/s]",\
                r"$\dot{\Theta}$[rad/s]",  r"$\dot{\Theta}$[rad/s]", "rewards"]
    fig = plt.figure() 

    gs = fig.add_gridspec(no_of_outputs, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(14*no_of_outputs)
    fig.set_figwidth(37)

    plt.rc('font', size=41)          # controls default text sizes
    plt.rc('axes', titlesize=41)     # fontsize of the axes title
    plt.rc('axes', labelsize=63)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=46)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=46)    # fontsize of the tick labels
    plt.rc('legend', fontsize=47)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title
    line_width = 3.5 # this line width control the plot line width and legend line width
    marker_size = 10
    shading_alphas = [0.3,0.32]

    for i in range(no_of_outputs):

        for head in range(mu_part.shape[0]):

            ax[i].fill_between(no_of_inputs, np.min(mu_part[head], axis=0)[:,i].reshape(-1,),\
                                        np.max(mu_part[head], axis=0)[:,i].reshape(-1,), alpha=shading_alphas[head])
            r1, = ax[i].plot(no_of_inputs, ground_truth[:,i].reshape(-1,), "r", marker='*',\
                                                        markersize=marker_size, lw=line_width)
            m1, = ax[i].plot(no_of_inputs, mu_part[head,0,:,i].reshape(-1,), "m", marker='*',\
                                                        markersize=marker_size, lw=line_width)
            #k1, = ax[i].plot(no_of_inputs, mu_part[head,1,:,i].reshape(-1,), "k", marker='*',\
            #                                            markersize=marker_size, lw=line_width)
            colors = ["g", "b"]
            no_of_alphas = (mu_part[head].shape[0]-2) // 2 # change it to one if you don't have mean quantile
            color_id = 0
            """
            for j in range(1,mu_part[head].shape[0]-1):
                if colors[color_id] == "g":
                    g1, = ax[i].plot(no_of_inputs, mu_part[head,j+1,:,i].reshape(-1,), colors[color_id], markersize=marker_size, lw=line_width)
                else:    
                    b1, = ax[i].plot(no_of_inputs, mu_part[head,j+1,:,i].reshape(-1,), colors[color_id], markersize=marker_size, lw=line_width)
                if (j%no_of_alphas) == 0:
                    color_id += 1
            # just so I could comment these out in case I don't want to plot orig traj.        
            colors = (r1,m1,k1,b1,g1)    
            legend = ('Ground Truth', 'Mean Quant.', 'Prediction', 'Upper CI', 'Lower CI')
            """

        colors = (r1,m1)    
        legend = ('Ground Truth', 'Mean Quant.')    
                
        ax[i].grid(True)
        ax[i].set_ylabel(labels[i])
        ax[i].spines['top'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5) # set_visible(False)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)

    ax[-1].set_xlabel('time [s]')
    ax[0].legend(colors, legend,\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close() 

def plot_many(mu: np.ndarray, upper_mu: np.ndarray, lower_mu: np.ndarray,\
               ground_truth: np.ndarray, no_of_outputs:int, fig_name:str):

    no_of_inputs = np.linspace(0, stop=(mu.shape[1]-1), num=mu.shape[1], dtype=int)
    #labels = [r"$\Theta$[rad]", r"$\dot{\Theta}$[rad/s]", r"$x$[m]", r"$\dot{x}$[m/s]"]
    """
    labels = [r"height[m]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]",\
              r"$v[m/s]$", r"$v[m/s]$", r"$\dot{\Theta}$[rad/s]", r"$\dot{\Theta}$[rad/s]",\
                r"$\dot{\Theta}$[rad/s]",  r"$\dot{\Theta}$[rad/s]"]
    """
    labels = ['action 1', 'action 2', 'action 3', 'return']
    fig = plt.figure() 

    gs = fig.add_gridspec(no_of_outputs, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(13*no_of_outputs)
    fig.set_figwidth(30)

    plt.rc('font', size=41)          # controls default text sizes
    plt.rc('axes', titlesize=41)     # fontsize of the axes title
    plt.rc('axes', labelsize=63)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=46)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=46)    # fontsize of the tick labels
    plt.rc('legend', fontsize=47)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    for i in range(no_of_outputs):
        #ax[i].fill_between(no_of_inputs, lower_mu[i,:].reshape(-1,), upper_mu[i,:].reshape(-1,), alpha=0.3)
        r1, = ax[i].plot(no_of_inputs, ground_truth[i,:].reshape(-1,), "r", marker='*', markersize=4)
        k1, = ax[i].plot(no_of_inputs, mu[i,:].reshape(-1,), "k", marker='*', markersize=6)
        ax[i].grid(True)
        ax[i].set_ylabel(labels[i])
        ax[i].spines['top'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5) # set_visible(False)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)

    ax[no_of_outputs-1].set_xlabel('time [s]')
    ax[0].legend(('Ground Truth', 'Predictions'),\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()    

def error_hist(errors: np.ndarray, fig_name: str):

    plt.hist(errors, bins=200, color='blue', alpha=0.7, edgecolor='black', density=True)

    plt.title('Error Histogram')
    plt.xlabel('Errors')
    plt.ylabel('Density')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()