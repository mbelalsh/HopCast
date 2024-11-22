import torch
from torch.utils.data import Dataset
import torch.nn as nn
from utils import GaussianMSELoss, VarNormMSELoss
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 100, 50, 10 % uniformly random
class TransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, input_data, output_data):

        self.data_X = input_data
        self.data_y = output_data          

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]


class EnsembleTransitionDataset(Dataset):
    ## Dataset wrapper for sampled transitions
    def __init__(self, input_data, output_data, n_models=1):

        data_count = input_data.shape[0]

        idxs = torch.randint(data_count, size=[n_models, data_count])
        self._n_models = n_models
        self.data_X = input_data[idxs]
        self.data_y = output_data[idxs]
 
    def __len__(self):
        return self.data_X.shape[1]

    def __getitem__(self, index):
        return self.data_X[:, index], self.data_y[:, index]

class Model(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 params: dict,
                 h: int = 1024,
                 seed=0,
                 l2_reg_multiplier=1.,
                 num=0):

        super(Model, self).__init__()
        torch.manual_seed(seed)
        self.params = params
        if self.params['bayesian']:
            self.model = BayesianNeuralNetwork(input_dim, output_dim, params,\
                                                params['num_nodes'], l2_reg_multiplier, seed)
            self.weights = self.model.weights
        else:            
            self.model = VanillaNeuralNetwork(input_dim, output_dim, params,\
                                            params['num_layers'], params['num_nodes'], seed)
            self.weights = self.model.parameters()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _train_model_forward(self, x_batch):
        self.model.train()    # TRAINING MODE
        self.model.zero_grad()
        x_batch = x_batch.to(device, non_blocking=True)
        y_pred = self.forward(x_batch)
        return y_pred

    def train_model_forward(self, x_batch, y_batch):
        y_batch = y_batch.to(device, non_blocking=True)
        y_pred = self._train_model_forward(x_batch)
        y_batch = y_batch
        #loss = self.model.loss(y_pred, y_batch, self.params)
        loss = self.model.loss(y_pred, y_batch)
        return loss

    def get_predictions_from_loader(self, data_loader, return_targets=False, return_sample=False):
        self.model.eval()   # EVALUATION MODE
        preds, targets = [], []
        with torch.no_grad():
            for x_batch_val, y_batch_val in data_loader:
                x_batch_val, y_batch_val= x_batch_val.to(device, non_blocking=True),\
                                                y_batch_val.to(device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)

                preds.append(y_pred_val)
                if return_targets:
                    y_batch_val = y_batch_val
                    targets.append(y_batch_val)

        preds = torch.vstack(preds)

        if return_sample:
            mu, logvar = preds.chunk(2, dim=1)
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            sample = dist.sample()
            preds = torch.cat((sample, preds), dim=1)

        if return_targets:
            targets = torch.vstack(targets)
            return preds, targets
        else:
            return preds
                
    def get_validation_loss(self, validation_loader):
        self.model.eval()
        preds, targets = self.get_predictions_from_loader(validation_loader, return_targets=True)

        #return self.model.loss(preds, targets, self.params).item()
        return self.model.loss(preds, targets).item()
    
class VanillaNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                output_dim: int,
                params: dict,
                num_layers: int = 4,
                num_nodes: int = 200,            
                seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.params = params
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, num_nodes) if i == 0 else nn.Linear(num_nodes, num_nodes)
            for i in range(num_layers)
        ])
        self.delta = nn.Linear(num_nodes, output_dim)
        self.relu = nn.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.params['ode_name'] == 'glycolytic' or self.params['ode_name'] == 'lorenz96' or self.params['ode_name'] == 'FHNag':   
            self.loss = nn.MSELoss()
        elif self.params['ode_name'] == 'LVolt' or self.params['ode_name'] == 'lorenz': 
            self.loss = VarNormMSELoss(params)
        self.to(device)

    def forward(self, x):

        for layer in self.layers:
            x = self.relu(layer(x))
        delta = self.delta(x)
        return delta        

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 params: dict,
                 h: int = 200,
                 l2_reg_multiplier=1.,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        #del self.network
        self.params = params
        self.fc1 = nn.Linear(input_dim, h)
        reinitialize_fc_layer_(self.fc1)
        self.layers = [self.fc1]
        for i in range(params['num_layers']-1):
            setattr(self, f"fc{i+2}", nn.Linear(h, h))
            self.layers.append(getattr(self, f"fc{i+2}"))
            reinitialize_fc_layer_(self.layers[i+1])   
        #self.fc2 = nn.Linear(h, h)
        #reinitialize_fc_layer_(self.fc2)
        #self.fc3 = nn.Linear(h, h)
        #reinitialize_fc_layer_(self.fc3)
        #self.fc4 = nn.Linear(h, h)
        #reinitialize_fc_layer_(self.fc4)
        self.y = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.y)
        self.logvar = nn.Linear(h, output_dim)
        reinitialize_fc_layer_(self.logvar)
        self.layers_all = self.layers + [self.y] + [self.logvar]
        self.loss = GaussianMSELoss(params)
        self.activation = nn.SiLU()
        self.lambda_prec = 1.0
        self.max_logvar = None
        self.min_logvar = None
        params = [] # 12 dicts for all layers (w and b) from get_weight_bias_parameters_with_decays method 
        #self.layers_all = [self.fc1, self.fc2, self.fc3, self.fc4, self.y, self.logvar] #
        self.decays = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]) * l2_reg_multiplier
        for layer, decay in zip(self.layers_all, self.decays):
            params.extend(get_weight_bias_parameters_with_decays(layer, decay))
        self.weights = params
        self.to(device)

    @staticmethod
    def filter_inputs(input, input_filter):
        input_f = input_filter.filter_torch(input)
        return input_f   

    def get_l2_reg_loss(self):
        l2_loss = 0
        for layer, decay in zip(self.layers_all, self.decays):
            for name, parameter in layer.named_parameters():
                if 'weight' in name:
                    l2_loss += parameter.pow(2).sum() / 2 * decay
        return l2_loss

    def update_logvar_limits(self, max_logvar, min_logvar):
        self.max_logvar, self.min_logvar = max_logvar, min_logvar

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.activation(layer(x))
        """    
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        """
        y = self.y(x)
        logvar = self.logvar(x)

        # Taken from the PETS code to stabilise training (https://github.com/kchua/handful-of-trials)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return torch.cat((y, logvar), dim=1)

    def get_next_state(self, input: torch.Tensor, deterministic=False, return_mean=False):
        input_torch = torch.FloatTensor(input).to(device)
        mu, logvar = self.forward(input_torch).chunk(2, dim=1)
        mu_orig = mu

        if not deterministic:
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            mu = dist.sample()
 
        if return_mean:
            return (mu, mu_orig), logvar

        return mu, logvar

def reinitialize_fc_layer_(fc_layer):
    """
    Helper function to initialize a fc layer to have a truncated normal over the weights, and zero over the biases
    """
    input_dim = fc_layer.weight.shape[1]
    std = get_trunc_normal_std(input_dim)
    torch.nn.init.trunc_normal_(fc_layer.weight, std=std, a=-2 * std, b=2 * std)
    torch.nn.init.zeros_(fc_layer.bias)


def get_trunc_normal_std(input_dim):
    """
    Returns the truncated normal standard deviation required for weight initialization
    """
    return 1 / (2 * np.sqrt(input_dim))

def get_weight_bias_parameters_with_decays(fc_layer, decay):
    """
    For the fc_layer, extract only the weight from the .parameters() method so we don't regularize the bias terms
    """
    decay_params = []
    non_decay_params = []
    for name, parameter in fc_layer.named_parameters():
        if 'weight' in name:
            decay_params.append(parameter)
        elif 'bias' in name:
            non_decay_params.append(parameter)

    decay_dicts = [{'params': decay_params, 'weight_decay': decay}, {'params': non_decay_params, 'weight_decay': 0.}]

    return decay_dicts           