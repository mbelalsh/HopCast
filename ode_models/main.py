import argparse
import yaml
from typing import Dict, Tuple
from prepare_data import DataProcessor
from train import Ensemble
from uq_det import UncPropagate # DETERMINISTIC ENSEMBLES
from uq_prop import UncPropDeepEns # PROBABILISTIC ENSEMBLES
import torch
import os
from utils import check_or_make_folder, set_seed
import sys

def train(params: Dict):

    # change directory to save model/data
    params['ode_name'] = params['data_path'].split("/")[1]
    params['dataset_name'] = params['data_path'].split("/")[-1].split(".")[0]
    data_dir = params['data_path'].split("/")[1]
    check_or_make_folder(f"./{data_dir}")
    os.chdir(f"./{data_dir}")
    torch.cuda.empty_cache()
    set_seed(params)

    # prepare data and train attention model
    data_proc = DataProcessor(params)
    dataset = data_proc.get_data(load_dpEn=params['load_dpEn']) # update params dict
    deep_en = Ensemble(params, data_proc, dataset)
    if params['bayesian']:
        unc_prop = UncPropDeepEns(params, deep_en)
    else:    
        unc_prop = UncPropagate(params, deep_en)

    if not params['load_dpEn']:
        deep_en.set_loaders()
        deep_en.train_model(max_epochs=params['epochs'], save_model=True, )
    else:    
        # load calib_data and model weights from the file
        deep_en.load_model()
        for run in range(3):
            unc_prop.propagate(run=run)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--yaml_file', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--load_dpEn', type=bool, default=False)
    parser.add_argument('--num_models', type=int, default=0)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--dp_outputs', type=int, default=1)
    parser.add_argument('--state_dim', type=int, default=6)
    parser.add_argument('--one_step', type=bool, default=False)
    parser.add_argument('--rand_models', type=int, default=None)
    # deep ensemble params
    parser.add_argument('--uq_method', type=str, default=None)
    parser.add_argument('--n_particles', type=int, default=None)
    parser.add_argument('--bayesian', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--model_lr', type=float, default=0.0001)
    parser.add_argument('--l2_reg_multiplier', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_nodes', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=3)

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]             

    train(params) 

    return                

if __name__ == '__main__':
    main()