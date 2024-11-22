import argparse
import yaml
from typing import Dict, Tuple
from prepare_data import DataProcessor
from train_hopfield import TrainHopfield
from train_hopfield_same_enc import TrainHopfieldSameEnc
from train_hopfield_one_enc import TrainHopfieldOneEnc
from utils import set_seed
import torch
import os, pickle
from utils import check_or_make_folder, set_seed
import sys

def train(params: Dict):

    set_seed(params)
    # change directory to save model/data
    if params['temp_scale']:
        check_or_make_folder("./attn_data_temp_scale")
        os.chdir("./attn_data_temp_scale")
    else:    
        check_or_make_folder("./attn_data")
        os.chdir("./attn_data")
    torch.cuda.empty_cache()
    # prepare data and train attention model
    params['dataset_name'] = params['data_path'].split("/")[-1].split(".")[0]
    data_proc = DataProcessor(params)
    data_loaded = data_proc.get_data() # update params dict
    calib_data = data_proc.data_tuples(data_loaded) # update params dict
    # putting the following two lines here to run at inference time because we need input_filter
    # in params dict to save memory data for notebook. If not needed, remove the lines from here
    # to run only at training time only.
    calib_data = data_proc.build_context(calib_data, just_ts_ctxt=params["just_ts_ctxt"])
    calib_data = data_proc.normalize_calib_data(calib_data)    

    train_hopfield = TrainHopfieldSameEnc(data_proc.params)
    if not params['load_mhn']:
        #calib_data = data_proc.build_context(calib_data, just_ts_ctxt=params["just_ts_ctxt"])
        #calib_data = data_proc.normalize_calib_data(calib_data)

        train_hopfield.all_train_hopfield(calib_data)
    else:    
        # load calib_data and model weights from the file
        train_hopfield.all_eval_hopfield()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--yaml_file', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--load_mhn', type=bool, default=False)
    parser.add_argument('--num_mhn_models', type=int, default=0)
    parser.add_argument('--state_dim', type=int, default=6)
    parser.add_argument('--data_type', type=str, default=None)
    parser.add_argument('--out', type=int, default=None)
    parser.add_argument('--ode_name', type=str, default=None)
    parser.add_argument('--temp_scale', type=bool, default=False)
    parser.add_argument('--memory_dim', type=float, default=None)
    # MHN params
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--mhn_epochs', type=int, default=None)
    parser.add_argument('--mhn_lr', type=float, default=0.0001)
    parser.add_argument('--mhn_l2_reg', type=float, default=0.0001)
    parser.add_argument('--mhn_batch_size', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--use_base_enc', type=bool, default=False)
    parser.add_argument('--ctx_enc_out', type=int, default=None)
    # calibration option
    parser.add_argument('--mhn_output', type=str, choices=['y','delta'])
    parser.add_argument('--calib_horizon', type=int, default=None)
    parser.add_argument('--cp_sampling', type=Tuple, choices=[['topk',10],['sampling',20]])
    parser.add_argument('--cp_aggregate', type=str, choices=['long_seq','avg'])
    parser.add_argument('--cp_alphas', type=int, choices='any integer between 1 and less than len(_alphas_list) in ConformHopfield')
    parser.add_argument('--cp_replacement', type=bool, default=False)
    # context options
    parser.add_argument('--past_ts_ctxt', type=int, default=0) # How many timesteps ID as context
    parser.add_argument('--past_feat_ctxt', type=int, default=1) # How many features as context
    parser.add_argument('--past_pred_ctxt', type=int, default=0) # How many features as context
    parser.add_argument('--just_ts_ctxt', type=bool, default=False)
    parser.add_argument('--init_cond_ctxt', type=bool, default=False)
    parser.add_argument('--mirrorless_enc', type=bool, default=False)
    parser.add_argument('--pos_enc', type=bool, default=False)

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