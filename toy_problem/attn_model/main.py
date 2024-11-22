import argparse
import yaml
from typing import Dict, Tuple
from prepare_data import DataProcessor
from train_hopfield_same_enc import TrainHopfieldSameEnc
import torch
import os
from utils import check_or_make_folder, set_seed
import sys

def train(params: Dict):

    set_seed(params)
    # change directory to save model/data
    check_or_make_folder("./attn_data")
    os.chdir("./attn_data")
    torch.cuda.empty_cache()
    # prepare data and train attention model
    params['dataset_name'] = params['data_path'].split("/")[-1].split(".")[0]
    data_proc = DataProcessor(params)
    calib_data = data_proc.get_data() # update params dict

    train_hopfield = TrainHopfieldSameEnc(data_proc.params)
    if not params['load_mhn']:
        calib_data = data_proc.normalize_data(calib_data)
        calib_data_all = data_proc.mix_data(calib_data)
        m_calib_data = data_proc.pack_rand_data(calib_data_all)
        train_hopfield.all_train_hopfield(m_calib_data)
    else:
        # load calib_data and model weights from the file
        train_hopfield.all_eval_hopfield()

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--yaml_file', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--load_mhn', type=bool, default=False)
    parser.add_argument('--num_mhn_models', type=int, default=0)
    parser.add_argument('--state_dim', type=int, default=6)
    # MHN params
    parser.add_argument('--mhn_epochs', type=int, default=None)
    parser.add_argument('--mhn_lr', type=float, default=0.0001)
    parser.add_argument('--mhn_l2_reg', type=float, default=0.0001)
    parser.add_argument('--mhn_batch_size', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--use_base_enc', type=bool, default=False)
    parser.add_argument('--ctx_enc_out', type=int, default=None)
    # calibration option
    parser.add_argument('--seq_len', type=int, default=None)
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
    parser.add_argument('--pos_enc', type=bool, default=False)
    parser.add_argument('--ood_pred', type=bool, default=False)

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