from argparse import ArgumentParser
import logging
import pickle
import yaml
from typing import List, Dict, Tuple
from utils import check_or_make_folder

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from mdn import MixtureDensityNetwork, NoiseType, prepare_loaders, prepare_val_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(params: Dict):
    dataset_name = params['data_path'].split("/")[-1].split(".")[0]
    params['n_samples'] = dataset_name.split('_')[1]
    
    data_dict = pickle.load(open(params['data_path'], 'rb'))
    
    x, y = data_dict['x_data'],data_dict['y_data']
    train_loader, val_loader, x_val, y_val = prepare_loaders(x, y, params)
    hidden = tuple(params['nodes'] for _ in range(params['layers']))
    model = MixtureDensityNetwork(1, 1, n_components=params['n_components'],\
                                   hidden_dim=hidden, noise_type=NoiseType.DIAGONAL)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['l2_reg'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params["n_epochs"])

    for i in range(params["n_epochs"]):

        train_losses = []
        model.train()   
        for x_, y_ in train_loader:
            x_, y_ = x_.to(device),y_.to(device)
            optimizer.zero_grad()
            loss = model.loss(x_, y_).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.data)

        if i % 3 == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for x_, y_ in val_loader:
                    x_, y_ = x_.to(device),y_.to(device)
                    loss = model.loss(x_, y_)
                    val_losses.append(loss.mean())

        logger.info(f"Iter: {i}\t" + f"Train Loss: {torch.Tensor(train_losses).mean():.2f}\t" + f"Val Loss: {torch.Tensor(val_losses).mean():.2f}")

    save_dict = {'weights': model.state_dict(),
                'x_true': x_val,
                'y_true': y_val}
    save_dir = f"./mdn_data/{dataset_name}_{params['batch_size']}bs_{params['lr']}lr_{params['l2_reg']}l2_{params['n_epochs']}epc_{params['layers']}layers_{params['nodes']}nodes_{params['n_components']}comp"
    check_or_make_folder(save_dir)
    pickle.dump(save_dict, open(save_dir + f"/train_ckpts.pkl", "wb"))

def test(params: Dict):
    dataset_name = params['data_path'].split("/")[-1].split(".")[0]

    hidden = tuple(params['nodes'] for _ in range(params['layers']))
    model = MixtureDensityNetwork(1, 1, n_components=params['n_components'],\
                                   hidden_dim=hidden, noise_type=NoiseType.DIAGONAL)
    model.to(device)
    if args.is_optimal:
        opt_type = "opt"
    else:
        opt_type = "subopt"    
    save_dir = f"./mdn_data/{dataset_name}_{params['batch_size']}bs_{params['lr']}lr_{params['l2_reg']}l2_{params['n_epochs']}epc_{params['layers']}layers_{params['nodes']}nodes_{params['n_components']}comp"
    save_dict = pickle.load(open(save_dir + f"/train_ckpts.pkl", "rb"))
    model.load_state_dict(save_dict['weights'])
    model.eval()

    x_val, y_val = save_dict['x_true'], save_dict['y_true']
    # Val Loader
    val_loader = prepare_val_loader(x_val, y_val)

    model.eval()
    with torch.no_grad():
        for batch_id, _data in enumerate(val_loader):
            x_, y_ = _data
            x_, y_ = x_.to(device), y_.to(device)
            
            y_hat, log_pi, mu, sigma = model.sample(x_)

            data_dict = {'y_hat': y_hat.detach().cpu().numpy(), 'log_pi': log_pi.detach().cpu().numpy(), 
                            'mu': mu.detach().cpu().numpy(), 'sigma': sigma.detach().cpu().numpy(),
                            'x_true':x_.detach().cpu().numpy(), 'y_true':y_.detach().cpu().numpy()}
            pickle.dump(data_dict, open(save_dir + f"/infer_ckpts_{batch_id}.pkl", 'wb'))

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--data_path", type=str, default=None)
    argparser.add_argument('--yaml_file', type=str, default=None)
    argparser.add_argument("--train", type=bool, default=False)
    argparser.add_argument("--is_optimal", type=bool, default=False)
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--lr", type=float, default=0.005)
    argparser.add_argument("--l2_reg", type=float, default=0.005)
    argparser.add_argument("--n_epochs", type=int, default=200)
    argparser.add_argument("--n_components", type=int, default=3)
    argparser.add_argument("--layers", type=int, default=2)
    argparser.add_argument("--nodes", type=int, default=100)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]             

    if params["train"]:
        train(params)
    else:
        test(params)    


