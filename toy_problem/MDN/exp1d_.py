from argparse import ArgumentParser
import logging
import pickle

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from mdn import MixtureDensityNetwork, NoiseType

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_data(n=512):
    y = np.linspace(-1, 1, n)
    x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
    return x[:,np.newaxis], y[:,np.newaxis]

def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-8, 8)
    plt.ylim(-1, 1)
    plt.axis('off')

def train():
    x, y = gen_data()
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    model = MixtureDensityNetwork(1, 1, n_components=3, hidden_dim=50, noise_type=NoiseType.DIAGONAL)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iterations)

    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(x, y).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    save_dict = {'weights': model.state_dict(),
                'X': x,
                'Y': y}
    #pickle.dump(save_dict, open("./mdn_data/exp1d.pkl", "wb"))

def test(args):

    model = MixtureDensityNetwork(1, 1, n_components=3, hidden_dim=50, noise_type=NoiseType.DIAGONAL)

    if args.is_optimal:
        opt_type = "opt"
    else:
        opt_type = "subopt"    

    save_dict = pickle.load(open(f"./mdn_data/exp1d_{opt_type}.pkl", "rb"))
    model.load_state_dict(save_dict['weights'])
    model.eval()

    x, y = save_dict['X'], save_dict['Y']

    with torch.no_grad():
        y_hat, log_pi, mu, sigma = model.sample(x)
        data_dict = {'y_hat': y_hat, 'log_pi': log_pi, 
                     'mu': mu,'sigma': sigma,
                     'x_true':x, 'y_true':y}
        pickle.dump(data_dict, open(f"./mdn_data/exp1d_infer_{opt_type}.pkl", 'wb'))

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plot_data(x[:, 0].numpy(), y[:, 0].numpy())
    plt.title("Observed data")
    plt.subplot(1, 2, 2)
    plot_data(x[:, 0].numpy(), y_hat[:, 0].numpy())
    plt.title("Sampled data")
    #plt.show()
    #plt.savefig(f"./mdn_data/exp1d_{opt_type}.png")

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=2000)
    argparser.add_argument("--train", type=bool, default=False)
    argparser.add_argument("--is_optimal", type=bool, default=False)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.train:
        train()
    else:
        test(args)
