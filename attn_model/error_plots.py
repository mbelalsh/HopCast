import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_many(mu: np.ndarray, upper_mu: np.ndarray, lower_mu: np.ndarray,\
               ground_truth: np.ndarray, no_of_outputs:int, fig_name:str):

    no_of_inputs = np.linspace(0, stop=(mu.shape[1]-1), num=mu.shape[1], dtype=int)
    #labels = [r"$\Theta$[rad]", r"$\dot{\Theta}$[rad/s]", r"$x$[m]", r"$\dot{x}$[m/s]"]
    """
    labels = [r"height[m]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]", r"$\Theta$[rad]",\
              r"$v[m/s]$", r"$v[m/s]$", r"$\dot{\Theta}$[rad/s]", r"$\dot{\Theta}$[rad/s]",\
                r"$\dot{\Theta}$[rad/s]",  r"$\dot{\Theta}$[rad/s]"]
    """
    labels = [r"$x$", r"$y$", r"$z$", r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
    #labels = [r"$x$", r"$x$", r"$x$"]
    fig = plt.figure() 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

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

    for i in range(no_of_outputs): # mu: [particles,seq_len,outputs]
        #ax[i].fill_between(no_of_inputs, lower_mu[i,:].reshape(-1,), upper_mu[i,:].reshape(-1,), alpha=0.3)
        for j in range(mu.shape[0]):
            r1, = ax[i].plot(no_of_inputs, ground_truth[j,:,i].reshape(-1,), color=colors[j], marker='*', markersize=4)
            k1, = ax[i].plot(no_of_inputs, mu[j,:,i].reshape(-1,), color=colors[j], marker='*', markersize=6)
            ax[i].grid(True)
            ax[i].set_ylabel(labels[i])
        ax[i].spines['top'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5) # set_visible(False)
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)

    ax[no_of_outputs-1].set_xlabel('time [s]')
    #ax[0].legend((r1,k1), ('Ground Truth', 'Predictions'),\
    #              bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
    #                  borderaxespad=0, shadow=False)

    plt.savefig(fig_name, bbox_inches='tight')
    #plt.show()
    #plt.close()  

df = pd.read_csv("./../../my_folder/results/residuals_data/cs2_2017-01-01_2017-02-01_200step_10int.csv")
#df.head()

df['group'] = df.index // 201
grouped = df.groupby('group')
#df.iloc[195:205]
df

pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
group_arrs = []

for _, group in grouped:
    group_arr = group[pred_cols+gr_cols].values
    group_arrs.append(np.expand_dims(group_arr, axis=0))

data = np.concatenate(group_arrs, axis=0)
input_dim = len(pred_cols)
output_dim = len(pred_cols)

sim_data = data[:,:,:input_dim] 
gr_tr_data = data[:,:,input_dim:]

sim_input_data = sim_data[:,:-1,:]
sim_output_data = sim_data[:,1:,:]

gr_input_data = gr_tr_data[:,:-1,:]
gr_output_data = gr_tr_data[:,1:,:]

errors = gr_output_data - sim_output_data

start = 0

sim = errors[start:start+30,:,:]
gr = errors[start:start+30,:,:] 

plot_many(sim, None, None, gr, sim.shape[-1], f"./firsterror_same_seq_{start}.png")

