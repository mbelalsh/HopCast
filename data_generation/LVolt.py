import numpy as np
from scipy.integrate import solve_ivp
import pickle
from utils import check_or_make_folder
import argparse

# LV system parameters
alpha = 1.1 #1.1 1.5
beta = 0.4 #0.4 1
gamma = 0.4 #0.4 3
delta = 0.1 #0.1 1

# Lotka-Volterra system of ODEs
def lotka_volterra(t, state, alpha=alpha, beta=beta, gamma=gamma, delta=delta):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Time points for the simulation
t_span = (0, 30) # TODO: DECIDE DELT T CAREFULLY AND SEE WHAT WE HAVE IN LORENZ
#ts = 300  
#t_eval = np.linspace(t_span[0], t_span[1], ts)
dt = 0.1  # 0.001 minutes = 0.06 seconds
t_eval = np.arange(t_span[0], t_span[1], dt)

# Function to generate multiple trajectories with both (x, y) and (xdot, ydot)
def generate_lv_trajectories_with_velocity(num_trajectories, initial_range_x, initial_range_y, noise_std=0.1):
    trajectories = []
    for _ in range(num_trajectories):
        # Random initial condition within a specific range
        x0 = np.random.uniform(*initial_range_x)
        y0 = np.random.uniform(*initial_range_y)
        
        # Solve the LV system to get the populations (x, y)
        sol = solve_ivp(lotka_volterra, t_span, [x0, y0], t_eval=t_eval)
        
        # Get the populations (x, y)
        x, y = sol.y[0], sol.y[1]
        
        # Calculate the velocities (xdot, ydot) using the LV equations
        xdot = alpha * x - beta * x * y
        ydot = delta * x * y - gamma * y
        
        # Store both positions and velocities
        trajectory = {
            'x': x[:, np.newaxis],
            'y': y[:, np.newaxis],
            'xdot': xdot[:, np.newaxis],
            'ydot': ydot[:, np.newaxis]
        }
        trajectories.append(trajectory)
        
    return trajectories

# Add feature-dependent noise
def add_feature_dependent_noise(trajectories, noise_factor=0.05):
    noisy_trajectories = []
    for traj in trajectories:
        noisy_traj = {}
        
        # For each feature, calculate the standard deviation (as a proxy for scale)
        for key in ['x', 'y', 'xdot', 'ydot']:
            feature = traj[key]
            feature_std = np.std(feature)  # Scale (std deviation)
            
            # Add noise proportional to the feature's standard deviation
            noise = np.random.normal(0, noise_factor * feature_std, size=feature.shape)
            noisy_traj[key] = feature + noise
        
        noisy_trajectories.append(noisy_traj)
    
    return noisy_trajectories

def data_gen(params):

    np.random.seed(0)
    # Generate LV trajectories
    initial_range_x = (5, 20)  # Initial range for x and y
    initial_range_y = (5, 10)  # Initial range for x and y
    num_trajectories = 500  # Number of trajectories to generate
    trajectories = generate_lv_trajectories_with_velocity(num_trajectories, initial_range_x, initial_range_y)
    # Add feature-dependent noise
    noise_factor = params['noise_level']  # Proportional noise factor
    noisy_trajectories = add_feature_dependent_noise(trajectories, noise_factor=noise_factor)
    check_or_make_folder(f"./../ode_models/LVolt/")
    data_name = {'0.05':'5by100','0.1':'1by10','0.3':'3by10'}    

    pickle.dump(noisy_trajectories, open(f"./../ode_models/LVolt/{num_trajectories}traj_{alpha}a_{beta}b_{gamma}g_{delta}d_t{t_span[0]}_to_t{t_span[1]}_{t_eval.shape[0]}ts_data_noisy_{data_name[str(noise_factor)]}std.pkl", 'wb'))   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    data_gen(params) 

    return   

if __name__ == '__main__':
    main()