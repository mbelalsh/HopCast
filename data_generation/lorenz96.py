import numpy as np
from scipy.integrate import solve_ivp
import pickle
import argparse
from utils import check_or_make_folder

# Parameters for Lorenz 96
F = 8  # Forcing constant

# Lorenz 96 system of ODEs
def lorenz96(t, X, F=F):
    # X is a vector of the 5 state variables: [X1, X2, X3, X4, X5]
    dX = np.zeros(5)
    
    # The Lorenz 96 equations, cyclic boundary conditions (modulo arithmetic)
    for i in range(5):
        dX[i] = (X[(i+1) % 5] - X[(i-2) % 5]) * X[(i-1) % 5] - X[i] + F
        
    return dX

# Time span and evaluation points (same as glycolytic oscillator)
minutes = 3
t_span = (0, minutes)
dt = 0.01  # 0.01 minutes = 0.6 seconds
t_eval = np.arange(t_span[0], t_span[1], dt)

# Function to generate multiple Lorenz 96 trajectories
def generate_lorenz96_trajectories_with_velocity(num_trajectories, initial_ranges, noise_std=0.1):
    trajectories = []
    for _ in range(num_trajectories):
        # Random initial conditions for the 5 variables
        X0 = [np.random.uniform(*initial_ranges[var]) for var in range(len(initial_ranges))]
        
        # Solve the Lorenz 96 system to get the trajectories
        sol = solve_ivp(lorenz96, t_span, X0, t_eval=t_eval)
        
        # Get the 5 state variables over time
        X1, X2, X3, X4, X5 = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4]
        
        # Store the positions (trajectories) in a dictionary
        trajectory = {
            'X1': X1[:, np.newaxis],
            'X2': X2[:, np.newaxis],
            'X3': X3[:, np.newaxis],
            'X4': X4[:, np.newaxis],
            'X5': X5[:, np.newaxis]
        }
        trajectories.append(trajectory)
    
    return trajectories

# Add feature-dependent noise (same method as in the glycolytic oscillator case)
def add_feature_dependent_noise(trajectories, noise_factor=0.05):
    noisy_trajectories = []
    for traj in trajectories:
        noisy_traj = {}
        
        # For each feature (state variable), calculate standard deviation and add proportional noise
        for key in ['X1', 'X2', 'X3', 'X4', 'X5']:
            feature = traj[key]
            feature_std = np.std(feature)  # Scale (std deviation)
            
            # Add noise proportional to the feature's standard deviation
            noise = np.random.normal(0, noise_factor * feature_std, size=feature.shape)
            noisy_traj[key] = feature + noise
        
        noisy_trajectories.append(noisy_traj)
    
    return noisy_trajectories

def data_gen(params):
    np.random.seed(0)
    # Generate Lorenz 96 trajectories
    lower_s, upper_s = -10.5, 10.5
    initial_ranges = [(lower_s, upper_s) for i in range(5)]  # Initial ranges for all 5 states
    num_trajectories = 666  # Number of trajectories to generate
    trajectories = generate_lorenz96_trajectories_with_velocity(num_trajectories, initial_ranges)
    # Add feature-dependent noise to the trajectories
    noise_factor = params['noise_level']   # Proportional noise factor
    noisy_trajectories = add_feature_dependent_noise(trajectories, noise_factor=noise_factor)
    check_or_make_folder(f"./../ode_models/lorenz96/")
    data_name = {'0.05':'5by100','0.1':'1by10','0.3':'3by10'}

    pickle.dump(noisy_trajectories, open(f"./../ode_models/lorenz96/{num_trajectories}traj_t{t_span[0]}_to_t{t_span[1]}_{t_eval.shape[0]}ts_s{int(10*lower_s)}by10_to_s{int(10*upper_s)}by10_data_noisy_justState_{data_name[str(noise_factor)]}std.pkl", 'wb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    data_gen(params) 

    return   

if __name__ == '__main__':
    main()