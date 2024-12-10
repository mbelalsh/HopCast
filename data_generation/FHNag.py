import numpy as np
from scipy.integrate import solve_ivp
from utils import check_or_make_folder
import argparse
import pickle


# Parameters for FitzHugh-Nagumo model
a = 0.7
b = 0.8
epsilon = 0.08
I = 0.5  # External current input

# FitzHugh-Nagumo system of ODEs
def fitzhugh_nagumo(t, X, a=a, b=b, epsilon=epsilon, I=I):
    v, w = X
    dvdt = v - (v**3) / 3 - w + I
    dwdt = epsilon * (v + a - b * w)
    return [dvdt, dwdt]

# Time span and evaluation points (same as the previous system)
minutes = 200
t_span = (0, minutes)
dt = 0.5  # 0.01 minutes = 0.6 seconds
t_eval = np.arange(t_span[0], t_span[1], dt)

# Function to generate multiple FitzHugh-Nagumo trajectories
def generate_fhn_trajectories_with_velocity(num_trajectories, initial_ranges, noise_std=0.1):
    trajectories = []
    for _ in range(num_trajectories):
        # Random initial conditions for the 2 variables (v and w)
        X0 = [np.random.uniform(*initial_ranges[var]) for var in range(len(initial_ranges))]
        
        # Solve the FitzHugh-Nagumo system to get the trajectories
        sol = solve_ivp(fitzhugh_nagumo, t_span, X0, t_eval=t_eval)
        
        # Get the state variables (v and w) over time
        v, w = sol.y[0], sol.y[1]
        
        # Store the positions (trajectories) in a dictionary
        trajectory = {
            'v': v[:, np.newaxis],
            'w': w[:, np.newaxis]
        }
        trajectories.append(trajectory)
    
    return trajectories

# Add feature-dependent noise
def add_feature_dependent_noise(trajectories, noise_factor=0.05):
    noisy_trajectories = []
    for traj in trajectories:
        noisy_traj = {}
        
        # For each feature (state variable), calculate standard deviation and add proportional noise
        for key in ['v', 'w']:
            feature = traj[key]
            feature_std = np.std(feature)  # Scale (std deviation)
            
            # Add noise proportional to the feature's standard deviation
            noise = np.random.normal(0, noise_factor * feature_std, size=feature.shape)
            noisy_traj[key] = feature + noise
        
        noisy_trajectories.append(noisy_traj)
    
    return noisy_trajectories

def data_gen(params):
    np.random.seed(0)
    # Generate FitzHugh-Nagumo trajectories
    lower_s, upper_s = -1.5, 1.5  # Adjusted initial ranges for the FHN system
    initial_ranges = [(lower_s, upper_s) for i in range(2)]  # Initial ranges for both states
    num_trajectories = 350  # Number of trajectories to generate
    trajectories = generate_fhn_trajectories_with_velocity(num_trajectories, initial_ranges)
    # Add feature-dependent noise to the trajectories
    noise_factor = params['noise_level']  # Proportional noise factor
    noisy_trajectories = add_feature_dependent_noise(trajectories, noise_factor=noise_factor)
    check_or_make_folder(f"./../ode_models/FHNag/")
    data_name = {'0.05':'5by100','0.1':'1by10','0.3':'3by10'}    

    pickle.dump(noisy_trajectories, open(f"./../ode_models/FHNag/{num_trajectories}traj_t{t_span[0]}_to_t{t_span[1]}_{t_eval.shape[0]}ts_s{int(10*lower_s)}by10_to_s{int(10*upper_s)}by10_data_noisy_justState_{data_name[str(noise_factor)]}std.pkl", 'wb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    data_gen(params) 

    return   

if __name__ == '__main__':
    main()