import numpy as np
from scipy.integrate import solve_ivp
import pickle
from utils import check_or_make_folder
import argparse

# Parameters (convert to seconds)
J0 = 2.5 
k1 = 100 
k2 = 6
k3 = 16 
k4 = 100
k5 = 1.28
k6 = 12
k = 1.8 
kappa = 13 
q = 4
K1 = 0.52
psi = 0.1
N = 1
A = 4

minutes = 4
t_span = (0, minutes)  # PICK DELTA T FROM SINDy PAPER
dt = 0.01  # 0.001 minutes = 0.06 seconds
t_eval = np.arange(t_span[0], t_span[1], dt) 


# Lotka-Volterra system of ODEs
def glycolytic_oscillator(t, S, J0=J0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5, k6=k6, k=k, kappa=kappa, q=q, K1=K1, psi=psi, N=N, A=A):
    S1, S2, S3, S4, S5, S6, S7 = S
    
    # The system of ODEs
    dS1dt = J0 - (k1 * S1 * S6) / (1 + (S6 / K1)**q)
    dS2dt = (2 * k1 * S1 * S6) / (1 + (S6 / K1)**q) - k2 * S2 * (N - S5) - k6 * S2 * S5
    dS3dt = k2 * S2 * (N - S5) - k3 * S3 * (A - S6)
    dS4dt = k3 * S3 * (A - S6) - k4 * S4 * S5 - kappa * (S4 - S7)
    dS5dt = k2 * S2 * (N - S5) - k4 * S4 * S5 - k6 * S2 * S5
    dS6dt = -(2 * k1 * S1 * S6) / (1 + (S6 / K1)**q) + 2 * k3 * S3 * (A - S6) - k5 * S6
    dS7dt = psi * kappa * (S4 - S7) - k * S7
    
    return [dS1dt, dS2dt, dS3dt, dS4dt, dS5dt, dS6dt, dS7dt]

# Function to generate multiple trajectories with both (x, y) and (xdot, ydot)
def generate_lv_trajectories_with_velocity(num_trajectories, initial_ranges, noise_std=0.1):
    print("Generating trajectories...")
    trajectories = []
    for _ in range(num_trajectories):
        # params and initial conditions from the
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821
        S0 = []
        for var in range(len(initial_ranges)):
           S0.append(np.random.uniform(*initial_ranges[var]))
        #S0 = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 1.0]
        
        # Solve the LV system to get the populations (x, y)
        sol = solve_ivp(glycolytic_oscillator, t_span, S0, t_eval=t_eval)
        
        # Get the populations (x, y)
        S1, S2, S3, S4, S5, S6, S7 = sol.y[0], sol.y[1], sol.y[2], sol.y[3], sol.y[4], sol.y[5], sol.y[6]
        
        # Store both positions and velocities
        trajectory = {
            'S1': S1[:, np.newaxis],
            'S2': S2[:, np.newaxis],
            'S3': S3[:, np.newaxis],
            'S4': S4[:, np.newaxis],
            'S5': S5[:, np.newaxis],
            'S6': S6[:, np.newaxis],
            'S7': S7[:, np.newaxis]
        }
        trajectories.append(trajectory)
        
    return trajectories

# Add feature-dependent noise
def add_feature_dependent_noise(trajectories, noise_factor=0.005):
    noisy_trajectories = []
    for traj in trajectories:
        noisy_traj = {}
        
        # For each feature, calculate the standard deviation (as a proxy for scale)
        for key in ['S1','S2','S3','S4','S5','S6','S7']:
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
    initial_ranges = [(0.15, 1.60),(0.19, 2.16),(0.04, 0.20),(0.10, 0.35),(0.08, 0.30),(0.14, 2.67),(0.05, 0.10)]  # Initial ranges for all 7 states
    num_trajectories = 750  # Number of trajectories to generate
    trajectories = generate_lv_trajectories_with_velocity(num_trajectories, initial_ranges)
    # Add feature-dependent noise
    noise_factor = params['noise_level']  # Proportional noise factor
    noisy_trajectories = add_feature_dependent_noise(trajectories, noise_factor=noise_factor)
    check_or_make_folder(f"./../ode_models/glycolytic/")
    data_name = {'0.05':'5by100','0.1':'1by10','0.3':'3by10'}

    pickle.dump(noisy_trajectories, open(f"./../ode_models/glycolytic/{num_trajectories}traj_t{t_span[0]}_to_t{t_span[1]}_{t_eval.shape[0]}ts_data_noisy_justState_{data_name[str(noise_factor)]}std.pkl", 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    data_gen(params) 

    return   

if __name__ == '__main__':
    main()