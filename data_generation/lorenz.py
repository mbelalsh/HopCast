import numpy as np
from scipy.integrate import solve_ivp
import pickle
from utils import check_or_make_folder
import argparse

# Lorenz system equations
np.random.seed(0)
sigma = 10
rho = 28
beta = 8/3

def lorenz(t, state, sigma=sigma, rho=rho, beta=beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Time points for the simulation
t_span = (0, 3)  # From t=0 to t=40
ts = 300
t_eval = np.linspace(t_span[0], t_span[1], ts)  # 10000 time points for detailed trajectory

# Function to generate multiple trajectories with both (x,y,z) and (xdot, ydot, zdot)
def generate_trajectories_with_velocity(num_trajectories, initial_range, noise_std = 0.1):
    trajectories = []
    for _ in range(num_trajectories):
        # Random initial condition within a specific range
        x0 = np.random.uniform(*initial_range)
        y0 = np.random.uniform(*initial_range)
        z0 = np.random.uniform(0, 50)  # Typical range for z
        
        # Solve the Lorenz system to get the positions (x, y, z)
        sol = solve_ivp(lorenz, t_span, [x0, y0, z0], t_eval=t_eval)
        
        # Get the positions (x, y, z)
        x, y, z = sol.y[0], sol.y[1], sol.y[2]
        
        # Calculate the velocities (xdot, ydot, zdot) using the Lorenz equations
        xdot = sigma * (y - x)
        ydot = x * (rho - z) - y
        zdot = x * y - (beta) * z

        # Add noise to the positions and velocities
        #x += np.random.normal(0, noise_std, size=x.shape)
        #y += np.random.normal(0, noise_std, size=y.shape)
        #z += np.random.normal(0, noise_std, size=z.shape)
        #xdot += np.random.normal(0, noise_std, size=xdot.shape)
        #ydot += np.random.normal(0, noise_std, size=ydot.shape)
        #zdot += np.random.normal(0, noise_std, size=zdot.shape)
        
        # Store both positions and velocities
        trajectory = {
            'x': x[:,np.newaxis],
            'y': y[:,np.newaxis],
            'z': z[:,np.newaxis],
            'xdot': xdot[:,np.newaxis],
            'ydot': ydot[:,np.newaxis],
            'zdot': zdot[:,np.newaxis]
        }
        trajectories.append(trajectory)
        
    return trajectories

# Add feature-dependent noise
def add_feature_dependent_noise(trajectories, noise_factor=0.1):
    noisy_trajectories = []
    for traj in trajectories:
        noisy_traj = {}
        
        # For each feature, calculate the standard deviation (as a proxy for scale)
        for key in ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']:
            feature = traj[key]
            feature_std = np.std(feature)  # Scale (std deviation)
            
            # Add noise proportional to the feature's standard deviation
            noise = np.random.normal(0, noise_factor * feature_std, size=feature.shape)
            noisy_traj[key] = feature + noise
        
        noisy_trajectories.append(noisy_traj)
    
    return noisy_trajectories

def data_gen(params):
    # Generate and plot 2 trajectories as an example
    initial_range = (-20, 20)
    num_trajectories = 1000
    trajectories = generate_trajectories_with_velocity(num_trajectories, initial_range)
    # Add feature-dependent noise
    noise_factor = params['noise_level']  # Proportional noise factor
    noisy_trajectories = add_feature_dependent_noise(trajectories, noise_factor=noise_factor)    
    check_or_make_folder(f"./../ode_models/lorenz/")
    data_name = {'0.05':'5by100','0.1':'1by10','0.3':'3by10'}

    pickle.dump(noisy_trajectories, open(f"./../ode_models/lorenz/{num_trajectories}traj_t{t_span[0]}_to_t{t_span[1]}_10sig_28rho_8by3beta_{ts}ts_data_noisy_{data_name[str(noise_factor)]}std.pkl", 'wb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    data_gen(params) 

    return   

if __name__ == '__main__':
    main()