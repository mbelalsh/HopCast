## Data Generation
```
##### Lorenz
python data_generation/lorenz.py --noise_level 0.05
python data_generation/lorenz.py --noise_level 0.1
python data_generation/lorenz.py --noise_level 0.3
##### FitzHugh–Nagumo 
python data_generation/FHNag.py --noise_level 0.05
python data_generation/FHNag.py --noise_level 0.1
python data_generation/FHNag.py --noise_level 0.3
##### Lotka–Volterra
python data_generation/LVolt.py --noise_level 0.05
python data_generation/LVolt.py --noise_level 0.1
python data_generation/LVolt.py --noise_level 0.3
##### Lorenz95
python data_generation/lorenz96.py --noise_level 0.05
python data_generation/lorenz96.py --noise_level 0.1
python data_generation/lorenz96.py --noise_level 0.3
##### Glycolytic
python data_generation/glycolytic.py --noise_level 0.05
python data_generation/glycolytic.py --noise_level 0.1
python data_generation/glycolytic.py --noise_level 0.3
```

## Deterministic Ensemble

We will train deterministic ensemble and then pick the first model from the ensemble to generate residuals for the MHN to learn. The residual file will be saved if it does not already exit.

### Glycolytic
#### MHN

```
cd ode_models
##### Train deterministic ensemble (Base model)
python main.py --yaml_file model_params.yml --data_path "./glycolytic/750traj_t0_to_t4_400ts_data_noisy_justState_3by10std.pkl" --num_layers 3
##### Test deterministic ensemble (save residuals file)
python main.py --yaml_file model_params.yml --data_path "./glycolytic/750traj_t0_to_t4_400ts_data_noisy_justState_3by10std.pkl" --load_dpEn True --rand_models 6 --num_layers 3
cd ..
cd attn_model
##### Train MHN sigma = 0.3
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_0out.yml
##### Test MHN sigma = 0.3
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_0out.yml --load_mhn True
```


