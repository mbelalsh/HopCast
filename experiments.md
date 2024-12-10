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

## MHN 

We will train deterministic ensemble and then pick the first model from the ensemble to generate residuals for the MHN to learn. The residual file will be saved if it does not already exit. The results for deterministic ensembles will be generated at the same time and saved as `pkl` files. It would suffice to train just one deterministic model to generate residuals, but we will train an ensemble and generate results for deterministic ensembles as well.

### Glycolytic

#### sigma = 0.3
```
cd ode_models
##### Train deterministic ensemble (Base model)
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --num_layers 3
##### Test deterministic ensemble (saves residuals file as well)
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --load_dpEn True --rand_models 6 --num_layers 3
cd ..
cd attn_model
##### Train MHN (one for each state dimension of glycolytic)
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_0out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_1out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_2out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_3out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_4out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_5out.yml
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_6out.yml
##### Test MHN
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_0out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_1out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_2out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_3out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_4out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_5out.yml --load_mhn True
python main.py --yaml_file yml_files/glycolytic_3by10/model_params_glycolytic_3by10_6out.yml --load_mhn True
```

## Probabilistic Ensembles

These ensembels can be trained and then tested using several propagation methods. 

### Glycolytic

#### sigma = 0.3

```
cd ode_models
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --num_layers 3 --bayesian True ##### Train
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --num_layers 3 --bayesian True --load_dpEn True --rand_models 3 --uq_method trajectory_sampling  ##### Test Trajectory Sampling
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --num_layers 3 --bayesian True --load_dpEn True --rand_models 8 --uq_method moment_matching  ##### Test Moment Matching
python main.py --yaml_file yml_files/model_params_glycolytic_3by10.yml --num_layers 3 --bayesian True --load_dpEn True --rand_models 7 --uq_method expectation  ##### Test Expectation
```


