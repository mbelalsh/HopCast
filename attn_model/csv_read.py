import pandas as pd 
import numpy as np
import sys

df = pd.read_csv("./../../my_folder/results/residuals_data/cs2_2017-01-01_2019-01-01_200step_200int.csv")

df['group'] = df.index // 201
grouped = df.groupby('group')
#df.iloc[195:205]
df

pred_cols = ['x_pred','y_pred','z_pred','x_dot_pred','y_dot_pred','z_dot_pred']
gr_cols = ['x_gr','y_gr','z_gr','x_dot_gr','y_dot_gr','z_dot_gr']
group_arrs = []
incomplete_groups = []

expected_horizon = set(range(201))

# Sliding window to create groups of 201 rows
for start_idx in range(0, len(df) - 200, 201):
    group = df.iloc[start_idx:start_idx + 201]
    horizon_values = set(group['horizon'].values)

    if horizon_values != expected_horizon:
        print(horizon_values)
        incomplete_groups.append((start_idx, start_idx + 201, sorted(horizon_values)))
        print(start_idx)
        break
    else:
        group_arr = group[pred_cols + gr_cols].values
        group_arrs.append(np.expand_dims(group_arr, axis=0))

#for _, group in grouped:
#    group_arr = group[pred_cols+gr_cols].values
#    group_arrs.append(np.expand_dims(group_arr, axis=0))

data = np.concatenate(group_arrs, axis=0)