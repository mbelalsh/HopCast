import os
import random
import numpy as np
import pandas as pd
from pyro import sample
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
import sys, pickle

from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    GatedAdditiveTreeEnsembleConfig,
    MDNConfig
)
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
# from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.models.common.heads import LinearHeadConfig, MixtureDensityHeadConfig
np.random.seed(42)

def generate_non_linear_example(samples=int(1e5)):
    x_data = np.float32(np.random.uniform(-10, 10, (1, samples)))
    r_data = np.array([np.random.normal(scale=np.abs(i)) for i in x_data])
    y_data = np.float32(np.square(x_data)+r_data*2.0)

    x_data2 = np.float32(np.random.uniform(-10, 10, (1, samples)))
    r_data2 = np.array([np.random.normal(scale=np.abs(i)) for i in x_data2])
    y_data2 = np.float32(-np.square(x_data2)+r_data2*2.0)

    x_data = np.concatenate((x_data,x_data2),axis=1).T
    y_data = np.concatenate((y_data,y_data2),axis=1).T

    min_max_scaler = MinMaxScaler()
    y_data = min_max_scaler.fit_transform(y_data)

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.5, random_state=42, shuffle=True)
    x_test = np.linspace(-10,10,int(1e3))[:, np.newaxis].astype(np.float32)
    df_train = pd.DataFrame({"col1": x_train.ravel(), "target": y_train.ravel()})
    df_valid = pd.DataFrame({"col1": x_valid.ravel(), "target": y_valid.ravel()})
    # test = sorted(df_valid.col1.round(3).unique())
    # df_test = pd.DataFrame({"col1": test})
    df_test = pd.DataFrame({"col1": x_test.ravel()})
    return (df_train, df_valid, df_test, ["target"])

df_train, df_valid, df_test, target_col = generate_non_linear_example()
epochs = 200
batch_size = 2048
steps_per_epoch = int((len(df_train)//batch_size)*0.9)
data_config = DataConfig(
    target=['target'],
    continuous_cols=['col1'],
    categorical_cols=[],
#         continuous_feature_transform="quantile_uniform"
)
trainer_config = TrainerConfig(
    auto_lr_find=False, # Runs the LRFinder to automatically derive a learning rate
    batch_size=batch_size,
    max_epochs=epochs,
    early_stopping="valid_loss",
    early_stopping_patience=5,
    checkpoints="valid_loss"
)
optimizer_config = OptimizerConfig(lr_scheduler="ReduceLROnPlateau", lr_scheduler_params={"patience":3})

mdn_head_config = MixtureDensityHeadConfig(num_gaussian=2, weight_regularization=2, lambda_mu=10, lambda_pi=5).__dict__

backbone_config_class = "CategoryEmbeddingModelConfig"
backbone_config = dict(
    task="backbone",
    layers="128-64",  # Number of nodes in each layer
    activation="ReLU",  # Activation between each layers
    head=None,
)

model_config = MDNConfig(
    task="regression",
    backbone_config_class=backbone_config_class,
    backbone_config_params=backbone_config,
    head_config=mdn_head_config,
    learning_rate=1e-3,
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config
)

tabular_model.fit(train=df_train, validation=df_valid)
tabular_model.save_model(f"./lightning_logs/non_linear_{epochs}epc_{batch_size}bs_{100000}data", inference_only=True)

pred_df = tabular_model.predict(df_test, quantiles=[0.25,0.5,0.75], n_samples=100, ret_logits=True)
x_test = np.linspace(-10,10,int(1e3))[:, np.newaxis].astype(np.float32)
pred_df["col1"] = x_test.ravel()
pred_df.head()

df = df_valid.sample(10000)

fig = go.Figure([
    go.Scatter(
        name='Ground Truth',
        x=df['col1'],
        y=df['target'],
        mode='markers',
        line=dict(color='rgba(153, 115, 142, 0.2)'),
    ),
    go.Scatter(
        name='Component 1',
        x=pred_df['col1'],
        y=pred_df['mu_0'],
        mode='lines',
        line=dict(color='rgba(36, 37, 130, 1)'),
    ),
    go.Scatter(
        name='Component 2',
        x=pred_df['col1'],
        y=pred_df['mu_1'],
        mode='lines',
        line=dict(color='rgba(246, 76, 114, 1)'),
    ),
    go.Scatter(
        name='Upper Bound 1',
        x=pred_df['col1'],
        y=pred_df['mu_0']+pred_df['sigma_0'],
        mode='lines',
        marker=dict(color='rgba(47, 47, 162, 0.5)'),
#         line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound 1',
        x=pred_df['col1'],
        y=pred_df['mu_0']-pred_df['sigma_0'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(47, 47, 162, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name='Upper Bound 2',
        x=pred_df['col1'],
        y=pred_df['mu_1']+pred_df['sigma_1'],
        mode='lines',
        marker=dict(color='rgba(250, 152, 174, 0.5)'),
#         line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound 2',
        x=pred_df['col1'],
        y=pred_df['mu_1']-pred_df['sigma_1'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(250, 152, 174, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
])
fig.update_layout(
    yaxis_title='y',
    title='Mixture Density Network Prediction',
    hovermode="x"
)
# fig.show()
fig.write_image("./lightning_logs/prob_reg_mdn_2.png")