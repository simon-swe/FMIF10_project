from pysr import PySRRegressor
import pandas as pd
import numpy as np
import re
import os

cols = [
    "Human Development Index",
    "Population",
    "Agriculture per capita",
    "Land-use change and forestry per capita",
    "Waste per capita",
    "Buildings per capita",
    "Industry per capita",
    "Manufacturing and construction per capita",
    "Transport per capita",
    "Electricity and heat per capita",
    "Fugitive emissions per capita", 
]

df = pd.read_csv("Dataset/filtered-data.csv", usecols=cols)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
y = df.iloc[:, -1].to_numpy(dtype=np.float64)

variable_names = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in cols[:-1]]

model = PySRRegressor(
    procs=4,  # 
    timeout_in_seconds=60,
    niterations=10,    
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
    ),

    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "exp",
        "log",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    progress=True,
    turbo=True,
)

model.fit(X, y, variable_names=variable_names)
print(model)