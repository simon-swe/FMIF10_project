import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score

cols = [
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
    "Human Development Index",
]

df = pd.read_csv("Dataset/filtered-data.csv", usecols=cols)
df = df[cols] 

df = pd.read_csv("Dataset/filtered-data.csv", usecols=cols)
df = df[cols]

feature_cols = cols[:-1]
normed = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
X_exp = np.exp(normed).to_numpy(dtype=np.float64)
y = df.iloc[:, -1].to_numpy(dtype=np.float64)
feature_names = df.columns[:-1]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores, r2_scores = [], []

for train_idx, test_idx in kf.split(X_exp):
    X_train, X_test = X_exp[train_idx], X_exp[test_idx]  # use X_exp
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()                        # fit only on train
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse_scores.append(root_mean_squared_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))

print(f"Average RMSE : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average R²   : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Final model
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X_exp)

final_model = LinearRegression()
final_model.fit(X_scaled, y)

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": final_model.coef_
}).sort_values("Coefficient", ascending=False)

print(f"\nIntercept: {final_model.intercept_:.4f}")
print(coefficients.to_string(index=False))