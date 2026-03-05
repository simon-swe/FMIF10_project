import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

cols = [
    "Entity",
    "Year",
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

feature_cols = [c for c in cols if c not in ("Entity", "Year", "Human Development Index")]

df = pd.read_csv("Dataset/filtered-data.csv", usecols=cols)
df = df[cols].dropna()
df = df.set_index(["Entity", "Year"])

print("Years in dataset:", sorted(df.index.get_level_values("Year").unique()))
print("Rows per country:\n", df.groupby("Entity").size())

for col in feature_cols:
    df[f"{col}_lag5"] = df.groupby("Entity")[col].shift(5)

df_lagged = df.dropna()
print(f"\nRows before lag dropna: {len(df)}")
print(f"Rows after lag dropna:  {len(df_lagged)}")

lag_features = " + ".join([f'Q("{c}_lag5")' for c in feature_cols])

formula_lagged = f"""
    Q("Human Development Index") ~
    1
    + {lag_features}
    + EntityEffects
"""

print("\n" + "=" * 60)
print("MODEL 1: Entity Fixed Effects with 5-year lags")
print("=" * 60)
model_lagged = PanelOLS.from_formula(formula_lagged, data=df_lagged)
result_lagged = model_lagged.fit(cov_type="robust") 

coef_df = pd.DataFrame({
    "Feature": result_lagged.params.index,
    "Coefficient": result_lagged.params.values,
    "Std Error": result_lagged.std_errors.values,
    "P-value": result_lagged.pvalues.values,
}).sort_values("Coefficient", ascending=False)
coef_df["Significant"] = coef_df["P-value"] < 0.05
print(coef_df.to_string(index=False))
print(f"\nR² (within):  {result_lagged.rsquared_within:.4f}")
print(f"R² (between): {result_lagged.rsquared_between:.4f}")
print(f"R² (overall): {result_lagged.rsquared_overall:.4f}")


# --- First differences ---
df_reset = df[feature_cols + ["Human Development Index"]].copy()
df_diff = df_reset.groupby("Entity").diff().dropna()

diff_features = " + ".join([f'Q("{c}")' for c in feature_cols])

formula_diff = f"""
    Q("Human Development Index") ~
    1
    + {diff_features}
    + EntityEffects
"""

print("\n" + "=" * 60)
print("MODEL 2: First Differences (year-over-year changes)")
print("=" * 60)
model_diff = PanelOLS.from_formula(formula_diff, data=df_diff)
result_diff = model_diff.fit(cov_type="robust")  # robust SE

coef_df2 = pd.DataFrame({
    "Feature": result_diff.params.index,
    "Coefficient": result_diff.params.values,
    "Std Error": result_diff.std_errors.values,
    "P-value": result_diff.pvalues.values,
}).sort_values("Coefficient", ascending=False)
coef_df2["Significant"] = coef_df2["P-value"] < 0.05
print(coef_df2.to_string(index=False))
print(f"\nR² (within):  {result_diff.rsquared_within:.4f}")
print(f"R² (between): {result_diff.rsquared_between:.4f}")
print(f"R² (overall): {result_diff.rsquared_overall:.4f}")