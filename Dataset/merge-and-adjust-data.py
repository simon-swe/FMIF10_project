import pandas as pd

# Read main data
df_main = pd.read_csv('ghg.csv')

# Read HDI data and clean it
df_hdi = pd.read_csv('hdi.csv')
df_hdi_clean = df_hdi[['Entity', 'Year', 'Human Development Index']]

# Read population data
df_pop = pd.read_csv('pop.csv')
# Keep only necessary columns and rename for easier use
df_pop_clean = df_pop[['Entity', 'Year', 'Population, total']].rename(
    columns={'Population, total': 'Population'}
)

# Merge HDI first (using right join to keep all HDI data)
df_merged = pd.merge(df_main, df_hdi_clean, on=['Entity', 'Year'], how='right')

# Now merge population data
df_with_pop = pd.merge(df_merged, df_pop_clean, on=['Entity', 'Year'], how='right')

# List of emission columns to convert to per capita
emission_columns = [
    'Agriculture',
    'Land-use change and forestry', 
    'Waste',
    'Buildings',
    'Industry',
    'Manufacturing and construction',
    'Transport',
    'Electricity and heat',
    'Fugitive emissions',
    'Other fuel combustion',
    'Aviation and shipping'
]

# Create per capita versions of each emission column
per_capita_columns = []
for col in emission_columns:
    if col in df_with_pop.columns:
        per_capita_col = f"{col} per capita"
        df_with_pop[per_capita_col] = df_with_pop[col] / df_with_pop['Population']
        per_capita_columns.append(per_capita_col)

# Create total emissions per capita
if all(col in df_with_pop.columns for col in emission_columns):
    df_with_pop['Total emissions per capita'] = df_with_pop[emission_columns].sum(axis=1) / df_with_pop['Population']
    per_capita_columns.append('Total emissions per capita')

# ===== NEW: Keep only the columns we want =====
# Option A: Select specific columns to keep
columns_to_keep = [
    'Entity', 
    'Code',           # if you want to keep the country code
    'Year', 
    'Human Development Index', 
    'Population'
] + per_capita_columns

# Make sure all columns exist before keeping them
columns_to_keep = [col for col in columns_to_keep if col in df_with_pop.columns]

# Keep only the selected columns
df_final = df_with_pop[columns_to_keep]

# Save result
df_final.to_csv('ghg-hdi-pop.csv', index=False)

print("Merged HDI and population data onto GHG data")
print("Created per-capita emission columns using 'Population, total'")

# Display basic info
print(f"\nTotal rows: {len(df_final)}")
print(f"Columns: {df_final.columns.tolist()}")
print(f"\nSample of first 5 rows:")
print(df_final[['Entity', 'Year', 'Human Development Index', 'Population'] + 
                  [f"{col} per capita" for col in emission_columns[:3] if f"{col} per capita" in df_final.columns]].head())