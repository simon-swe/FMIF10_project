import pandas as pd
import numpy as np

# Read your data
df = pd.read_csv('ghg-hdi-pop.csv')

non_countries = ['High-income countries']
df = df[~df['Entity'].isin(non_countries)]

# Step 1: Filter to years between 1990 and 2023
df_time_filtered = df[(df['Year'] >= 1990) & (df['Year'] <= 2023)]

print(f"Rows after year filter: {len(df_time_filtered)}")
print(f"Years range: {df_time_filtered['Year'].min()} - {df_time_filtered['Year'].max()}")

# List of emission columns to check for completeness
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

# Also include HDI in the check
columns_to_check = emission_columns + ['Human Development Index']

# Step 2: Find countries that have EVER had HDI < 0.8
countries_below_08 = df_time_filtered[df_time_filtered['Human Development Index'] < 0.8]['Entity'].unique()

# Step 3: Find countries with complete data for ALL columns (1990-2023)
expected_years = set(range(1990, 2024))  # 1990 to 2023 inclusive
total_expected_years = len(expected_years)

countries_with_complete_data = []
countries_with_missing_data = []

for country in df_time_filtered['Entity'].unique():
    country_data = df_time_filtered[df_time_filtered['Entity'] == country]
    
    # Check if country has all years
    years_present = set(country_data['Year'].unique())
    has_all_years = len(years_present) == total_expected_years
    
    # Check each column for missing values
    missing_in_columns = {}
    has_complete_columns = True
    
    for col in columns_to_check:
        if col in country_data.columns:
            # Check if this column has any missing values for the years that exist
            col_data = country_data.set_index('Year')[col]
            missing_years = col_data[col_data.isna()].index.tolist()
            if missing_years:
                missing_in_columns[col] = missing_years
                has_complete_columns = False
    
    if has_all_years and has_complete_columns:
        countries_with_complete_data.append(country)
    else:
        countries_with_missing_data.append({
            'country': country,
            'missing_years': not has_all_years,
            'missing_columns': missing_in_columns
        })

print(f"\nFound {len(countries_with_complete_data)} countries with complete data for all columns 1990-2023")
print(f"Found {len(countries_with_missing_data)} countries with incomplete data")

# Step 4: Apply HDI filter to the complete data countries
countries_kept = []
for country in countries_with_complete_data:
    if country not in countries_below_08:
        countries_kept.append(country)

# Step 5: Filter the dataframe to only include kept countries
df_final = df_time_filtered[df_time_filtered['Entity'].isin(countries_kept)]

# Save the filtered data
df_final.to_csv('filtered-data.csv', index=False)

# Print detailed summary
print("\n" + "="*60)
print("FILTERING SUMMARY")
print("="*60)
print(f"Original rows: {len(df):,}")
print(f"Rows after year filter (1990-2023): {len(df_time_filtered):,}")
print(f"Total countries in dataset: {df_time_filtered['Entity'].nunique()}")
print(f"Countries with complete data (all years, all columns): {len(countries_with_complete_data)}")
print(f"Countries with HDI < 0.8 at some point: {len(countries_below_08)}")
print(f"Final countries kept: {len(countries_kept)}")
print(f"Final rows: {len(df_final):,}")

# Show which countries were removed and why
print("\n" + "="*60)
print("COUNTRIES REMOVED (sample)")
print("="*60)

removed_count = 0
for item in countries_with_missing_data[:15]:  # Show first 15
    country = item['country']
    reasons = []
    
    if country in countries_below_08:
        reasons.append("HDI < 0.8")
    
    if item['missing_years']:
        reasons.append("missing years")
    
    for col, years in item['missing_columns'].items():
        reasons.append(f"missing {col} data for {len(years)} years")
    
    if reasons:
        print(f"  - {country}: {', '.join(reasons)}")
        removed_count += 1

if len(countries_with_missing_data) > 15:
    print(f"  ... and {len(countries_with_missing_data) - 15} more")

# Verify completeness of kept countries
print("\n" + "="*60)
print("VERIFYING KEPT COUNTRIES")
print("="*60)

for country in countries_kept[:10]:  # Check first 10 kept countries
    country_data = df_final[df_final['Entity'] == country].sort_values('Year')
    
    # Check years
    years = country_data['Year'].tolist()
    print(f"\n{country}:")
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Check a sample column for missing values
    if emission_columns:
        sample_col = emission_columns[0]
        if sample_col in country_data.columns:
            missing = country_data[sample_col].isna().sum()
            print(f"  Missing in {sample_col}: {missing}")

# Save list of kept countries
pd.DataFrame({'Country': countries_kept}).to_csv('kept_countries.csv', index=False)
print(f"\nSaved list of {len(countries_kept)} kept countries to 'kept_countries.csv'")