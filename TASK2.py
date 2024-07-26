import pandas as pd

# Load the CSV file into a Pandas DataFrame
file_path = './01.Data Cleaning and Preprocessing.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to inspect its structure and contents
print("First few rows of the DataFrame:")
print(df.head())
print("\nDataFrame info:")
df.info()

# Step 1: Filtering Data - Filter the DataFrame where 'Y-Kappa' is greater than 25
filtered_df = df[df['Y-Kappa'] > 25]
print("\nFiltered DataFrame (Y-Kappa > 25):")
print(filtered_df.head())
print(f"Filtered DataFrame shape: {filtered_df.shape}")

# Step 2: Handling Missing Values
# Fill missing values for numeric columns with the mean
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing values for non-numeric columns with a placeholder (e.g., 'missing')
non_numeric_cols = df.select_dtypes(exclude='number').columns
df[non_numeric_cols] = df[non_numeric_cols].fillna('missing')

print("\nDataFrame with filled missing values:")
print(df.head())

# Step 3: Calculating Summary Statistics
# Summary statistics for the original DataFrame
original_summary_stats = df.describe(include='all')

# Summary statistics for the filtered DataFrame
filtered_summary_stats = filtered_df.describe(include='all')

print("\nSummary statistics for the original DataFrame:")
print(original_summary_stats)

print("\nSummary statistics for the filtered DataFrame:")
print(filtered_summary_stats)
