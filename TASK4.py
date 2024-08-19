import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/mnt/data/USvideos.csv')

# Display basic info and summary statistics
print(df.info())
print(df.describe())

# Distribution of numerical variables
plt.figure(figsize=(12, 6))
df.hist(bins=30, figsize=(20, 15), color='skyblue')
plt.suptitle('Distribution of Numerical Variables')
plt.show()

# Boxplots to identify outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.title('Boxplots for Outlier Detection')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
