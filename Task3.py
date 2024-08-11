import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/mnt/data/householdtask3.csv'
df = pd.read_csv(file_path)

# Extract data for visualization
years = df['year']
total_households = df['tot_hhs']
average_income = df['income']

# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart for total households by year
ax1.bar(years, total_households, color='skyblue')
ax1.set_title('Total Households by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Households')
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45)

# Line chart for average income over the years
ax2.plot(years, average_income, color='green', marker='o')
ax2.set_title('Average Income Over the Years')
ax2.set_xlabel('Year')
ax2.set_ylabel('Average Income')
ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45)

# Add legends
ax1.legend(['Total Households'])
ax2.legend(['Average Income'])

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
