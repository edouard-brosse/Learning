import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate dataset
num_rows = 300
num_columns = 6

# Generate data for each column
data = {
    'Column1': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column2': np.random.normal(loc=3, scale=2, size=num_rows),
    'Column3': np.random.randint(0, 10, size=num_rows),
    'Column4': np.random.uniform(low=1, high=5, size=num_rows),
    'Column5': np.random.normal(loc=2.5, scale=1.5, size=num_rows),
    'Column6': np.random.normal(loc=-2, scale=3, size=num_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate correlations
correlation_data = {
    'Column1': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column2': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column3': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column4': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column5': np.random.normal(loc=0, scale=1, size=num_rows),
    'Column6': np.random.normal(loc=0, scale=1, size=num_rows)
}
print("start")
# Add correlations
df_corr = df + pd.DataFrame(correlation_data)

# Save to CSV
df_corr.to_csv('artificial_dataset.csv', index=False)
print('Dataset saved to CSV')