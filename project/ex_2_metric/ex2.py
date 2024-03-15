import pandas as pd
import numpy as np

data = pd.read_csv("dataset.csv")

def custom_metric(sample1, sample2):
    numeric_difference = np.sum((sample1[['age', 'height']] - sample2[['age', 'height']]) ** 2)
    
    weights = {
        'job': 1 / data['job'].value_counts(),
        'city': 1 / data['city'].value_counts(),
        'favorite music style': 1 / data['favorite music style'].value_counts()
    }
    
    categorical_difference = 0
    for feature in ['job', 'city', 'favorite music style']:
        categorical_difference += weights[feature][sample1[feature]] != weights[feature][sample2[feature]]
    
    return numeric_difference + categorical_difference

num_samples = len(data)
dissimilarity_matrix = np.zeros((num_samples, num_samples))
for i in range(num_samples):
    for j in range(num_samples):
        dissimilarity_matrix[i, j] = custom_metric(data.iloc[i], data.iloc[j])

mean_dissimilarity = np.mean(dissimilarity_matrix)
std_dissimilarity = np.std(dissimilarity_matrix)

np.save("dissimilarity_matrix.npy", dissimilarity_matrix)

print("Mean dissimilarity:", mean_dissimilarity)
print("Standard deviation of dissimilarity:", std_dissimilarity)
