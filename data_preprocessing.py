# noramalization, standardization, log transformation, aggregation ,sampling ,binarization, discretization

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Fetch Heart Disease dataset from UCI repository
heart_disease = fetch_openml(name='cleveland', version=1)

# Convert the dataset to pandas dataframe
heart_df = pd.DataFrame(data=heart_disease.data, columns=heart_disease.feature_names)
heart_df['target'] = heart_disease.target

# Display the first few rows of the dataset
print(heart_df.head())

# Select only numeric columns for normalization, standardization, transformation, aggregation, discretization, and binarization
numeric_cols = heart_df.select_dtypes(include='number').columns.tolist()

# Apply Min-Max scaling (Normalization)
min_max_scaler = MinMaxScaler()
heart_df_normalized = heart_df.copy()
heart_df_normalized[numeric_cols] = min_max_scaler.fit_transform(heart_df[numeric_cols])

# Apply Standardization (z-score normalization)
standard_scaler = StandardScaler()
heart_df_standardized = heart_df.copy()
heart_df_standardized[numeric_cols] = standard_scaler.fit_transform(heart_df[numeric_cols])

# Apply Transformation (log transformation)
transformed_features = heart_df[numeric_cols].apply(lambda x: np.log(x + 1))  # Adding 1 to avoid log(0)
heart_df_transformed = pd.concat([transformed_features, heart_df['target']], axis=1)

# Apply Aggregation (calculate mean, median, min, max)
agg_functions = ['mean', 'median', 'min', 'max']
heart_df_aggregated = heart_df[numeric_cols].agg(agg_functions)
heart_df_aggregated = heart_df_aggregated.transpose()  # Transpose to make columns represent statistics

# Handle missing values
imputer = SimpleImputer(strategy='mean')
heart_df_imputed = pd.DataFrame(imputer.fit_transform(heart_df[numeric_cols]), columns=numeric_cols)

# Apply Discretization after replacing NaN values with a constant
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)  # Set subsample=None
heart_df_discretized = heart_df_imputed.fillna(-9999)  # Fill NaN values with a constant
heart_df_discretized[numeric_cols] = discretizer.fit_transform(heart_df_discretized)

# Apply Binarization
binarizer = Binarizer(threshold=0.5)
heart_df_binarized = heart_df_imputed.copy()
heart_df_binarized[numeric_cols] = binarizer.fit_transform(heart_df_binarized[numeric_cols])

# Apply Random Sampling
random_state = 42  # for reproducibility
sample_size = 100  # number of samples to select
heart_df_sampled = heart_df_imputed.sample(n=sample_size, random_state=random_state)

# Display the normalized dataset
print("\nNormalized dataset:")
print(heart_df_normalized.head())

# Display the standardized dataset
print("\nStandardized dataset:")
print(heart_df_standardized.head())

# Display the transformed dataset
print("\nTransformed dataset:")
print(heart_df_transformed.head())

# Display the aggregated dataset
print("\nAggregated dataset:")
print(heart_df_aggregated)

# Display the discretized dataset
print("\nDiscretized dataset:")
print(heart_df_discretized.head())

# Display the binarized dataset
print("\nBinarized dataset:")
print(heart_df_binarized.head())

# Display the sampled dataset
print("\nSampled dataset:")
print(heart_df_sampled.head())
