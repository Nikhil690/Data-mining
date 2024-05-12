from sklearn.datasets import fetch_openml
import pandas as pd

# Fetch the Bank Marketing dataset from UCI repository via OpenML
bank_marketing = fetch_openml(name='bank-marketing', version=1)

# Convert the dataset to a pandas DataFrame
bank_df = pd.DataFrame(data=bank_marketing.data, columns=bank_marketing.feature_names)
bank_df['target'] = bank_marketing.target

# Display the initial dataset details
print("Original dataset:")
print(bank_df.head())

# Print column names for further investigations
print("\nColumn Names:")
print(bank_df.columns)

# Handling Missing Values - Count and display any missing values
missing_values_count = bank_df.isnull().sum()
print("\nMissing values count:")
print(missing_values_count)

# Define validation rules for the data
def age_validation_rule(df):
    if 'age' in df.columns:
        return (df['age'] >= 18) & (df['age'] <= 100)
    else:
        return pd.Series(False, index=df.index)

# Apply validation rules to the dataset
def apply_validation_rules(df):
    validation_results = pd.DataFrame(index=df.index)
    validation_results['Age Validation'] = age_validation_rule(df)
    # Additional validation rules can be applied here
    return validation_results

# Apply the validation rules and print results
validation_results = apply_validation_rules(bank_df)
print("\nValidation Results:")
print(validation_results)
