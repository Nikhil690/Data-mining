import pandas as pd
import numpy as np
df  = pd.read_csv('titanic.csv')
print(df.head())

df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip()) #Dealing with 'Name' attribute
# print(df["Title"].value_counts())

df["Title"] = df["Title"].replace(["Dr","Col","Rev","Ms","Dona"],"Other")
# print(df["Title"].value_counts())

df['Age'] = df['Age'].clip(lower=1) #Dealing with 'Age' attribute, by converting it into int and filling NaN values with fillna.
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'] = df['Age'].astype(int)
print(df["Age"].isna())

print(df['Fare'].fillna(df['Fare'].median())) #Dealing with 'Fare' attribute by filling its NaN values with median value.

df['Cabin'].dropna().str.isalnum().sum() #Dealing with 'Cabin' attribute by droping and filling NaN values with mode.
mode = df["Cabin"].mode()[0]
df["Cabin"] = df["Cabin"].fillna(mode)
print("\n" + 15 * "-" + "Missing Values".center(15) + 15 * "-")
print(df.isnull().sum())

gender_mapping = {"male": 0, "female": 1}
df["Sex"] = df["Sex"].map(gender_mapping) #Mapping 'Sex' attribute with values 0 and 1.
print(df.head())
