"""
Task 1: Data Pipeline Development
Internship: CodTech
Author: mayank sagar 

This script loads sample employee data, preprocesses it using numeric/categorical pipelines,
transforms the features, and saves them as CSV files for modeling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# 1. Load Data (make sure sample_data.csv is in the same folder)
df = pd.read_csv('sample_data.csv')  # CSV file with employee data

# 2. Separate features and target column
X = df.drop('target', axis=1)
y = df['target']

# 3. Identify column types
numeric_features = ['age', 'salary']
categorical_features = ['department', 'city']

# 4. Define individual pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 5. Combine with ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 6. Split dataset (keeping the ratio modest because data is small)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 7. Fit-transform on train, transform on test
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 8. Get new feature names from one-hot encoder
onehot_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_columns = numeric_features + list(onehot_cols)

# 9. Save transformed data (ETL complete)
pd.DataFrame(X_train_transformed, columns=all_columns).to_csv("X_train_transformed.csv", index=False)
pd.DataFrame(X_test_transformed, columns=all_columns).to_csv("X_test_transformed.csv", index=False)
pd.DataFrame({'target': y_train}).to_csv("y_train.csv", index=False)
pd.DataFrame({'target': y_test}).to_csv("y_test.csv", index=False)

print("\nETL Data Pipeline completed successfully.")
print("Files saved: X_train_transformed.csv, X_test_transformed.csv, y_train.csv, y_test.csv\n")