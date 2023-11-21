from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load iris dataset

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# Add any other classifiers you want to try here

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Extract the 'sex' column
sex = df['sex']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'sex' column
sex_encoded = label_encoder.fit_transform(sex)

# Update the 'sex' column in the DataFrame
df['sex'] = sex_encoded

# Preprocess the dataset
X = df.drop('age', axis=1)
y = df['age']

# Define the target variable and features
target = 'age'
features = [col for col in df.columns if col != target]


# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Compute pairwise correlations between features
correlations = X.corr()
# Train a random forest classifier to estimate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Print results
print("Pairwise feature correlations:")
print(correlations)
print("\nFeature importances:")
print(importances)

