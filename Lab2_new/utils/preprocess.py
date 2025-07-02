# utils/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_default_dataset():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def load_uploaded_dataset(uploaded_file, target_column):
    df = pd.read_csv(uploaded_file)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
