from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_dataset(name="breast_cancer"):
    if name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = np.where(data.target == 0, -1, 1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
