import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DatasetInstance(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

def get_sudoku_dataset(csv_path):
    data = pd.read_csv(csv_path)

    feat = []
    label = []
    for quizz in data['quizzes']:

        x = np.array([int(j) for j in quizz]).reshape((1,9,9))
        feat.append(x)

    feat = np.array(feat)
    feat = feat/9
    feat -= .5

    for solution in data['solutions']:

        x = np.array([int(j) for j in solution]).reshape((9,9)) - 1
        label.append(x)

    label = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)

    train_dataset = DatasetInstance(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = DatasetInstance(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return train_dataset, test_dataset