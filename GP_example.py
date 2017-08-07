import numpy as np
from models.GP import GP
from sklearn.datasets import make_regression
import torch
from torch.autograd import Variable

if __name__ == "__main__":
    # 1. LOAD DATASET
    np.random.seed(1)

    X, y = make_regression(n_samples=100, n_features=20, 
                           n_informative=10)

    X = ((X - X.mean(axis=0)) / X.std(axis=0))
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 2. CONVERT DATASET to VAR
    X = Variable(torch.FloatTensor(X))
    y = Variable(torch.FloatTensor(y))

    # 2. LOAD MODEL
    gp = GP(kernel="rbf")

    # 3. OPTIMIZE PARAMETERS
    gp.fit(X, y, verbose=1, lr=1e-4, epochs=100)
