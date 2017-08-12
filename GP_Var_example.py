import numpy as np
from models.GP_Var import GP_Var
from sklearn.datasets import make_classification
import torch
from torch.autograd import Variable

if __name__ == "__main__":
    # 1. LOAD DATASET
    np.random.seed(1)

    n_classes = 2
    X, y = make_classification(n_samples=100, n_features=20, 
                               n_informative=10, n_classes=n_classes)
    X = ((X - X.mean(axis=0)) / X.std(axis=0))
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 2. CONVERT DATASET to VAR
    X = Variable(torch.FloatTensor(X))
    y = Variable(torch.LongTensor(y))

    # 2. LOAD MODEL
    gp = GP_Var(n_outputs=2, kernel="rbf", n_inducing=50)

    # 3. OPTIMIZE PARAMETERS
    gp.fit(X, y, verbose=1, lr=1., epochs=1000)

