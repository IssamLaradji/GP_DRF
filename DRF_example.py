import numpy as np
from models.DRF import DRF
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
    drf = DRF(n_features=X.size(1),
             n_outputs=n_classes, kernel_name="rbf",  
             n_rff=500, df=50, n_layers=2)

    # 3. OPTIMIZE PARAMETERS
    drf.fit(X, y, verbose=1, lr=1e-1, epochs=800)   
 