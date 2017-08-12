import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
import utils as ut
# MODELS


dtype = torch.FloatTensor

class GP(nn.Module):
    def __init__(self, kernel="rbf"):
        super(GP, self).__init__()

        self.mean = Variable(torch.zeros(1).type(dtype), 
                             requires_grad=True)
        self.noise = Variable(torch.ones(1).type(dtype)*0.001, 
                              requires_grad=True)
        
        if kernel == "rbf":
            self.rbf_scale_length = Variable(torch.zeros(2).type(dtype), 
                                             requires_grad=True)
            self.K = ut.rbf

    def fit(self, X, y, verbose=0, lr=1e-3, epochs=100):
        self.X = X
        self.y = y

        for i in range(epochs):
            loss = self.compute_objective(X, y)
            loss.backward()

            # UPDATE PARAMS
            self.rbf_scale_length.data -= (lr * 
                                           self.rbf_scale_length.grad.data)
            self.rbf_scale_length.grad.data.zero_()

            if verbose:
                print "\n%d - loss: %.3f" % (i, loss.data[0])

                # y_pred, _ = self.predict(X)
                # mse = np.mean((ut.var2numpy(y_pred) - ut.var2numpy(y))**2)
                # print "%d - mse: %.3f" % (i, mse)

                # gp.mean.data -= lr * gp.mean.grad.data
                # gp.mean.grad.data.zero_()

                # gp.noise.data -= lr * gp.noise.grad.data
                # gp.noise.grad.data.zero_()
    

    def compute_objective(self, X, y):
        N = y.size(0)

        y_diff = (y - self.mean.expand_as(y))

        Knn = (self.K(X, X, self.rbf_scale_length) + 
              torch.exp(self.noise).expand(N,N) * Variable(torch.eye(N)))
        
        A = - 0.5 * ut.determinent(Knn)
        B = - 0.5 * y_diff.dot(torch.mv(Knn.inverse(), y_diff))
        C = - 0.5 * N * np.log(2 * np.pi)

        neg_logp =  - (A + B + C)

        return neg_logp

    def predict(self, Xt):
        N = self.y.size(0)
        Knn = (self.K(self.X, self.X, self.rbf_scale_length) + 
               torch.exp(self.noise).expand(N,N) * Variable(torch.eye(N)))

        Kmn = self.K(Xt, self.X, self.rbf_scale_length)
        Kmm = self.K(Xt, Xt, self.rbf_scale_length)

        K_inv = ut.Inverse()(Knn) 

        mu = torch.mv(Kmn, 
                      torch.mv(K_inv, self.y - self.mean.expand_as(self.y)))
        mu += self.mean.expand_as(mu)

        sigma = Kmm - torch.mm(Kmn, torch.mm(K_inv, Kmn.t()))

        return mu, sigma

