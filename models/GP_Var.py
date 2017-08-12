import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
import utils as ut 

dtype = torch.FloatTensor 

class GP_Var(nn.Module):
    def __init__(self, n_outputs, kernel="rbf", n_inducing=50, cov_diag=True):
        super(GP_Var, self).__init__()

        if kernel == "rbf":
            self.rbf_scale_length = Variable(torch.randn(n_outputs, 2).type(dtype), requires_grad=True)

            self.K = ut.rbf

        self.n_funcs = n_outputs
        self.n_inducing = n_inducing

        # 1. --------- SET UP GP LAYER PARAMETERS
        self.p_mu = Variable(torch.zeros(self.n_funcs, n_inducing).type(dtype),                  requires_grad=False)

        self.q_mu = Variable(torch.zeros(self.n_funcs, n_inducing).type(dtype),                  requires_grad=True)

        # Ensure q_sigma is positive definite 
        if cov_diag:
            self.p_sqrt = Variable(torch.ones(self.n_funcs, n_inducing), 
                                    requires_grad=False)
            self.q_sqrt = Variable(self.p_sqrt.data.clone(), 
                                    requires_grad=True)

        # Q = np.random.randn(self.n_funcs, n_inducing, n_inducing)

        # for i in range(self.n_funcs):
        #     Q[i] = Q[i].dot(Q[i].T)

        self.cov_diag = cov_diag

        self.q_eps = Variable(torch.ones(self.n_funcs).type(dtype), requires_grad=False)

    def forward(self, X):
        n = X.size(0)
        m = self.U.size(0)
        outs = Variable(torch.zeros(self.n_funcs, n))
        #cov_outs = Variable(torch.zeros(self.n_funcs, n))

        # 1. ----- FIRST LAYER IS GP LAYER
        for i in range(self.n_funcs):
            Knm = self.K(X, self.U, self.rbf_scale_length[i])
            Kmm = (self.K(self.U, self.U, self.rbf_scale_length[i]) + 
                        Variable(torch.eye(m)))

            KmmInv = Kmm.inverse()

            # Equation 18 in [2] 

            # COMPUTE MEANS

            A = torch.mm(Knm, KmmInv)

            Z = torch.mv(A, self.q_mu[i])
       
            # DIAGONAL ONLY FOR COV
            if self.cov_diag:
                # https://github.com/GPflow/GPflow/blob/master/GPflow/conditionals.py

                V = torch.diag(self.K(X, X, self.rbf_scale_length[i]))
                V -= torch.sum(A**2, 1)
                V += torch.sum(((A * self.q_sqrt[i])**2), 1)

            else:
                pass
                # V = torch.diag(self.K(X, X, self.rbf_scale_length[i]))
                # V = V + A.mm(self.q_sigma[i] - Kmm).mm(A.t())
               
                # V = torch.diag(V) 

            # https://github.com/GPflow/GPflow/blob/master/GPflow/likelihoods.py in Variational Expectation                    
            outs[i] = Z + torch.sqrt(2.0 * V) * self.q_eps[i]

        return outs.t() 
        
    def compute_objective(self, X, y):
        F = self.forward(X)

        # SOFTMAX LOSS
        y_logprob = nn.LogSoftmax()(F)

        loss = nn.NLLLoss()(y_logprob, y)

        # KL regularization on the parameters 
        for i in range(self.n_funcs):       
            loss += 0.001 * ut.KL_diagSqrt(self.q_mu[i],  self.p_mu[i], 
                                           self.q_sqrt[i], self.p_sqrt[i])

        return loss, torch.exp(y_logprob)

    def fit(self, X, y, verbose=0, lr=1e-3, epochs=100): 
        self.U = X[:self.n_inducing]
        self.y = y

        for i in range(epochs):
            loss, y_prob = self.compute_objective(X, y)
            loss.backward()

            print "\n%d - loss: %.3f" % (i, loss.data[0])

            acc = np.mean(np.argmax(ut.var2numpy(y_prob), 1) == ut.var2numpy(y))
            print "%d - acc: %.3f" % (i, acc)

            self.q_mu.data -= lr * self.q_mu.grad.data
            self.q_mu.grad.data.zero_()

            self.q_sqrt.data -= lr * self.q_sqrt.grad.data
            self.q_sqrt.grad.data.zero_()

        print y_prob
        print "%d - loss: %.3f" % (i, loss.data[0])
        print "%d - acc: %.3f" % (i, acc)

# [1] Random Feature Expansions for Deep Gaussian Processes

# [2] Scalable Variational Gaussian Process Classification