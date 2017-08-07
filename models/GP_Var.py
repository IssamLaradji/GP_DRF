import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
import utils as ut 

dtype = torch.FloatTensor 

class GP_Var(nn.Module):
    def __init__(self, n_outputs, kernel="rbf", n_inducing=100):
        super(GP_Var, self).__init__()

        if kernel == "rbf":
            self.rbf_scale_length = Variable(torch.randn(2).type(dtype), 
                                       requires_grad=True)
            self.K = ut.rbf

        self.n_output = n_outputs
        self.n_inducing = n_inducing

         # 1. --------- SET UP GP LAYER PARAMETERS
        self.q_mu = Variable(torch.randn(n_inducing, self.n_output).type(dtype),                  requires_grad=True)

        self.q_cov = Variable(torch.randn(n_inducing, self.n_output).type(dtype), requires_grad=True)

        self.q_eps = Variable(torch.ones(n_inducing, self.n_output).type(dtype), requires_grad=False)



    def forward(self, X):
        m = self.U.size(0)

        # 1. ----- FIRST LAYER IS GP LAYER
        Knm = self.K(X, self.U, self.rbf_scale_length)
        Kmm = (self.K(self.U, self.U, self.rbf_scale_length) + 
                    Variable(torch.eye(m)))

        KmmInv = ut.Inverse()(Kmm)

        # COMPUTE MEANS
        KnmKmmInv = torch.mm(Knm, KmmInv)

        A = torch.mm(KnmKmmInv, self.q_mu)
       
        # DIAGONAL ONLY FOR COV
        # B = self.K(self.X, self.X, self.rbf_scale_length, diag=True)
        # B = B - torch.squeeze(torch.sum(KnmKmmInv**2, 1))
        # B = torch.unsqueeze(B, 1)

        # B = B.repeat(1, self.n_latent)
        # B = B + torch.mm(KnmKmmInv, self.gp_cov)

        # B = torch.diag(B)

        #F = A + B * self.gp_eps
        F = A

        return F 
        
    def compute_objective(self, X, y):
        F = self.forward(X)

        # SOFTMAX LOSS
        y_logprob = nn.LogSoftmax()(F)
        loss = nn.NLLLoss()(y_logprob, y)

        # TODO: ADD KLL regularization on the parameters

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

            # self.q_cov.data -= lr * self.q_cov.grad.data
            # self.q_cov.grad.data.zero_()

        print y_prob
        print "%d - loss: %.3f" % (i, loss.data[0])
        print "%d - acc: %.3f" % (i, acc)

# [1] Random Feature Expansions for Deep Gaussian Processes