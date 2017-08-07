import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
import utils as ut 


class DRF(nn.Module):
    def __init__(self, n_features, n_outputs,
                 kernel_name="rbf", n_layers=1, n_rff=10, df=10):
        super(DRF, self).__init__()

        d_in = n_features
        d_out = n_outputs

        dtype = torch.FloatTensor

        if kernel_name == "rbf": 
            self.K = ut.rbf

        self.n_layers = n_layers

        n_rff = n_rff

        self.n_rff = n_rff * np.ones(n_layers, dtype = np.int64)
        self.df = df * np.ones(n_layers, dtype=np.int64)

        self.d_in = np.concatenate([[d_in], self.df[:(n_layers - 1)]])
        self.d_out = self.n_rff

        self.dhat_in = self.n_rff * 2
        self.dhat_out = np.concatenate([self.df[:-1], [d_out]])


        # SET UP PARAMETERS
        self.theta_logsigma = Variable(torch.randn(self.n_layers).type(dtype), 
                               requires_grad=True)
        self.theta_loglength = Variable(torch.ones(self.n_layers).type(dtype),
                               requires_grad=False)
        
        self.W_mean= [Variable(torch.zeros(self.dhat_in[i], self.dhat_out[i]), 
                      requires_grad=True) for i in range(self.n_layers)]

        self.W_logsigma = [Variable(torch.zeros(self.dhat_in[i], self.dhat_out[i]), 
                           requires_grad=True) for i in range(self.n_layers)]
        
        self.Omega_mean= [Variable(torch.zeros(self.d_in[i], self.d_out[i]), 
                          requires_grad=False)  for i in range(self.n_layers)]

        self.Omega_logsigma = [self.theta_loglength[i].expand(self.d_in[i],                   self.d_out[i]) * Variable(torch.ones(self.d_in[i                      ], self.d_out[i]), requires_grad=False) for i                    in range(self.n_layers)]
   
        self.Omega_eps = [Variable(torch.randn(self.d_in[i], self.d_out[i]), 
                          requires_grad=False)  for i in range(self.n_layers)]

        self.W_eps = [Variable(torch.randn(self.dhat_in[i], self.dhat_out[i]), 
                          requires_grad=False)  for i in range(self.n_layers)]       
    def forward(self, x):
        # SAMPLE FOR OMEGA - Reparametrization (Section 3.3 in [1])
        Omega_approx = []
        for i in range(self.n_layers):
            eps = Variable(torch.randn(self.d_in[i], self.d_out[i]))
            Omega_approx += [self.Omega_mean[i] + 
                             torch.exp(self.Omega_logsigma[i] / 2) * 
                             self.Omega_eps[i]]
        # SAMPLE FOR W - Reparametrization (Equation 12 in [1])
        W_approx = []
        for i in range(self.n_layers):
            eps = Variable(torch.randn(self.dhat_in[i], self.dhat_out[i]))
            W_approx += [self.W_mean[i] + torch.exp(self.W_logsigma[i] / 2)
                         * self.W_eps[i]]
        F = x
        N = x.size(0)
        for i in range(self.n_layers):
            N_rf = self.n_rff[i] 

            # TODO: FeedForward Approach - add X input at each hiden layer
            
            # Equation 6 in [1]
            phi_half = torch.mm(F, Omega_approx[i])

            phi = torch.exp(0.5 * self.theta_logsigma[i]).expand(N, N_rf)
            phi = phi / np.sqrt(1. * N_rf)

            A = phi * torch.cos(phi_half)
            B = phi * torch.sin(phi_half)

            phi = torch.cat([A,B], 1)

            # First line under Equation 6 in [1]
            F = torch.mm(phi, W_approx[i])

        return F 
        
    def compute_objective(self, X, y):
        F = self.forward(X)

        # SOFTMAX LOSS
        y_logprob = nn.LogSoftmax()(F)
        loss = nn.NLLLoss()(y_logprob, y)

        # TODO: ADD KLL regularization on the parameters

        return loss, torch.exp(y_logprob)

    def fit(self, X, y, verbose=0, lr=1e-3, epochs=100): 
        for i in range(epochs):
            loss, y_prob = self.compute_objective(X, y)
            loss.backward()

            print "\n%d - loss: %.3f" % (i, loss.data[0])

            acc = np.mean(np.argmax(ut.var2numpy(y_prob), 1) == ut.var2numpy(y))
            print "%d - acc: %.3f" % (i, acc)

            self.theta_logsigma.data -= lr * self.theta_logsigma.grad.data
            self.theta_logsigma.grad.data.zero_()

            # self.theta_loglength.data -= lr * self.theta_loglength.grad.data
            # self.theta_loglength.grad.data.zero_()

            for i in range(self.n_layers):
                self.W_mean[i].data -= lr * self.W_mean[i].grad.data
                self.W_mean[i].grad.data.zero_()

                self.W_logsigma[i].data -= lr * self.W_logsigma[i].grad.data
                self.W_logsigma[i].grad.data.zero_()

        print y_prob
        print "%d - loss: %.3f" % (i, loss.data[0])
        print "%d - acc: %.3f" % (i, acc)

# [1] Random Feature Expansions for Deep Gaussian Processes