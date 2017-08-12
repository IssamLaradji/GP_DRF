import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn


# KERNELS

def rbf(x, xp, scale_length, diag=False):
    scale = torch.exp(scale_length[0])
    length = torch.exp(scale_length[1])

    n, d = x.size()
    m, d = xp.size()

    res = 2 * torch.mm(x, xp.transpose(0,1))

    x_sq = torch.bmm(x.view(n, 1, d), x.view(n, d, 1))
    xp_sq = torch.bmm(xp.view(m, 1, d), xp.view(m, d, 1))

    x_sq = x_sq.view(n, 1).expand(n, m)
    xp_sq = xp_sq.view(1, m).expand(n, m)

    res = res - x_sq - xp_sq
    res = (0.5 * res) / length.expand_as(res)
    res = scale.expand_as(res) * torch.exp(res)

    if diag:
        return torch.diag(res) 
        
    return res  

# FUNCTIONS

class Cholesky(torch.autograd.Function):

    def forward(ctx, a):
        l = torch.potrf(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_tensors
        # Gradient is l^{-H} @ ((l^{H} @ grad) * (tril(ones)-1/2*eye)) @ l^{-1}
        # TODO: ideally, this should use some form of solve triangular instead of inverse...
        linv =  l.inverse()
        
        inner = torch.tril(torch.mm(l.t(),grad_output))*torch.tril(1.0-l.new(l.size(1)).fill_(0.5).diag())
        s = torch.mm(linv.t(), torch.mm(inner, linv))

        # could re-symmetrise 
        #s = (s+s.t())/2.0
            
        return s

def log_determinent(M):
    return 2. * Cholesky()(M).diag().log().sum()

def KL_diagLog(mean, mean_prior, logsigma, logsigma_prior):
    # https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
    mu = mean_prior.view(-1)
    mu_pr = mean.view(-1)

    logSig = logsigma.view(-1)
    logSig_prior = logsigma_prior.view(-1)

    A = logSig_prior - logSig
    B = torch.pow(mu - mu_pr, 2)  / torch.exp(logSig_prior)
    C = torch.exp(logSig - logSig_prior) - 1

    return 0.5 * torch.sum(A + B + C)

def KL_diagSqrt(mean, mean_prior, sqrtsigma, sqrtsigma_prior):
    # https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
    # https://github.com/GPflow/GPflow/blob/master/GPflow/kullback_leiblers.py

    N = mean.size(0)
    sigma_prior_inv = torch.diag(sqrtsigma_prior).inverse()
    # Prior log-det term.
    A = log_determinent(torch.diag(sqrtsigma_prior**2))

    # Post log-det term
    B = - torch.sum(torch.log(sqrtsigma**2))

    # Mahalanobis term.
    mu_diff = (mean_prior - mean)
    C = torch.sum(sigma_prior_inv.mv(mu_diff)**2)

    # Trace Term
    D = torch.dot(torch.diag(sigma_prior_inv), sqrtsigma**2)

    return 0.5 * (A + B + C + D - N)

# def KL_full(mean, mean_prior, sqrtsigma, sqrtsigma_prior):
#     # https://tgmstat.wordpress.com/2013/07/10/kullback-leibler-divergence/
#     # https://github.com/GPflow/GPflow/blob/master/GPflow/kullback_leiblers.py
    
#     N = mean.size(0)
#     sigma_prior_inv = sqrtsigma_prior.inverse()
#     import pdb; pdb.set_trace()  # breakpoint 7915e892 //

#     A = log_determinent(sqrtsigma_prior) - log_determinent(sigma))
#     B = torch.trace(sigma_prior_inv.mm(sigma))

#     mu_diff = (mean_prior - mean)
#     C = mu_diff.dot(sigma_prior_inv.mv(mu_diff))

#     return 0.5 * A + B + C - N


def numpy2var(x):
    return torch.autograd.Variable(torch.FloatTensor(x))

def var2numpy(x):
    return x.data.numpy()
