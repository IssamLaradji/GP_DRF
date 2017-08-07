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

class Inverse(torch.autograd.Function):

    def forward(self, input):
        inverse = torch.inverse(input)
        self.save_for_backward(inverse)
        return inverse


    def backward(self, grad_output):
        input, = self.saved_tensors
        return -torch.mm(input.t(), torch.mm(grad_output, input.t()))



class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def numpy2var(x):
    return torch.autograd.Variable(torch.FloatTensor(x))

def var2numpy(x):
    return x.data.numpy()
