"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, dropout, act, use_bias):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in)
        self.hyp_act = HypAct(act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}, dropout={}, use_bias={}'.format(
            self.in_features, self.out_features, self.c, self.dropout, self.use_bias
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        return support_t

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, act):
        super(HypAct, self).__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)

    def extra_repr(self):
        return 'act={}'.format(self.act.__name__ if hasattr(self.act, '__name__') else str(self.act))

class NewHyperbolicGraphConvolution(nn.Module):
    """
    New Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, dropout, act, use_bias, flag):
        super(NewHyperbolicGraphConvolution, self).__init__()
        self.linear = LinearGCN(in_features, out_features)
        self.agg = MobiusAdd(manifold, c_in, out_features, use_bias)
        self.hyp_act = NewHypAct(act, dropout, out_features, flag)

    def forward(self, entering):
        x, adj = entering

        h = self.linear.forward(x, adj)
        h = self.agg.forward(h)
        h = self.hyp_act.forward(h)
        
        output = h, adj
        return output

class LinearGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))

    def forward(self, h, adj):
        # y=D^-1*Adj*H
        y = torch.spmm(adj, h)
        # y*W^T
        res = F.linear(input=y, weight=self.weight, bias=None) 
        return res
    
class MobiusAdd(nn.Module):
    def __init__(self, manifold, out_features, c_in, use_bias):
        super(MobiusAdd, self).__init__()
        self.manifold = manifold
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features)) 
        self.c = c_in
    
    def reset_parameters(self):
        init.constant_(self.bias, 0)

    def forward(self, h):
        # exp_0(D^-1*Adj*H*W^T)
        h_hyp = self.manifold.expmap0(u=h, c=self.c)
        h_hyp = self.manifold.proj(x=h_hyp, c=self.c)

        # h_hyp \oplus exp0(bias)
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(h_hyp, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)

        res = self.manifold.logmap0(p=res, c=self.c)
        return res

class NewHypAct(nn.Module):
    def __init__(self, act, dropout, out_features, flag):
        super(NewHypAct, self).__init__()
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.flag = flag
        if act == F.leaky_relu:
            self.act = lambda x: F.leaky_relu(x, negative_slope=0.1)
        else:
            self.act = act

    def forward(self, x):
        if self.flag:
            x_norm = self.batch_norm(x)
            x_norm = self.act(x_norm)
            x_residual = x+x_norm
            res = self.dropout(x_residual)
        else:
            res = self.dropout(x)
        return res