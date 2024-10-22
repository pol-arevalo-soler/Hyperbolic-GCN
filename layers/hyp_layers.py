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
    acts = [act] * args.num_layers
    dims = [args.feat_dim] + ([args.dim] * args.num_layers)

    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(args.num_layers)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(args.num_layers)]
    
    return dims, acts, curvatures

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations (e.g., Poincaré ball).
    in_features : int
        Number of input features for each node.
    out_features : int
        Number of output features for each node.
    c : float
        The curvature of the hyperbolic space.
    dropout : float
        Dropout rate to apply during training.
    act : callable
        Activation function in hyperbolic space.
    use_bias : bool
        Whether to use bias in the linear layer.
    use_att : bool
        Whether to use attention mechanism in aggregation.
    local_agg : bool
        Whether to use local aggregation in hyperbolic aggregation.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.agg = HypAgg(manifold, c, out_features, dropout)
        self.hyp_act = HypAct(manifold, c, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        h = self.hyp_act(h)
        output = h, adj
        return output

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.

    This layer performs a linear transformation in hyperbolic space using 
    Möbius matrix-vector multiplication, followed by projection back to the 
    hyperbolic manifold. Optionally adds a bias in hyperbolic space.
    
    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations (e.g., Poincaré ball).
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    c : float
        The curvature of the hyperbolic space.
    dropout : float
        Dropout rate applied to the weights.
    use_bias : bool
        Whether to use a bias term in the transformation.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weights and bias."""
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.use_bias:
            init.constant_(self.bias, 0)

    def forward(self, x):
        """Forward pass for the hyperbolic linear layer."""
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        
        # Möbius matrix-vector multiplication
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)  
        
        if self.use_bias:
            hyp_bias = self.manifold.expmap0(self.bias.view(1, -1), self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)  

        return res

    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'in_features={}, out_features={}, c={}, dropout={}, use_bias={}'.format(
            self.in_features, self.out_features, self.c, self.dropout, self.use_bias
        )

class HypAgg(Module):
    """
    Hyperbolic aggregation layer.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations.
    c : float
        Curvature of the hyperbolic space.
    in_features : int
        Number of input features for each node.
    dropout : float
        Dropout rate applied during training.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        support_t = F.dropout(support_t, self.dropout, training=self.training)

        return support_t

    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'c={}, in_features={}, dropout={}'.format(
            self.c, self.in_features, self.dropout
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations.
    c : float
        Curvature of the hyperbolic space.
    act : callable
        Activation function to apply in the tangent space (e.g., F.relu, F.leaky_relu).
    """

    def __init__(self, manifold, c, act=F.relu):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c = c
        self.act = act

    def forward(self, x):  
        return self.act(x)
                  
    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'c={}, act={}'.format(self.c, self.act.__name__ if hasattr(self.act, '__name__') else str(self.act))
    
class OriginHyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations (e.g., Poincaré ball).
    in_features : int
        Number of input features for each node.
    out_features : int
        Number of output features for each node.
    c : float
        The curvature of the hyperbolic space.
    dropout : float
        Dropout rate to apply during training.
    act : callable
        Activation function in hyperbolic space.
    use_bias : bool
        Whether to use bias in the linear layer.
    use_att : bool
        Whether to use attention mechanism in aggregation.
    local_agg : bool
        Whether to use local aggregation in hyperbolic aggregation.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(OriginHyperbolicGraphConvolution, self).__init__()
        self.linear = OriginHypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = OriginHypAgg(manifold, c_in, out_features, dropout)
        self.hyp_act = OriginHypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output

class OriginHypLinear(nn.Module):
    """
    Hyperbolic linear layer.

    This layer performs a linear transformation in hyperbolic space using 
    Möbius matrix-vector multiplication, followed by projection back to the 
    hyperbolic manifold. Optionally adds a bias in hyperbolic space.
    
    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations (e.g., Poincaré ball).
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    c : float
        The curvature of the hyperbolic space.
    dropout : float
        Dropout rate applied to the weights.
    use_bias : bool
        Whether to use a bias term in the transformation.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(OriginHypLinear, self).__init__()
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
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'in_features={}, out_features={}, c={}, dropout={}, use_bias={}'.format(
            self.in_features, self.out_features, self.c, self.dropout, self.use_bias
        )

class OriginHypAgg(Module):
    """
    Hyperbolic aggregation layer at the origin ATT_0.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations.
    c : float
        Curvature of the hyperbolic space.
    in_features : int
        Number of input features for each node.
    dropout : float
        Dropout rate applied during training.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(OriginHypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        
    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        # Aggregation at the origin
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'c={}, in_features={}, dropout={}'.format(
            self.c, self.in_features, self.dropout
        )

class OriginHypAct(Module):
    """
    Hyperbolic activation layer.

    Parameters:
    -----------
    manifold : object
        The hyperbolic manifold used for computations.
    c : float
        Curvature of the hyperbolic space.
    act : callable
        Activation function to apply in the tangent space (e.g., F.relu, F.leaky_relu).
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(OriginHypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        """Additional information to display when printing the layer."""
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out, self.act
        )