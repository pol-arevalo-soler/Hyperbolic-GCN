"""Euclidean layers."""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def get_dim_act(args):
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
    return dims, acts


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Network (GCN) layer.
    
    Parameters:
    -----------
    in_features : int
        Number of input features for each node.
    out_features : int
        Number of output features for each node.
    dropout : float
        Dropout rate to apply during training.
    act : callable
        Activation function (e.g., `torch.nn.ReLU()`).
    use_bias : bool
        Whether to use bias in GCNConv layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.conv = GCNConv(in_channels=in_features, 
                            out_channels=out_features,
                            add_self_loops=False,
                            normalize=True,
                            bias=use_bias)

    def forward(self, input):
        x, adj = input
        x_out = self.conv(x, adj)  
        x_out = self.act(x_out)    
        x_out = F.dropout(x_out, self.dropout, training=self.training)  
        return x_out, adj  
    
    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )