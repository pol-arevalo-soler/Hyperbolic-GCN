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

class Linear(nn.Module):
    """
    Simple Linear layer with dropout.

    Parameters:
    -----------
    in_features : int
        Number of input features (dimension of input).
    out_features : int
        Number of output features (dimension of output).
    dropout : float
        Dropout rate for regularization (between 0 and 1).
    act : callable
        Activation function to apply after the linear transformation. 
        Common choices are F.relu, F.sigmoid, etc.
    use_bias : bool
        Whether to use a bias term in the linear layer.
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, use_bias=True):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, bias=use_bias) 
        self.act = act

    def forward(self, x):
        """
        Forward pass through the linear layer with dropout and activation.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (N, in_features), where N is the batch size.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (N, out_features) after applying the linear transformation, 
            dropout, and activation function.
        """
        hidden = self.linear(x)  
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out