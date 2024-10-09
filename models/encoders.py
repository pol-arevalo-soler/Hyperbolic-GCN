"""Graph encoders."""
import torch.nn as nn

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, get_dim_act

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, x, adj):
        input = (x, adj)
        output, _ = self.layers.forward(input)

        return output

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__()
        assert args.num_layers > 0, "Number of layers must be greater than 0."
        dims, acts = get_dim_act(args)

        self.layers = nn.Sequential(
            *[
                GraphConvolution(
                    in_dim=dims[i],
                    out_dim=dims[i+1],
                    dropout=args.dropout,
                    act=acts[i],
                    bias=args.bias
                )
                for i in range(len(dims)-1)
            ]
        )

class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, args):
        super(HGCN, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0, "Number of layers must be greater than 0."
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)

        self.layers = nn.Sequential(
            *[
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold,
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    c=self.curvatures[i],
                    dropout=args.dropout,
                    act=acts[i],
                    bias=args.bias
                )
                for i in range(len(dims) - 1)
            ]
        )

    def encode(self, x, adj):        
        return super(HGCN, self).encode(x, adj)