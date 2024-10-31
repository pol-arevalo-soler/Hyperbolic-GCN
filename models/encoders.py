import numpy as np
import torch
import torch.nn as nn

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act

class Encoder(nn.Module):
    """
    Abstract base class for graph encoders.
    
    Methods:
    --------
    encode(x, adj):
        Encodes the input node features and adjacency matrix by passing 
        them through the encoder layers.

    Attributes:
    -----------
    layers: nn.Sequential 
        This holds the actual layers of the encoder (e.g., graph convolutional 
        layers), which need to be defined in a subclass. The input is propagated
        through these layers to produce the output embeddings.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, x, adj):
        input = (x, adj)
        output, _ = self.layers.forward(input)

        return output

class GCN(Encoder):
    """
    Graph Convolution Networks (GCN).
    Parameters:
    -----------
    c : float
        A curvature parameter (used for hyperbolic embeddings, if applicable).
    args : Namespace
        A namespace object containing hyperparameters for the GCN, including:
            - num_layers (int): Number of layers in the GCN.
            - dropout (float): Dropout rate for regularization.
            - bias (bool): Whether to include a bias term in the convolutional layers.

    Attributes:
    -----------
    layers : nn.Sequential
        A sequential container that holds the GraphConvolution layers. Each layer 
        transforms the input node features according to the specified dimensions 
        and activation functions.

    Methods:
    --------
    encode(x, adj):
        Encodes the input node features and adjacency matrix by passing 
        them through the graph convolution layers.
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        assert args.num_layers > 0, "Number of layers must be greater than 0."
        dims, acts = get_dim_act(args)

        self.layers = nn.Sequential(
            *[
                GraphConvolution(
                    in_features=dims[i],
                    out_features=dims[i+1],
                    dropout=args.dropout,
                    act=acts[i],
                    use_bias=args.bias
                )
                for i in range(len(dims)-1)
            ]
        )

class HGCN(Encoder):
    """
    Hyperbolic Graph Convolutional Network (HGCN).

    Parameters:
    -----------
    args : Namespace
        A namespace object containing hyperparameters for the HGCN, including:
            - manifold (str): The type of manifold to use for hyperbolic geometry.
            - num_layers (int): Number of layers in the HGCN.
            - dropout (float): Dropout rate for regularization.
            - bias (bool): Whether to include a bias term in the convolutional layers.

    Attributes:
    -----------
    manifold : Manifold
        The manifold object that defines the hyperbolic space used in the model.
    layers : nn.Sequential
        A sequential container that holds the HyperbolicGraphConvolution layers. 
        Each layer transforms the input node features according to the specified 
        dimensions and activation functions.
    curvatures : list of float
        A list of curvature parameters for each layer, defining the hyperbolic
        geometry characteristics.

    Methods:
    --------
    encode(x, adj):
        Encodes the input node features and adjacency matrix by passing 
        them through the hyperbolic graph convolution layers.
    """

    def __init__(self, args):
        super(HGCN, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0, "Number of layers must be greater than 0."
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        
        self.layers = nn.Sequential(
            *[
                hyp_layers.OriginHyperbolicGraphConvolution(
                    manifold=self.manifold,
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    c_in=self.curvatures[i],
                    c_out=self.curvatures[i+1],
                    dropout=args.dropout,
                    act=acts[i],
                    use_bias=args.bias,
                    use_att=args.use_att,
                    local_agg=args.local_agg
                )
                for i in range(len(dims) - 1)
            ]
        )

    def encode(self, x, adj):        
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)

class sHGCN(Encoder):
    """
    Simplified Hyperbolic Graph Convolutional Network (sHGCN).

    Parameters:
    -----------
    args : Namespace
        A namespace object containing hyperparameters for the sHGCN, including:
            - manifold (str): The type of manifold to use for hyperbolic geometry.
            - num_layers (int): Number of layers in the HGCN.
            - dropout (float): Dropout rate for regularization.
            - bias (bool): Whether to include a bias term in the convolutional layers.

    Attributes:
    -----------
    manifold : Manifold
        The manifold object that defines the hyperbolic space used in the model.
    layers : nn.Sequential
        A sequential container that holds the HyperbolicGraphConvolution layers. 
        Each layer transforms the input node features according to the specified 
        dimensions and activation functions.
    curvatures : list of float
        A list of curvature parameters for each layer, defining the hyperbolic
        geometry characteristics.

    Methods:
    --------
    encode(x, adj):
        Encodes the input node features and adjacency matrix by passing 
        them through the hyperbolic graph convolution layers.
    """

    def __init__(self, args):
        super(sHGCN, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0, "Number of layers must be greater than 0."
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)

        self.layers = nn.Sequential(
            *[
                hyp_layers.HyperbolicGraphConvolution(
                    manifold=self.manifold,
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    c=self.curvatures[i],
                    dropout=args.dropout,
                    act=acts[i],
                    use_bias=args.bias
                )
                for i in range(len(dims) - 1)
            ]
        )

    def encode(self, x, adj):        
        return super(sHGCN, self).encode(x, adj)

class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)