from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *

class FullyConnectedNet(object):
    """A fully-connected neural network with an arbitrary number of hidden layers, ReLU
    nonlinearities, and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be
    
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch/layer normalization and dropout are optional and the {...} block is repeated L - 1 times.
    Learnable parameters are stored in the self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout_keep_ratio=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength. If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using this datatype. float32 is faster but less accurate, so you should use float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This will make the dropout layers deterministic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network, storing all values in the self.params dictionary.
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i + 1])
            self.params['b%d' % (i + 1)] = np.zeros(layer_dims[i + 1])
            if self.normalization and i < len(hidden_dims):
                self.params['gamma%d' % (i + 1)] = np.ones(layer_dims[i + 1])
                self.params['beta%d' % (i + 1)] = np.zeros(layer_dims[i + 1])

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = None
        caches = {}
        dropout_caches = {}
        out = X

        for i in range(1, self.num_layers):
            W, b = self.params['W%d' % i], self.params['b%d' % i]
            out, cache = affine_forward(out, W, b)
            caches['affine%d' % i] = cache
            if self.normalization == 'batchnorm':
                gamma, beta = self.params['gamma%d' % i], self.params['beta%d' % i]
                out, cache = batchnorm_forward(out, gamma, beta, self.bn_params[i - 1])
                caches['batchnorm%d' % i] = cache
            out, cache = relu_forward(out)
            caches['relu%d' % i] = cache
            if self.use_dropout:
                out, cache = dropout_forward(out, self.dropout_param)
                dropout_caches['dropout%d' % i] = cache

        W, b = self.params['W%d' % self.num_layers], self.params['b%d' % self.num_layers]
        scores, cache = affine_forward(out, W, b)
        caches['affine%d' % self.num_layers] = cache

        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        for i in range(self.num_layers):
            W = self.params['W%d' % (i + 1)]
            loss += 0.5 * self.reg * np.sum(W ** 2)

        dout, dW, db = affine_backward(dscores, caches['affine%d' % self.num_layers])
        grads['W%d' % self.num_layers] = dW + self.reg * self.params['W%d' % self.num_layers]
        grads['b%d' % self.num_layers] = db

        for i in reversed(range(1, self.num_layers)):
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches['dropout%d' % i])
            dout = relu_backward(dout, caches['relu%d' % i])
            if self.normalization == 'batchnorm':
                dout, dgamma, dbeta = batchnorm_backward(dout, caches['batchnorm%d' % i])
                grads['gamma%d' % i] = dgamma
                grads['beta%d' % i] = dbeta
            dout, dW, db = affine_backward(dout, caches['affine%d' % i])
            grads['W%d' % i] = dW + self.reg * self.params['W%d' % i]
            grads['b%d' % i] = db

        return loss, grads
