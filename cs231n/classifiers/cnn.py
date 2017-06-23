import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False, # to correctly implement batchnorm, split the layers.1
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) *weight_scale
    first_dim = num_filters*input_dim[1]*input_dim[2]/4 # runtime setting half input image dimensions after the conv-relu-maxpool layer.  
    self.params['W2'] = np.random.randn(first_dim, hidden_dim) *weight_scale
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) *weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)
    
    if use_batchnorm:
      self.params['gamma1'] = np.ones(first_dim)
      self.params['beta1'] = np.zeros(first_dim)
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2} # preserves input dimension

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2} # halfs the input dimensions

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    next_input, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    if self.use_batchnorm:
      next_input, sbn_cache = spatial_batchnorm_forward(next_input, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    next_input, ar_cache = affine_relu_forward(next_input, W2, b2)
    if self.use_batchnorm:
      next_input, bn_cache = batchnorm_forward(next_input, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
    scores, a_cache = affine_forward(next_input, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))    
    
    dout, grads['W3'], grads['b3'] = affine_backward(dout, a_cache)
    if self.use_batchnorm:
      dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, bn_cache)
    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, ar_cache)
    if self.use_batchnorm:
      dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_bacward(dout, sbn_cache)
    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, crp_cache)
    
    grads['W1'] += W1*self.reg
    grads['W2'] += W2*self.reg
    grads['W3'] += W3*self.reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

class FourLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=(32, 32), filter_size=(7, 3),
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False, # to correctly implement batchnorm, split the layers.1
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.randn(num_filters[0], input_dim[0], filter_size[0], filter_size[0]) *weight_scale
    self.params['W2'] = np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1]) *weight_scale
    second_dim = num_filters[1]*input_dim[1]*input_dim[2]/4 # runtime setting half input image dimensions after the conv-relu-maxpool layer.  
    self.params['W3'] = np.random.randn(second_dim, hidden_dim) *weight_scale
    self.params['W4'] = np.random.randn(hidden_dim, num_classes) *weight_scale
    self.params['b1'] = np.zeros(num_filters[0])
    self.params['b2'] = np.zeros(num_filters[1])
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['b4'] = np.zeros(num_classes)
    
    if use_batchnorm:
      self.params['gamma2'] = np.ones(first_dim)
      self.params['beta2'] = np.zeros(first_dim)
      self.params['gamma3'] = np.ones(hidden_dim)
      self.params['beta3'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2} # preserves input dimension
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W2.shape[2]
    conv_param2 = {'stride': 1, 'pad': (filter_size - 1) / 2} # preserves input dimension

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2} # halfs the input dimensions

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    next_input, cr_cache = conv_relu_forward(X, W1, b1, conv_param1)
    next_input, crp_cache = conv_relu_pool_forward(next_input, W2, b2, conv_param2, pool_param)
    if self.use_batchnorm:
      next_input, sbn_cache = spatial_batchnorm_forward(next_input, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    next_input, ar_cache = affine_relu_forward(next_input, W3, b3)
    if self.use_batchnorm:
      next_input, bn_cache = batchnorm_forward(next_input, self.params['gamma2'], self.params['beta2'], self.bn_params[1])
    scores, a_cache = affine_forward(next_input, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) + np.sum(W4*W4))
    
    dout, grads['W4'], grads['b4'] = affine_backward(dout, a_cache)
    if self.use_batchnorm:
      dout, grads['gamma2'], grads['beta2'] = batchnorm_backward(dout, bn_cache)
    dout, grads['W3'], grads['b3'] = affine_relu_backward(dout, ar_cache)
    if self.use_batchnorm:
      dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_bacward(dout, sbn_cache)
    dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, crp_cache)
    dout, grads['W1'], grads['b1'] = conv_relu_backward(dout, cr_cache)
    
    grads['W1'] += W1*self.reg
    grads['W2'] += W2*self.reg
    grads['W3'] += W3*self.reg
    grads['W4'] += W4*self.reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
