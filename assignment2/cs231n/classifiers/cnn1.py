import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:
  
  INPUT -> [[CONV -> SPATIAL_BATCH_NORM? -> RELU]*I -> POOL2X2?]*J -> [FC -> RELU]*K -> FC
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """    
    
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], 
               filter_size=[3], hidden_dim=[100], num_classes=10, 
               pool_interval=1, weight_scale=1e-3, reg=0.0, 
               use_batchnorm=False, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in each convolutional layer
    - filter_size: Size of filters to use in each convolutional layer
    - hidden_dim: Number of units to use in each fully-connected hidden layer
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
    self.pool_interval = pool_interval
    
    if use_batchnorm:
        self.bn_param = {}
    
    ############################################################################
    # Initialize weights and biases for the three-layer convolutional          #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights, biases, gammas and betas should be stored in the dictionary #
    # self.params.                                                             #
    ############################################################################  
    C, H, W = input_dim
    num_conv = len(num_filters)
    num_affine = len(hidden_dim)  # Except the last affine layer which output scores
    
    self.num_conv = num_conv
    self.num_affine = num_affine
    
    assert num_conv % pool_interval == 0
    assert num_conv == len(filter_size)
    
    num_pool = num_conv / pool_interval
    Hp, Wp = H / (2**num_pool), W / (2**num_pool)
    
    # Initialize conv parameters and spatial batchnorm parameters
    for i in xrange(num_conv):
        if i==0:
            num_C = C
        else:
            num_C = num_filters[i-1]
        self.params['W'+str(i+1)] = weight_scale *\
            np.random.randn(num_filters[i], num_C, filter_size[i], filter_size[i]) 
        self.params['b'+str(i+1)] = np.zeros(num_filters[i])  
        
        if use_batchnorm:
            self.bn_param[i] = {'mode': 'train'}
            self.params['gamma'+str(i+1)] = np.ones(num_filters[i])
            self.params['beta'+str(i+1)] = np.zeros(num_filters[i])

    # Initialize affine(FC) parameters 
    for j in xrange(num_affine):
        if j==0:
            prev_dim = num_filters[-1]*Hp*Wp
        else : 
            prev_dim = hidden_dim[j-1]
        
        self.params['W'+str(j + num_conv + 1)] = weight_scale * \
                                    np.random.randn(prev_dim, hidden_dim[j]) 
        self.params['b'+str(j + num_conv + 1)] = np.zeros(hidden_dim[j])
     
    # Initialize the last affine layer parameters
    self.params['W'+str(num_affine + num_conv + 1)] = weight_scale * \
                                    np.random.randn(hidden_dim[-1], num_classes) 
    self.params['b'+str(num_affine + num_conv + 1)] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.   
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    scores = None
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    cache = {}                                                       
    out = X  # Output of each layer 
    
    ############################################################################
    # Implement the forward pass for the three-layer convolutional net,        #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
         
    # Forward pass for the conv & pool layers
    for i in xrange(self.num_conv):
        
        # Compute the output of current conv layer
        Wi = self.params['W'+str(i+1)] 
        bi = self.params['b'+str(i+1)]                                                   
        filter_size = Wi.shape[2]  # (num_filters, C, filter_size, filter_size)                                     
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}  # Full-size conv                                                
         
        # Compute the output of current [conv - bn? - relu] layer
        if self.use_batchnorm:
            gamma = self.params['gamma'+str(i+1)]
            beta = self.params['beta'+str(i+1)]
            out, cache['conv'+str(i+1)] = \
                    conv_bn_relu_forward(out, Wi, bi, gamma, beta, conv_param, self.bn_param[i]) 
        else :
            out, cache['conv'+str(i+1)] = \
                    conv_relu_forward(out, Wi, bi, conv_param) 
        
        # Compute the output of current pool layer if needed
        if (i+1) % self.pool_interval == 0:
            out, cache['pool'+str((i+1)/self.pool_interval)] = \
                    max_pool_forward_fast(out, pool_param)
                                
                
    # Forward pass for the affine layers
    for j in xrange(self.num_affine):
        Wj = self.params['W'+str(self.num_conv + j + 1)] 
        bj = self.params['b'+str(self.num_conv + j + 1)]
        out, cache['affine'+str(j+1)] = affine_relu_forward(out, Wj, bj)
        
    # Forward pass for the last affine layer
    W_last = self.params['W'+str(self.num_conv + self.num_affine + 1)] 
    b_last = self.params['b'+str(self.num_conv + self.num_affine + 1)]
    scores, cache['affine'+str(self.num_affine + 1)] = affine_forward(out, W_last, b_last)
                
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################                                                 
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the three-layer convolutional net,       #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores,y)
    
    # Backward pass for the last affine layer
    dW_last, db_last, cache_last = 0, 0, cache['affine'+str(self.num_affine+1)]
    dout, dW_last, db_last = affine_backward(dscores, cache_last)
    grads['W'+str(self.num_conv + self.num_affine + 1)] = dW_last
    grads['b'+str(self.num_conv + self.num_affine + 1)] = db_last
    
    # Backward pass for other affine layers
    for j in range(self.num_affine)[::-1]:
        dWj, dbj, affine_cache = 0, 0, cache['affine'+str(j+1)]
        dout, dWj, dbj = affine_relu_backward(dout, affine_cache)
        grads['W'+str(self.num_conv + j + 1)] = dWj
        grads['b'+str(self.num_conv + j + 1)] = dbj
        
    # Backward pass for conv layers
    for i in range(self.num_conv)[::-1]:
        
        # The backward pass for pool layer if needed
        if (i+1) % self.pool_interval == 0:
            pool_cache = cache['pool'+str((i+1)/self.pool_interval)]
            dout = max_pool_backward_fast(dout, pool_cache)
            
        # The backward pass for current [conv - bn? - relu] layer
        conv_cache = cache['conv'+str(i+1)]
        if self.use_batchnorm:              
            dout, dWi, dbi, dgamma, dbeta = conv_bn_relu_backward(dout, conv_cache)
            grads['gamma'+str(i+1)] = dgamma
            grads['beta'+str(i+1)] = dbeta 
        else :
            dout, dWi, dbi = conv_relu_backward(dout, conv_cache)
        grads['W'+str(i+1)] = dWi
        grads['b'+str(i+1)] = dbi        
                                                                
    # Count the reg
    for L in xrange(self.num_conv + self.num_affine + 1):
        WL = self.params['W'+str(L+1)]
        loss += 0.5 * self.reg * np.sum(WL**2)
        grads['W'+str(L+1)] += self.reg * WL

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
