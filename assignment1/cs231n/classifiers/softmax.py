import numpy as np
from random import shuffle


def softmax_for_single_sample(scores):
    """
    Normalize scores to softmax probabilities
    
    Inputs:
      - scores: A numpy array of shape (C,) containing scores of one single example.

    Returns a numpy array of shape (C,):
      - nomalized probabilities    
    """
    scores -= np.max(scores)
    return np.exp(scores) / np.sum(np.exp(scores))

def softmax_for_batch(scores):
    """
    Normalize scores to softmax probabilities
    
    Inputs:
      - scores: A numpy array of shape (N,C) containing scores of all training examples.

    Returns a numpy array of shape (N,c):
      - nomalized probabilities of all training examples   
    """
    scores -= np.max(scores, axis=1).reshape(-1,1)
    return np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1)
    

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_train = X.shape[0]
  num_class = W.shape[1]
  dim = X.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)    
    curSoftmax = softmax_for_single_sample(scores)
    loss += -1*np.log(curSoftmax[y[i]])
    for j in xrange(num_class):
        if j == y[i]:
            dW[:,j] += (curSoftmax[j]-1)*X[i] 
        else:
            dW[:,j] += curSoftmax[j]*X[i]
   
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
    
  scores = X.dot(W)
  softmax = softmax_for_batch(scores)
    
  loss = np.sum(-1*np.log(softmax[range(num_train),y])) / num_train + 0.5*reg*np.sum(W*W)
  softmax[range(num_train),y] -= 1
  dW = X.T.dot(softmax) / num_train + reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

