import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_train = X.shape[0];
  num_class = W.shape[1];
  for i in range(num_train):
        S = X[i].dot(W);
        S -= np.max(S); 
        for j in range(num_class):
            p = np.exp(S[j]) / np.sum(np.exp(S)) # safe to do, gives the correct answer
            if j == y[i]:
                loss += -np.log(p);
                dW[:, j] += -(1 - p)*X[i].T;
            else:
                dW[:, j] += p*X[i].T;
     
  loss /= num_train;
  dW /= num_train;
    
  # Add regularization to the loss and gradient
  loss += reg * np.sum(W * W);
  dW += 2*reg*W;  
   
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
  num_train = X.shape[0];
  num_class = W.shape[1];
  S = X.dot(W); # N x C
  S = S.T - np.amax(S, axis=1); # C x N
  expS = np.exp(S);
  p = expS / np.sum(expS, axis=0); # C x N
  loss = -np.sum(np.log(p[y, range(num_train)]));  
  delta = p;
  delta[y, range(num_train)] = -(1 - p[y, range(num_train)]);
  dW = X.T.dot(delta.T);
     
  loss /= num_train;
  dW /= num_train;
    
  # Add regularization to the loss and gradient
  loss += reg * np.sum(W * W);
  dW += 2*reg*W;      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

