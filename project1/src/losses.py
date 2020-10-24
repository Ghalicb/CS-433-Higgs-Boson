import numpy as np

def compute_mse_loss(y, tx, w):
  """Mean Square Error loss

  Parameters
  ----------
  y : numpy array
    Targets vector (N,1)
  tx : numpy array
    Feature matrix (N,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  loss : float
  """  
  e = y - tx @ w
  return (1./(2*len(y)) * e.T @ e).item() 


def compute_mse_gradient(y, tx, w):
  """Mean Square Error gradient for a mini-batch of B points

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  sg : numpy array 
    gradient of mse loss (D,1)
  """ 
  B = len(y)
  e = y - tx @ w
  sg = -1./B * tx.T @ (e)
  return sg

def sigmoid(t):
  """apply the sigmoid function on t.

  Parameters
  ----------
  t: numpy array (B,1)

  Returns
  -------
  sig(t): numpy array (B,1)
  
  Warning: this method can return values of 0. and 1.
  """
  #numerically stable version without useless overflow warnings
  #uses 1/(1+e^-t) for positive values and 
  #uses e^t/(e^t+1) for negative values
  
  pos_mask = (t >= 0)
  neg_mask = (t < 0)
  z = np.zeros_like(t, dtype=np.float64)
  z[pos_mask] = np.exp(-t[pos_mask])
  z[neg_mask] = np.exp(t[neg_mask])
  top = np.ones_like(t, dtype=np.float64)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def compute_regularized_logistic_loss(y, tx, w, lambda_):
  """Regularized Logistic loss. 

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  lambda_ : float
    regularization constant
  
  Returns
  -------
  loss : float
    regularized negative log-likelihood   
  """  
  # Smallest positive value we can have
  min_value = np.nextafter(0, 1)
  
  # Avoiding numerical unstability i.e making pred in ]0,1[
  pred = sigmoid(tx @ w)
  pred[pred < min_value] = min_value
  
  one_minus_pred = 1 - pred
  one_minus_pred[one_minus_pred < min_value] = min_value
  
  reg_term = lambda_/2 * np.sum(w ** 2)
  loss = -(y.T @ np.log(pred) + (1 - y).T @ np.log(one_minus_pred)) + reg_term
  return loss.item() 


def compute_logistic_loss(y, tx, w):
  """Logistic loss. 

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  
  Returns
  -------
  loss : float
    negative log-likelihood
  """  
  return compute_regularized_logistic_loss(y, tx, w, lambda_ = 0)
  

def compute_regularized_logistic_gradient(y, tx, w, lambda_):
  """Regularized Logistic gradient for a mini-batch of B points.

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)
  lambda_ : float
    regularization constant  

  Returns
  -------
  gradient : numpy array 
    Gradient of logistic loss (D,1)
  """  
  reg_term = lambda_ * w
  return tx.T @ (sigmoid(tx @ w) - y) + reg_term


def compute_logistic_gradient(y, tx, w):
  """Logistic gradient for a mini-batch of B points.

  Parameters
  ----------
  y : numpy array
    Targets vector (B,1)
  tx : numpy array
    Feature matrix (B,D)
  w : numpy array
    weights vector (D,1)

  Returns
  -------
  gradient : numpy array 
    Gradient of logistic loss (D,1)
  """  
  return compute_regularized_logistic_gradient(y, tx, w, lambda_=0)
