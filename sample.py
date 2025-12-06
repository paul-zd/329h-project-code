import numpy as np
from typing import List, Tuple

def sigma(n):
  """
  Docstring for sigma
  
  :param n: Sigmoid function
  """
  return 1/(1+np.exp(-n))

class User:
  def __init__(self, gt:np.array, sigma=0.05):
    self.scores = gt + np.random.normal(0,scale=sigma, size=gt.shape)
    
  def sample(self, idx1, idx2)->bool:
    """
    Sample a user
    
    :param self: The user
    :param idx1: First index to sample
    :param idx2: Second index to sample
    :return: if item 1 beats item 2 when sampling this user
    :rtype: bool
    """
    return np.random.random() < sigma(self.scores[idx1]-self.scores[idx2])

class NormalUser(User):
  def __init__(self, gt:np.array, sigma=0.05):
    super().__init__(gt, sigma)

class SchemingUser(User):
  def __init__(self, gt, S:list, sigma=0.05):
    super().__init__(gt, sigma)
    for x in S:
      self.scores[x]+=1000

class CarelessUser(User):
  def __init__(self, gt, sigma=0.05):
    super().__init__(gt, sigma)
    self.scores = self.scores*0

class MaliciousUser(User):
  def __init__(self, gt, sigma=0.05):
    super().__init__(gt, sigma)
    self.scores = -self.scores

def initialize(N, M, scale=0.5, alpha=0, ssize=0.3, sset = None, sigma=0.05)->Tuple[List[User], List, int]:
  """
  Initialize a environment with some behaving and scheming users. Scheming users at the front of the list.
  
  :param N: Number of users
  :param M: Number of items
  :param scale: Scale of preferences (low will be more noisy)
  :param alpha: Fraction of scheming users
  :param ssize: Relative size of scheming set if not otherwise given
  :param sset: Preset for the scheming set
  :param sigma: Individual noise for each user
  :return: List of user, ground truth preferences, number of schemers
  :rtype: Tuple[List[User], List, int]
  """
  gt = np.arange(M)*scale
  things = []
  if sset is None:
    sset = np.random.choice(np.arange(M),int(M*ssize),replace=False)
  #print("Scheming set is ",sset)
  nscheme = 0
  for i in range(N):
    if i < N*alpha:
      things.append(SchemingUser(gt, sset, sigma))
      nscheme+=1
    else:
      if (i-1)<N*alpha:
        pass
        #print(str(i)+" scheming users out of "+str(N))
      things.append(NormalUser(gt, sigma))
  return things, gt, nscheme



