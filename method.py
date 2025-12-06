

import sample
import torch as t
import torch.nn.functional as F
from typing import List, Tuple, Dict

gd = "cuda" if t.cuda.is_available() else "cpu"
print(gd)



def logLikelihoodSplit(obs:t.Tensor, th:t.Tensor)->t.Tensor:
  """
  Compute log likelihood log p(obs|theta) for our theta estimate (batched)
  
  :param obs: observtions shape B... x M x M
  :type obs: t.Tensor 
  :param th: theta shape B... x M
  :type th: t.Tensor
  :return: the log likelihood for each batch
  """
  return (obs*F.logsigmoid(th.unsqueeze(-1)-th.unsqueeze(-2))).sum(dim=(-1,-2))


def getSamplePairs(obs:t.Tensor, th:t.Tensor, tau=0.002)->t.Tensor:
  """
  Generate sample pairs for each user that minimizes uncertainty
  
  :param obs: observations for each user shape B... x M x M
  :type obs: t.Tensor
  :param th: theta estimate for each user shape B... x M
  :type th: t.Tensor
  :param tau: temperature for the multinomial sampling
  :return: a list of sample indices (flattened as M*i+j)
  """
  with t.no_grad():
    #build fisher information
    pEst = t.sigmoid(th.unsqueeze(-1)-th.unsqueeze(-2))
    nComp = obs+obs.transpose(dim0=-1, dim1=-2)
    p_oneminus_p = pEst*(1-pEst)
    finfo = nComp*p_oneminus_p

    #build the uncertainty matrix
    happrox = finfo.sum(dim=1)
    happroxinv = 1/(happrox+0.000001)
    uncertainty = happroxinv.unsqueeze(-1) + happroxinv.unsqueeze(-2)

    score:t.Tensor = p_oneminus_p*uncertainty
    #don't sample the diagonal
    view = score.diagonal(dim1=-1,dim2=-2)
    view.copy_(t.tensor([-t.inf]).to(score.device).reshape([1]*(score.dim()-1)))
    #print(score)

    #choose an index to sample
    probs = t.softmax(score.flatten(start_dim=-2)/tau, dim=-1)
    idx = t.multinomial(probs,1)
    return idx.to("cpu")
def getNaiveSamplePairs(obs:t.Tensor, th:t.Tensor,tau=1)->t.Tensor:
  """
  Naively generate sample pairs, choosing samples that would maximize information gain
  
  :param obs: observations for each user shape B... x M x M
  :type obs: t.Tensor
  :param th: theta estimate for each user shape B... x M
  :type th: t.Tensor
  :param tau: temperature for the multinomial sampling
  :return: a list of sample indices (flattened as M*i+j)
  """
  pEst = t.sigmoid(th.unsqueeze(-1)-th.unsqueeze(-2))
  p_oneminus_p = pEst*(1-pEst)
  #expected information
  score = t.zeros_like(obs)+p_oneminus_p
  view = score.diagonal(dim1=-1,dim2=-2)
  view.copy_(t.tensor([-t.inf]).to(score.device).reshape([1]*(score.dim()-1)))
  probs = t.softmax(score.flatten(start_dim=-2)/tau, dim=-1)
  idx = t.multinomial(probs,1)
  return idx.to("cpu")

def sampleUsers(obs:t.Tensor, users:List[sample.User], samples):
  """
  Sample each user according to the given sample index. Should 
  
  :param obs: shape |users|... x M x N
  :param users: list of users
  :type users: List[sample.User]
  :param samples: tensor of sample indices (flattened)
  """
  with t.no_grad():
    for i,user in enumerate(users):
      x,y = divmod(samples[i].item(),obs.shape[-1])
      if user.sample(x,y):
        obs[i,x,y]+=1
      else:
        obs[i,y,x]+=1




def mle(obs:t.Tensor, tInitial=None, iters=20, print_freq=None):
  """
  Find the theta that maximizes the likelihood of the observations (batched)
  
  :param obs: observations B... x M x M
  :type obs: t.Tensor
  :param tInitial: Optional tensor with initial state for the start of the process
  :param iters: number of gradient steps to take
  """
  #if there is no initial t, zero-initiailize
  tOptim = None
  if tInitial is None:
    tOptim = t.zeros(obs.shape[:-1]).to(obs.device).detach().clone().requires_grad_(True)
  else:
    tOptim = tInitial.detach().clone().requires_grad_(True)

  
  optim = t.optim.AdamW([tOptim],lr=0.02)
  for iter in range(iters):
    optim.zero_grad()
    loss = -logLikelihoodSplit(obs,tOptim).sum()
    if print_freq is not None and iter%print_freq==0:
      print(iter, " ", loss.item())
    loss.backward()
    optim.step()
  with t.no_grad():
    tOptim -= tOptim.mean(dim=-1, keepdim=True)
  return tOptim.detach()


def softtopk(values:t.Tensor, k):
  """
  Compute the soft topk
  
  :param values: Tensor of values
  :type values: t.Tensor
  :param k: k
  """
  with t.no_grad():
    origShape = values.shape
    smax = t.softmax(values.flatten(), dim=0)*k
    for i in range(10):
      smax = t.clip(smax,0,1)
      inds = smax!=1
      ones = (smax==1).sum()
      #s = smax[inds].sum()
      #smax[inds] = smax[inds]*((k-ones)/(s+0.00001))
      smax[inds] = t.softmax(values[inds], dim=0)*(k-ones)
    return smax.reshape(origShape).clip(0,1)
  
#print(softtopk(t.tensor([1,5,6,30,21],dtype=t.float),3))
def remap(value, lowOld=0, highOld=1, lowNew=0, highNew=1):
  """
  Simple linear remap
  """
  return (value-lowOld)/(highOld-lowOld)*(highNew-lowNew)+lowNew




    




t.set_printoptions(sci_mode=False, linewidth=160)

def runTrial(
    N, M=5, SFrac = 0.2, Afrac = None, sigma = 0.05, scale=0.5, verbose=False, naive=False, iters=100, endDiscrim=10, sset = None, interval = None
  )->Tuple[Tuple[t.Tensor,t.Tensor], t.Tensor, Dict]:
  """
  Docstring for runTrial
  
  :param N: Number of users
  :param M: Number of items
  :param SFrac: Fraction of scheming users
  :param Afrac: Fraction of items that will be weighted (k in topk is N*Afrac)
  :param sigma: Noise for user preferences
  :param scale: Scale of ground truth preferences
  :param naive: Whether to run the naive experiment
  :param iters: Number of samples per user
  :param endDiscrim: Inverse 'temperature' for maxSoftk by the end of the process
  :param sset: Scheming set
  :param interval: How often to recompute weights
  :return: The estimates, ground truth as well as training information for graphing
  :rtype: Tuple[Tuple[Tensor, Tensor], Tensor, Dict]
  """
  if Afrac is None:
    if SFrac<=0.1:
      Afrac = 1-2*SFrac
    elif SFrac<=0.25:
      Afrac = 1-1.5*SFrac
    else:
      Afrac = 1-1.2*SFrac
  if sset is None:
    sset = [M//2]
  if interval is None:
    interval = max(iters//10,1)
  users, gt, numscheme = sample.initialize(N,M,alpha=SFrac,sigma=sigma,scale=scale,sset=sset)

  #We use laplace smoothing for the initial weights to improve training stability. Subtracted at the end of the process.
  initial = (t.ones((N,M,M), requires_grad=False)-t.eye(M).unsqueeze(0)).to(gd)*0.5
  uweights = t.ones(N).to(gd)*Afrac
  observ = initial.clone()


  #the individual theta estimates
  t_ = t.zeros(observ.shape[:-1],requires_grad=False).to(gd)
  #our main theta estimate
  tmain = t.zeros(observ.shape[1:-1],requires_grad=False).to(gd)
  iScores = []
  eScores = []
  wScores = []

  for ep in range(iters):
    t_ = mle(observ,t_)
    #observNorm = observ/(observ.sum(dim=(-1,-2),keepdim=True)+0.000001)
    weightedObserv = ((observ)*uweights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0,keepdim=True)

    if ep%interval == interval-1 and not naive:
      pEst = t.sigmoid(t_.unsqueeze(-1)-t_.unsqueeze(-2))

      #fit theta hat to our weighted observations
      tmain = mle(weightedObserv[0], tmain, iters=50)

      #compute intrinsic score
      intr = logLikelihoodSplit(pEst, tmain)/M
      intr = intr-intr.mean()

      #compute extrinsic score
      extr = logLikelihoodSplit(weightedObserv, t_)/observ.sum()
      extr = extr-extr.mean()


      uweights = softtopk((intr+extr)*remap(ep,highOld=iters,highNew=endDiscrim),N*Afrac)
      iScores.append(intr.to("cpu"))
      eScores.append(extr.to("cpu"))
      wScores.append(uweights.to("cpu"))
      if verbose:
        print("intrinsic ", intr)
        print("extrinsic",extr)
        print("weights", uweights)
        print()

    #Build our sample pairs and sample our users
    if naive:
      tmain = mle(weightedObserv[0], tmain)
      #samples = getSamplePairs(observ, t_)
      samples = getNaiveSamplePairs(t.stack([weightedObserv[0]]*N), t.stack([tmain]*N))
    else:
      samples = getSamplePairs(observ+weightedObserv/N/10, t_)
    #i,j = divmod(samples[0].item(), M)
    sampleUsers(observ,users,samples)

  #Compute our final estimation
  observ = observ-initial
  weightedObserv = ((observ)*uweights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0,keepdim=False)
  
  estimate = [None,None]
  estimate[0] = mle(weightedObserv,iters=100).to("cpu")

  nweights = None
  if not naive:
    #Compute an additional estimation using a hard topk
    values, indices = t.topk(uweights, int(round(N*Afrac)))
    nweights = t.zeros_like(uweights)
    nweights[indices]=1
    #print(nweights[:numscheme])
    #print(nweights[numscheme:])
    weightedObserv = ((observ)*nweights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0,keepdim=False)
    estimate[1] = mle(weightedObserv,iters=100).to("cpu")
  
  return estimate,gt, {
    "intrinsic":[] if naive else t.stack(iScores),
    "extrinsic":[] if naive else t.stack(eScores),
    "weights":[] if naive else t.stack(wScores), 
    "final_weights":uweights,
    "hard_weights":nweights,
    "schemers":numscheme, 
    "users":users
  }

if __name__=="__main__":
  estimate, truth, stats = runTrial(40,verbose=False, scale=0.25, iters=40, alpha=0.2)
  print(estimate)
  print(truth)
  print(stats["final_weights"])


# uweights = t.ones(N).to(gd)*Afrac
# observ = initial.clone()

# t_ = t.zeros(observ.shape[:-1],requires_grad=False).to(gd)

# for epoch in range(300):
#   pairs = getSamplePairs(observ,t_)
#   sampleUsers(observ, users, pairs)
#   t_ = mle(observ, t_)
  

# print(mle(observ.sum(axis=0)))
# print(observ)

#def calculateImpact()