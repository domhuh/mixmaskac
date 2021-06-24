import torch.optim as optim
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random 
import gym
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import multiprocessing as mp
from functools import partial
from IPython.display import clear_output
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import copy
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control.cartpole import CartPoleEnv

class nCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super(nCartPoleEnv,self).__init__()
        self.gravity = 9.8
        self.length = random.random() *2 # actually half the pole's length
        self.masspole = random.random()
        self.masscart = random.random()*5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
    def reset(self):
        obs = super().reset()
        self.length = random.random() *2
        self.masspole = random.random()
        self.masscart = random.random()*5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        return obs
        
def conv(): return nn.Sequential(nn.Conv1d(2,2,kernel_size = 3, padding=1),nn.LeakyReLU(), nn.BatchNorm1d(2))

def information_radius(P,Q):
    mu = (P.loc+Q.loc)/2
    if type(P)==MultivariateNormal:
        C = (P.scale_tril+Q.scale_tril)/4
        M = MultivariateNormal(mu,C)
    else:
        std = ((P.variance+Q.variance)/4).sqrt()
        M = Normal(mu,std)
    return (kl_divergence(P,M)+kl_divergence(Q,M))/2

def j_divergence(P,Q):
    return (kl_divergence(P,Q)+kl_divergence(Q,P))/2

def ag_divergence(P,Q):
    mu = (P.loc+Q.loc)/2
    if type(P)==MultivariateNormal:
        C = (P.scale_tril+Q.scale_tril)/4
        M = MultivariateNormal(mu,C)
    else:
        std = ((P.variance+Q.variance)/4).sqrt()
        M = Normal(mu,std)
    return (kl_divergence(M,P)+kl_divergence(M,Q))/2

class SILogStd(nn.Module):
    def __init__(self,ni,nh):
        super(SILogStd,self).__init__()
        self.value = nn.Parameter(torch.ones(nh)*1e-3)
    def forward(self,x): return self.value
    
class SDLogStd(nn.Module):
    def __init__(self,ni,nh):
        super(SDLogStd,self).__init__()
        self.M = nn.Sequential(nn.Linear(ni,nh),
                               nn.LeakyReLU(),
                               nn.Linear(nh,nh))
    def forward(self,x):
        return self.M(x)
    
def reflect_major_diag(x):
    h,w = x.shape #assume square matrix
    y = x*torch.tensor(np.tri(h))
    return (y+y.T)/(torch.ones(h,w)+torch.eye(w))

class MixerMLP(nn.Module):
    def __init__(self,nh):
        super(MixerMLP,self).__init__()
        self.mlp = nn.ModuleList([nn.Linear(2,2),
                                  nn.Linear(nh,nh)])
        self.norm = nn.LayerNorm(nh)
    def forward(self,x):
        return self.norm(self.mlp[1](self.mlp[0](x.T).relu().T).relu())
        
class SICovarianceMatrix(nn.Module):
    def __init__(self,ni,nh):
        super(SICovarianceMatrix,self).__init__()
        self.value = nn.Parameter(torch.eye(nh))
        self.mask = torch.tensor(np.tri(nh),requires_grad=False)
        self.nh = nh
    def forward(self,x):
        L = self.value*self.mask
        L = torch.matmul(L,L.T)*self.mask+torch.eye(self.nh)
        return L.float()
    
class SDCovarianceMatrix(nn.Module):
    def __init__(self,ni,nh):
        super(SDCovarianceMatrix,self).__init__()
        self.L = nn.Sequential(nn.Linear(ni,nh),
                               nn.ReLU())
        self.mask = torch.tensor(np.tri(nh),requires_grad=False)
        self.nh = nh
        init_weights(self)
    def forward(self,x):
        L = self.L(x).unsqueeze(-1)
        L = torch.matmul(L,L.T)
        L = L*self.mask+torch.eye(self.nh)
        return torch.matmul(L,L.T).float() 
        
    
def set_seed(env=None,proc_num=1):
    random.seed(proc_num)
    torch.manual_seed(proc_num)
    np.random.seed(proc_num)
    if env is not None: env.seed(proc_num)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def batch_term(x,y,term=information_radius):
    score = 0
    for x_,y_ in zip(x,y):
        score += term(x_,y_).sum()
    return score/len(x)

def reshape(x):
    if len(x.shape)==1:
        x = x.unsqueeze(0) #singular batch 
    return x
    
class Attention(nn.Module):
    def __init__(self,ni):
        super(Attention,self).__init__()
        self.key = nn.Linear(ni,ni)
        self.value = nn.Linear(ni,ni)
        self.query = nn.Linear(ni,ni)
    def forward(self,k,v,q,negate = False):
        [k,v,q] = [reshape(x) for x in [k,v,q]]
        mask = torch.bmm(self.query(q).unsqueeze(-1),self.key(k).unsqueeze(-1).transpose(1,-1)).softmax(1)
        if negate:
            mask = F.normalize(1-mask,1)
            return torch.bmm(mask, v.unsqueeze(-1)).squeeze()
        return torch.bmm(mask, v.unsqueeze(-1)).squeeze()
    
class LatentQuery(nn.Module):
    def __init__(self, nh):
        super(LatentQuery,self).__init__()
        self.query = nn.Parameter(torch.ones(nh).float())
    def forward(self,x):
        return self.query