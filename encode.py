from cv2 import split
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import *
from model import *
from config import *
from dataset import *

def sine_encode(x):
    # x in [-inf, inf], y in [0,1], len(y)=16
    ans = []
    for i in range(-4,12):
        freq = 2**i
        ans.append(np.sin(x*freq)*0.5+0.5)
    return np.array(ans,dtype=np.float64)

def sine_encode_3d(p):
    ans = []
    ans += list(sine_encode(p[0]))
    ans += list(sine_encode(p[1]))
    ans += list(sine_encode(p[2]))
    return np.array(ans,dtype=np.float64)

def gaussian_nonorm(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def oneblob_encode(x):
    # x in [0,1], y in [0,1], len(y)=8
    ans = []
    for i in range(0,8):
        u = (i+0.5)/8
        s = 1/8
        ans.append(gaussian_nonorm(x,u,s))
    return np.array(ans,dtype=np.float64)

def oneblob_encode_2d(p):
    ans = []
    ans += list(oneblob_encode(p[0]))
    ans += list(oneblob_encode(p[1]))
    return np.array(ans,dtype=np.float64)
