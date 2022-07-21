from cv2 import split
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from tqdm import tqdm
from config import *

def remove_last_empty(s):
    if len(s)>0 and (s[-1]=='\n' or s[-1]==''):
        return s[:-1]
    return s

class DatasetFromTxtFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        f = open(filename)
        lines = f.readlines()
        splited = [list(map(float,remove_last_empty(list(i.split(' '))))) for i in lines]
        self.x = np.array([i[:dim_raw_input] for i in splited], dtype=np.float32)
        self.y = np.array([i[dim_raw_input:] for i in splited], dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DatasetFromStr(torch.utils.data.Dataset):
    def __init__(self, strs):
        lines = strs
        splited = [list(map(float,remove_last_empty(list(i.split(' '))))) for i in lines]
        self.x = np.array([i[:dim_raw_input] for i in splited], dtype=np.float32)
        self.y = np.array([i[dim_raw_input:] for i in splited], dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]