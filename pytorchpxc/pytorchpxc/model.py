from cv2 import split
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from tqdm import tqdm
from config import *

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, dim_output, bias=True)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x