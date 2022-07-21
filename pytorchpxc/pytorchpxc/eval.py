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
from encode import *
import sys

model = None
optimizer = None

def load():
    global model
    global optimizer
    model = torch.load(model_filename)
    optimizer = torch.load(optimizer_filename)
    print("[PYINFO]","load",model_filename)

    if use_cuda: model=model.cuda()

def eval(sss):
    global model
    global optimizer
    print("[PYINFO]","eval",sss)
    eval_dataset = DatasetFromStr(sss.split("\n"))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss()

    for batch_idx, (b_x, b_y) in enumerate(eval_dataloader):
        x = b_x.view(-1, dim_raw_input)
        x = my_encoding_batch(x)
        x = torch.tensor(x)
        print(x.shape)
        if use_cuda: x=x.cuda()
        out = model(x)
        return float(out.item())


if __name__ == "__main__":
    load()
    eval(sys.argv[1])