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


def train(NEW_MODEL = True):
    train_dataset = DatasetFromTxtFile("data.txt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if NEW_MODEL:
        print("[PYINFO] create new model")
        model = Model()
        model.apply(weights_init)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    else:
        print("[PYINFO] load model")
        model = torch.load(model_filename)
        optimizer = torch.load(optimizer_filename)

    if use_cuda: model=model.cuda()
    criterion = torch.nn.MSELoss()

    for epoch in range(n_epoch):
        train_loss = 0
        for batch_idx, (b_x, b_y) in tqdm(enumerate(train_dataloader)):
            x = b_x.view(-1, 1)
            if use_cuda: x=x.cuda()
            y_true = b_y
            if use_cuda: y_true=y_true.cuda()
            out = model(x)
            loss = criterion(out, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= len(train_dataloader)
        print("[PYINFO] loss =", train_loss)

    print("[PYINFO] saving")
    torch.save(model, model_filename)
    torch.save(optimizer, optimizer_filename)
    print("[PYINFO] end train")


if __name__ == "__main__":
    train()