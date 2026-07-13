import pandas as pd
import numpy as np

import os
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from .icnn import View

from PIL import Image
from tqdm import tqdm

import gc

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)   
    
def train_identity_map(D, sampler, batch_size=1024, max_iter=5000, lr=1e-3, tol=1e-3, blow=3, convex=True, verbose=False):
    "Trains potential D to satisfy D.push(x)=x by using MSE loss w.r.t. sampler's distribution"
    unfreeze(D)
    opt = torch.optim.Adam(D.parameters(), lr=lr, weight_decay=1e-10)
    if verbose:
        print('Training the potentials to satisfy push(x)=x')
    for iteration in tqdm_notebook(range(max_iter)) if verbose else range(max_iter):
        X = sampler.sample(batch_size)
        with torch.no_grad():
            X *= blow
        X.requires_grad_(True)

        loss = F.mse_loss(D.push(X), X.detach())
        loss.backward()
        opt.step(); opt.zero_grad()
        
        if convex:
            D.convexify()

        if loss.item() < tol:
            break
            
    loss = loss.item()
    gc.collect(); torch.cuda.empty_cache()
    return loss