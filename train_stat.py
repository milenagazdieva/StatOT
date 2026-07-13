import os, sys
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import gc
import argparse

from sklearn.decomposition import PCA
from IPython.display import clear_output

from src.icnn import DenseICNN
from src.tools import unfreeze, freeze
import src.map_benchmark as mbm

import pandas as pd
import random

from tqdm import tqdm
from IPython.display import clear_output

def sq_cost (X, Y):
    return (X-Y).square().flatten(start_dim=1).mean(dim=1)

## CONFIG ############

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--DIM', type=int, default=2,
                     help='1')
ARGS = parser.parse_args()

DIM = 2**ARGS.DIM
BATCH_SIZE = 1024

GPU_DEVICE = 0

BENCHMARK = 'Mix3toMix10'

DEVICE = 'cuda:0'
MAX_ITER = 10001
LR = 1e-3
CONVEX = True
INNER_ITERS = 10
COST = sq_cost

D_HYPERPARAMS = {
    'dim' : DIM,
    'rank' : 1,
    'hidden_layer_sizes' : [max(2*DIM, 64), max(2*DIM, 64), max(DIM, 32)],
    'strong_convexity' : 1e-4
}

OUTPUT_PATH = '../logs/' + BENCHMARK
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

assert torch.cuda.is_available()
OUTPUT_SEED = 0x000000
torch.cuda.set_device(GPU_DEVICE)
torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

##### BENCHMARK SETUP #####
benchmark = mbm.Mix3ToMix10Benchmark(DIM)
emb_X = PCA(n_components=2).fit(benchmark.input_sampler.sample(2**14).cpu().detach().numpy())
emb_Y = PCA(n_components=2).fit(benchmark.output_sampler.sample(2**14).cpu().detach().numpy())

####### POTENTIALS SETUP #####
torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

T = nn.Sequential(
        nn.Linear(DIM, 512),
        nn.ReLU(True),
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 2048),
        nn.ReLU(True),
        nn.Linear(2048, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Linear(512, DIM),
    ).to(DEVICE)

D_conj = DenseICNN(**D_HYPERPARAMS).cuda()

D_conj_opt = torch.optim.Adam(D_conj.parameters(), lr=LR, weight_decay=1e-10)
T_opt = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-10)

###### UTILS #####
def sample_from_tensor(t, size=BATCH_SIZE):
    indices = random.choices(range(t.shape[0]), k=min(size, t.shape[0]))
    s = t[indices,:]
    return t

size = 4096
X_test = benchmark.input_sampler.sample(size); X_test.requires_grad_(True)
Y_test = benchmark.map_fwd(X_test, nograd=True); Y_test.requires_grad_(True)

def score_fitted_maps(benchmark, T, size=4096):
    '''Estimates L2-UVP and cosine metrics for transport map.'''
    
    freeze(T)
    X = X_test
    Y = Y_test
    
    X_push = T(X)
    
    with torch.no_grad():
        L2_UVP_fwd = 100 * (((Y - X_push) ** 2).sum(dim=1).mean())
    
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd

###### TRAINING ########

torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

data_sizes = [100,175,300,600,1000,1750,3000,6000,10000,17500]
df_uvp = pd.DataFrame(index=[DIM], columns=data_sizes)

for size in data_sizes:
    uvp_list = []
    torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)
    for i in range(3):
        seed = np.random.randint(low=1e6)
        torch.manual_seed(seed); np.random.seed(seed)
        
        L2_UVP_fwd_min, L2_UVP_inv_min = np.inf, np.inf
                            
        T = nn.Sequential(
            nn.Linear(DIM, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, DIM),
        ).to(DEVICE)
                            
        par = np.sum([np.prod(p.shape) for p in T.parameters()])

        D_conj = DenseICNN(**D_HYPERPARAMS).cuda()
        D_conj_opt = torch.optim.Adam(D_conj.parameters(), lr=LR, weight_decay=1e-10)
        T_opt = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-10)

        X_train = benchmark.input_sampler.sample(size); X_train.requires_grad_(True);
        Y_train = benchmark.output_sampler.sample(size); Y_train.requires_grad_(True);

        for iteration in tqdm(range(MAX_ITER)):
            X = sample_from_tensor(X_train, size=BATCH_SIZE)
            Y = sample_from_tensor(Y_train, size=BATCH_SIZE)

            unfreeze(D_conj); freeze(T) # D_conj - convexify
            # Negative Wasserstein distance
            X_inv = T(X).detach()
            D_conj_opt.zero_grad()  
            W_loss = (D_conj(Y) - D_conj(X_inv)).mean()
            W_loss.backward(); D_conj_opt.step();

            if CONVEX: D_conj.convexify(); 

            unfreeze(T); freeze(D_conj)
            for inner_it in range(INNER_ITERS): 
                X = sample_from_tensor(X_train, size=BATCH_SIZE)
                X.requires_grad_(True)

                T_opt.zero_grad()
                X_push = T(X)
                conj_loss = (D_conj(X_push) - (X_push * X).sum(dim=1, keepdims=True)).mean()
                conj_loss.backward()
                T_opt.step();

            if iteration % 10 == 0:
                clear_output(wait=True)
                L2_UVP_fwd = score_fitted_maps(benchmark, T)        

                if L2_UVP_fwd < L2_UVP_fwd_min:
                    L2_UVP_fwd_min = L2_UVP_fwd
        uvp_list.append(L2_UVP_fwd_min)
        torch.save(T.state_dict(), f'state_dicts/T__{INNER_ITERS}_{DIM}_{LR}_{size}_{i}.pt')
        torch.save(D_conj.state_dict(), f'state_dicts/Dconj__{INNER_ITERS}_{DIM}_{LR}_{size}_{i}.pt')
        
    df_uvp.loc[DIM, size] = f'{np.mean(uvp_list):.5f} ± {np.std(uvp_list):.2f}'  
    df_uvp.to_csv(f'metrics/stat_err_{INNER_ITERS}_{DIM}_{LR}.csv')            