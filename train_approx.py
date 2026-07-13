import os, sys
sys.path.append("..")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import comet_ml
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
parser.add_argument('--D_HIDDEN', type=int, default=6,
                     help='1')
ARGS = parser.parse_args()

DIM = 4
BATCH_SIZE = 1024

GPU_DEVICE = 0

BENCHMARK = 'Mix3toMix10'

DEVICE = 'cuda:0'
MAX_ITER = 1#10001
LR = 1e-3
CONVEX = True
INNER_ITERS = 1#10
INV_MAX_ITER = 1#1000
INV_TOL=1e-3
COST = sq_cost
D_HIDDEN = 2**ARGS.D_HIDDEN

D_HYPERPARAMS = {
    'dim' : DIM,
    'rank' : 1,
    'hidden_layer_sizes' : [max(2*DIM, D_HIDDEN), max(2*DIM, D_HIDDEN), max(DIM, D_HIDDEN//2)],
    'strong_convexity' : 1e-4
}

EXP_NAME = f'{BENCHMARK}_{DIM}_{D_HIDDEN}'
OUTPUT_PATH = '../logs/' + BENCHMARK
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

assert torch.cuda.is_available()
OUTPUT_SEED = 0x000000
torch.cuda.set_device(GPU_DEVICE)
torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

config = dict(
    BENCHMARK=BENCHMARK,
    DIM=DIM,
    D_HIDDEN=D_HIDDEN
)

##### BENCHMARK SETUP #####
benchmark = mbm.Mix3ToMix10Benchmark(DIM)
emb_X = PCA(n_components=2).fit(benchmark.input_sampler.sample(2**14).cpu().detach().numpy())
emb_Y = PCA(n_components=2).fit(benchmark.output_sampler.sample(2**14).cpu().detach().numpy())

###### UTILS #####
def sample_from_tensor(t, size=BATCH_SIZE):
    indices = random.choices(range(t.shape[0]), k=min(size, t.shape[0]))
    s = t[indices,:]
    return t

size = 4096
X_test = benchmark.input_sampler.sample(size); X_test.requires_grad_(True)
Y_test = benchmark.map_fwd(X_test, nograd=True); Y_test.requires_grad_(True)

def invert(D, Y, lr=0.1, max_iter=INV_MAX_ITER, tol=INV_TOL):
    freeze(D)
    Y_inv = torch.rand_like(Y, requires_grad=True)

    opt = torch.optim.Adam([Y_inv], lr=lr)
    torch.sum(Y_inv).backward() # Terrible solution to init gradients
    opt.zero_grad()

    for it in range(max_iter):
        loss = D(Y_inv).sum() - (Y * Y_inv).sum()
        loss.backward()

        if torch.sqrt((Y_inv.grad.data ** 2).mean(dim=1)).mean() < tol:
            break

        opt.step()
        opt.zero_grad()
    return Y_inv.detach()

def score_fitted_maps(benchmark, T, D_conj, size=10000):
    '''Estimates L2-UVP and cosine metrics for transport map.'''
    
    freeze(T); freeze(D_conj);
    
    X = X_test
    Y = Y_test
    
    X_push = T(X).detach()
    
    with torch.no_grad():
        L2_UVP_fwd = ((Y - X_push) ** 2).sum(dim=1).mean().item()
    
    gc.collect(); torch.cuda.empty_cache() 
    return L2_UVP_fwd

###### TRAINING ########

torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)

hidden_sizes = [1,2,4,8,16,32,64,128,256,512,1024]
df_uvp = pd.DataFrame(index=[DIM], columns=hidden_sizes)

for size in hidden_sizes:
    uvp_list = []
    torch.manual_seed(OUTPUT_SEED); np.random.seed(OUTPUT_SEED)
    for i in range(3):
        exp = comet_ml.start(
            project_name="stat_ot",
            experiment_config=comet_ml.ExperimentConfig(
                name=EXP_NAME,
                tags=["stat_ot"],
                parse_args=False
            ),
        )
        
        seed = np.random.randint(low=1e6)
        torch.manual_seed(seed); np.random.seed(seed)
        
        L2_UVP_fwd_min, L2_UVP_inv_min = np.inf, np.inf
        

        T = nn.Sequential(
            nn.Linear(DIM, size),
            nn.ReLU(True),
            nn.Linear(size, size*2),
            nn.ReLU(True),
            nn.Linear(size*2, size*4),
            nn.ReLU(True),
            nn.Linear(size*4, size*2),
            nn.ReLU(True),
            nn.Linear(size*2, size),
            nn.ReLU(True),
            nn.Linear(size, DIM),
        ).to(DEVICE)
        par = np.sum([np.prod(p.shape) for p in T.parameters()])

        D_conj = DenseICNN(**D_HYPERPARAMS).cuda()
        D_conj_opt = torch.optim.Adam(D_conj.parameters(), lr=LR, weight_decay=1e-10)
        T_opt = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-10)

        for iteration in tqdm(range(MAX_ITER)):
            X = benchmark.input_sampler.sample(BATCH_SIZE); X.requires_grad_(True)
            Y = benchmark.output_sampler.sample(BATCH_SIZE); Y.requires_grad_(True)

            unfreeze(D_conj); freeze(T) # D_conj - convexify
            # Negative Wasserstein distance
            X_inv = T(X).detach()
            D_conj_opt.zero_grad()  
            W_loss = (D_conj(Y) - D_conj(X_inv)).mean()
            W_loss.backward(); D_conj_opt.step();
            exp.log_metric(name="D_loss", value=W_loss.item(), step=iteration)

            if CONVEX: D_conj.convexify(); 

            unfreeze(T); freeze(D_conj)
            for inner_it in range(INNER_ITERS): 
                X = benchmark.input_sampler.sample(BATCH_SIZE); X.requires_grad_(True)
                X.requires_grad_(True)

                T_opt.zero_grad()
                X_push = T(X)
                conj_loss = (D_conj(X_push) - (X_push * X).sum(dim=1, keepdims=True)).mean()
                conj_loss.backward()
                T_opt.step();
            exp.log_metric(name="T_loss", value=conj_loss.item(), step=iteration)

            if iteration % 10 == 0:
                clear_output(wait=True)
                L2_UVP_fwd = score_fitted_maps(benchmark, T, D_conj, size=10000)
                exp.log_metric(name="L2_UVP", value=L2_UVP_fwd, step=iteration)

                if L2_UVP_fwd < L2_UVP_fwd_min:
                    L2_UVP_fwd_min = L2_UVP_fwd

        torch.save(T.state_dict(), f'state_dicts/T_approx_{DIM}_{D_HIDDEN}_{size}_width_{i}.pt')
        torch.save(D_conj.state_dict(), f'state_dicts/Dconj_approx_{DIM}_{D_HIDDEN}_{size}_width_{i}.pt')
        uvp_list.append(L2_UVP_fwd_min)
        exp.end()
        
    df_uvp.loc[DIM, size] = f'{np.mean(uvp_list):.5f} ± {np.std(uvp_list):.2f}'  
    df_uvp.to_csv(f'metrics/approx_{DIM}_{D_HIDDEN}_width_.csv')