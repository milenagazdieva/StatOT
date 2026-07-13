# A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers
This is the official `Python` implementation of the [ICLR 2026](https://iclr.cc) paper **A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers** (paper [page](https://openreview.net/forum?id=FJTdyG8jeJ&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)) on ICLR) by [Roman Tarasov](https://scholar.google.com/citations?view_op=list_works&hl=en&user=DYVHi8QAAAAJ), [Petr Mokrov](https://scholar.google.com/citations?user=CRsi4IkAAAAJ&hl=ru), [Milena Gazdieva](https://scholar.google.com/citations?user=h52_Zx8AAAAJ&hl=en), [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru), and [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en).

This repository contains reproducible PyTorch code for experiments demonstrating that the empirically observed generalization errors of OT solvers are close to the theoretical bounds derived in Corollary 4.10. The experiments consider two regimes in which the generalization error is dominated primarily by either the *estimation error* or the *approximation error*. Accordingly, they illustrate: (1) the effect of the estimation error arising from the limited number of training samples; and (2) the effect of the approximation error arising from the limited expressive capacity of the neural network architectures.

## Related repositories
- [Repository](https://github.com/iamalexkorotin/NeuralOptimalTransport) for [Neural Optimal Transport](https://arxiv.org/abs/2201.12220) paper (ICLR 2023).
- [Repository](https://github.com/iamalexkorotin/Wasserstein2Benchmark) for [Wasserstein-2 Benchmark]([https://arxiv.org/abs/2201.12220](https://proceedings.neurips.cc/paper_files/paper/2021/file/7a6a6127ff85640ec69691fb0f7cb1a2-Paper.pdf)) paper (NeurIPS 2021).

## Citation
```
@inproceedings{
    tarasov2026statistical,
    title={A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers},
    author={Tarasov, Roman and Mokrov, Petr and Gazdieva, Milena and Burnaev, Evgeny and Korotin, Alexander},
    booktitle={International Conference on Learning Representations},
    year={2026}
}
```

## Repository structure
The implementation is GPU-based. Tested with `torch==2.0.0+cu117` and 1 Tesla V100.

Toy experiments are issued in `.py` files. Auxilary source code is moved to `.py` modules (`src/`). The `benchmarks` folder contains code from the Wasserstein-2 Benchmark GitHub repository, on which we test our method.
- ```train_stat.py``` - experiment on the estimation error;
- ```train_approx.py``` - experiment on the approximation error.

## Credits
- ```Comet ML``` developer tools for machine learning.
