# scMoDCA

## Model architecture

![scMoDCA model architecture](docs/images/scmodca_framework.png)

## Key innovations

1. **Mutual-information-constrained disentanglement.** scMoDCA learns a modality-shared representation and extracts modality-specific representations through residual decomposition. Basic and conditional CLUB objectives reduce redundant dependencies between representation branches, while an InfoNCE-based objective preserves information shared between modality encodings and the shared representation.

2. **Adaptive gated cross-attention fusion.** The shared representation queries modality-specific branches through cross-attention. A learnable gate controls the complementary contribution, which is integrated through a residual connection to form the final cellular representation.

## Dependencies

Python 3.10

- torch
- numpy
- pandas
- scipy
- scikit-learn
- scanpy
- anndata
- h5py
- episcanpy
- scikit-misc
- ezdict
- matplotlib
- seaborn
- umap-learn
- tqdm

## Environment setup

Create and activate a dedicated Conda environment:

```bash
conda create -n scmodca python=3.10 -y
conda activate scmodca
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scipy scikit-learn scanpy anndata h5py episcanpy scikit-misc ezdict matplotlib seaborn umap-learn tqdm
```

For GPU training, use a PyTorch build compatible with your CUDA environment. Check the installation and GPU availability:

```bash
python -c "import torch, scanpy, episcanpy; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

These commands provide a starting environment; dependency versions are not pinned to a validated reproduction environment.



