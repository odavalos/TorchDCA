# std libraries
import math as m
import random
import copy

# datascience/single cell libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
# plt.style.use('seaborn')

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from TorchDCA import DCA
from TorchDCA.loss import *


sc.settings.set_figure_params(dpi=320, facecolor='white')



# device for torch 
if torch.cuda.is_available():
    device = "cuda";
    print('==> Using GPU (CUDA)')

else :
    device = "cpu"
    print('==> Using CPU')
    print('    -> Warning: Using CPUs will yield to slower training time than GPUs')
    


# # read in testing data
# adata = sc.read_h5ad('~/JupyterNBs/Misc_Projects/pbmc_3k_dca.h5ad')
pbmc3k = sc.datasets.pbmc3k_processed()
adata = sc.datasets.pbmc3k()
celltypes_series = pbmc3k.obs['louvain']
filt_barcodes = list(celltypes_series.index)
adata.obs['barcodes'] = adata.obs.index

# filter certain barcodes
adata = adata[adata.obs.index.isin(filt_barcodes)]

adata.obs['louvain'] = celltypes_series

### std processing ###
# filter low expressed genes
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.filter_cells(adata, min_counts=3)

# keep raw data object
adata.raw = adata.copy()

sc.pp.normalize_per_cell(adata)
# calculate size factors from library sizes (n_counts)
adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

# log trans
sc.pp.log1p(adata)

# save raw object
adata.raw = adata

# scale data
sc.pp.scale(adata)


def train_AE(model, x, X_raw, size_factor, batch_size=128, lr=0.001, epochs=50):

    optimizer = torch.optim.Adam(params=model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999), 
                                eps=1e-08,
                                weight_decay=0.005, 
                                amsgrad=False)
    
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
#         dataset = TensorDataset(torch.Tensor(X_raw))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Training")

    for epoch in range(epochs):
        for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):

            x_tensor = Variable(x_batch).to(device)
            x_raw_tensor = Variable(x_raw_batch).to(device)
            sf_tensor = Variable(sf_batch).to(device)

            z, d, mean_tensor, disp_tensor, pi_tensor = model(x_tensor)

            mse_loss = F.mse_loss(d, x_tensor)
            z_loss = zinb_loss(x_raw_tensor, mean_tensor, disp_tensor, pi_tensor, sf_tensor)

            loss = mse_loss + z_loss

            optimizer.zero_grad()

            loss.backward()

            # optimizer.zero_grad()

            optimizer.step()
#                 print('Epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        print('Epoch [{}/{}], ZINB loss:{:.4f}'.format(epoch+1, epochs, loss.item()))


    return model

dca_model = DCA(input_size=adata.n_vars).to(device)


# train model
dca_trained = train_AE(dca_model, x=adata.X, X_raw=adata.raw.X.todense(), size_factor=adata.obs.size_factors, batch_size=64, lr=0.0001, epochs=100)

# 
z = dca_trained(torch.Tensor(adata.X))[0].detach().numpy()
d = dca_trained(torch.Tensor(adata.X))[1].detach().numpy()
means = dca_trained(torch.Tensor(adata.X))[2].detach().numpy()
disps = dca_trained(torch.Tensor(adata.X))[3].detach().numpy()
dropout = dca_trained(torch.Tensor(adata.X))[4].detach().numpy()


# add autoencoder latent data to scanpy object
adata.obsm['X_AE'] = z
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=None, use_rep='X_AE', random_state=2022, key_added='ae_cord')
sc.tl.leiden(adata, resolution=0.4,random_state=2022, restrict_to=None, key_added='leiden_ae', 
                  obsp='ae_cord_connectivities')

sc.tl.umap(adata, neighbors_key='ae_cord', n_components=2)
# sc.tl.draw_graph(adata, neighbors_key='ae_cord')

# sc.pl.draw_graph(adata, color=['leiden_ae','CD8A', 'NKG7', 'CD4'], use_raw=True)

sc.pl.umap(adata, color=['leiden_ae','CD8A', 'NKG7', 'CD4'], use_raw=True, save='_dca_latent.pdf')




