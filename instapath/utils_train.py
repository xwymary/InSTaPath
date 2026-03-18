# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 20:13:43 2026

@author: xwyma
"""
import torch
import numpy as np
from contextlib import nullcontext
import matplotlib.pyplot as plt

#%%
def process_epoch(loader, model, optimizer=None, device='cpu', train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_log_total=[]
    loss_log_recon=[]
    loss_log_kld=[]
    with torch.no_grad() if not train else nullcontext():
        for idx, (X_tensor, X_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            X_tensor = X_tensor.to(device)
            row_sums = X_tensor.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X_tensor_normalized = X_tensor / row_sums

            # Forward pass
            recon_loss_gene, kld_theta = model(X_tensor, X_tensor_normalized, X_batch)
            total_loss = recon_loss_gene + kld_theta

            if train:
                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            loss_log_total.append(total_loss.item())
            loss_log_recon.append(recon_loss_gene.item())
            loss_log_kld.append(kld_theta.item())
    return np.mean(loss_log_total), np.mean(loss_log_recon), np.mean(loss_log_kld)

#%%
def monitor_perf(perf_train, titles):
    ncols = len(titles)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3*ncols, 3))  # Adjusted figure size

    for i in range(len(titles)):  
        axes[i].plot(perf_train[:, i+1], color="blue")
        axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()
    

#%%
def process_epoch_multimodal(loader, model, optimizer=None, device='cpu', train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_log=[]
    loss_log_recon_gene=[]
    loss_log_recon_img=[]
    loss_log_kld=[]
    with torch.no_grad() if not train else nullcontext():
        for idx, (batch_gene, batch_img, batch_idx) in enumerate(loader):
        
            batch_gene, batch_img = batch_gene.to(device), batch_img.to(device)
            batch_idx = batch_idx.to(device)
            
            X_tensor_gene = batch_gene
            row_sums = X_tensor_gene.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X_tensor_normalized_gene = X_tensor_gene / row_sums

            X_tensor_img = batch_img
            row_sums = X_tensor_img.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X_tensor_normalized_img = X_tensor_img / row_sums
            

            # Forward pass
            recon_loss_gene, recon_loss_img, kld_theta = model(X_tensor_gene, X_tensor_img, X_tensor_normalized_gene, X_tensor_normalized_img, batch_idx)
            total_loss = recon_loss_gene + recon_loss_img + kld_theta

            if train:
                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            loss_log.append(total_loss.item())
            loss_log_recon_gene.append(recon_loss_gene.item())
            loss_log_recon_img.append(recon_loss_img.item())
            loss_log_kld.append(kld_theta.item())
    return np.mean(loss_log), np.mean(loss_log_recon_gene), np.mean(loss_log_recon_img), np.mean(loss_log_kld)
