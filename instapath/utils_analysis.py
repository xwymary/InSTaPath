# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:26:21 2026

@author: xwyma
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
#%%
def get_reconstruction(model, loader, device):
    model.eval()
    logvar_list=[]
    theta_list=[]
    with torch.no_grad():
        for idx, X_tensor in enumerate(loader):
        
            X_tensor = X_tensor.to(device)
            row_sums = X_tensor.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X_tensor_normalized = X_tensor / row_sums

            # Forward pass
            mu_theta, logvar_theta= model.encode(X_tensor_normalized)
            mu_theta = F.softmax(mu_theta, dim=-1)

            theta_list.append(mu_theta.detach().cpu().numpy())
            logvar_list.append(logvar_theta.detach().cpu().numpy())


        beta = model.get_beta()
        beta = beta.detach().cpu().numpy()

    theta_list = np.vstack(theta_list)
    preds = theta_list @ beta
    results = {'logvar_list': logvar_list, 
               'theta_list': theta_list,
               'beta': beta, 
               'preds': preds}
    return results

#%%
def get_reconstruction_multimodal(model, loader, device):
    model.eval()
    logvar_list_gene=[]
    theta_list_gene=[]
    logvar_list_img=[]
    theta_list_img=[]
    theta_list=[]
    batch_list=[]
    with torch.no_grad():
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
            mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img = model.encode(X_tensor_normalized_gene, X_tensor_normalized_img)
            mu, logvar = model.product_of_Gaussian(mu_theta_gene, logvar_theta_gene, mu_theta_img, logvar_theta_img)
            mu_theta_gene = F.softmax(mu_theta_gene, dim=-1)
            mu_theta_img = F.softmax(mu_theta_img, dim=-1)
            mu = F.softmax(mu, dim=-1)

            theta_list_gene.append(mu_theta_gene.detach().cpu().numpy())
            theta_list_img.append(mu_theta_img.detach().cpu().numpy())
            logvar_list_gene.append(logvar_theta_gene.detach().cpu().numpy())
            logvar_list_img.append(logvar_theta_gene.detach().cpu().numpy())
            theta_list.append(mu.detach().cpu().numpy())
            batch_list = batch_list + list(batch_idx)

        beta_gene, beta_img = model.get_beta()
        beta_gene, beta_img = beta_gene.detach().cpu().numpy(), beta_img.detach().cpu().numpy()

    batch_list = np.asarray([b.cpu() for b in batch_list])
    theta_list_gene = np.vstack(theta_list_gene)
    theta_list_img = np.vstack(theta_list_img)
    theta_list = np.vstack(theta_list)
    preds_gene = theta_list @ beta_gene
    preds_img = theta_list @ beta_img
    results = {'logvar_list_gene': logvar_list_gene, 'logvar_list_img': logvar_list_img,
               'theta_list_gene': theta_list_gene, 'theta_list_img': theta_list_img, 'theta_list': theta_list,
               'beta_gene': beta_gene, 'beta_img': beta_img,
               'batch_list': batch_list, 
               'preds_gene': preds_gene, 'preds_img': preds_img}
    return results

#%%
def get_reconstruction_perturb_gene(model, loader, device):
    # alpha is the perturbation strength, topic_idx, alpha
    model.eval()
    theta_list_gene=[]
    with torch.no_grad():
        for idx, (batch_gene, batch_img, batch_idx) in enumerate(loader):
        
            batch_gene, batch_img = batch_gene.to(device), batch_img.to(device)
            batch_idx = batch_idx.to(device)

            X_tensor_gene = batch_gene
            row_sums = X_tensor_gene.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            X_tensor_normalized_gene = X_tensor_gene / row_sums

            # Forward pass
            q_theta_gene = model.gene_encoder(X_tensor_normalized_gene)
            mu_theta_gene = model.mu_q_theta_gene(q_theta_gene)
            mu_theta_gene = F.softmax(mu_theta_gene, dim=-1)
            theta_list_gene.append(mu_theta_gene.detach().cpu().numpy())

        beta_img = F.softmax(model.alphas(model.rho_img), dim=0).transpose(1, 0)
        beta_img = beta_img.detach().cpu().numpy()

    theta_list_gene = np.vstack(theta_list_gene)
    preds_img = theta_list_gene @ beta_img
    results = {'theta_list_gene': theta_list_gene, 
               'beta_img': beta_img,
               'preds_img': preds_img}
    return results

#%%
def compute_de_image_words(img_data, img_data_p, img_names):
    """
    Compute differential expression (DE) between original and perturbed image features.

    Parameters
    ----------
    img_data : np.ndarray
        Original data (cells/spots × features)
    img_data_p : np.ndarray
        Perturbed data (same shape as img_data)
    img_names : list
        Names of image features (length = number of columns)

    Returns
    -------
    de_df : pd.DataFrame
        DataFrame with DE statistics
    """
    
    # means
    mean1 = img_data.mean(axis=0)
    mean2 = img_data_p.mean(axis=0)
    
    # log2 fold change
    eps = 1e-8
    log2_fc = np.log2((mean2 + eps) / (mean1 + eps))
    
    # p-values (Wilcoxon rank-sum)
    pvals = np.array([
        ranksums(img_data[:, j], img_data_p[:, j]).pvalue
        for j in range(img_data.shape[1])
    ])
    
    # multiple testing correction
    reject, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
    
    # dataframe
    de_df = pd.DataFrame({
        "img_word": img_names,
        "mean_group1": mean1,
        "mean_group2": mean2,
        "log2FC": log2_fc,
        "pval": pvals,
        "padj": pvals_adj,
        "significant": reject
    }).sort_values("padj")
    
    return de_df