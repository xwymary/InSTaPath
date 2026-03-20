# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from instapath.utils_general import set_seed, save_pickle, load_pickle
from instapath.utils_model import ETM_MutiModal_Dataset, InSTaPath
from instapath.utils_train import process_epoch_multimodal
from instapath.utils_analysis import get_reconstruction_multimodal, get_reconstruction_perturb_gene
from instapath.utils_plot import plot_beta_heatmap
from pathlib import Path

#%%
set_seed(2025) 
study_name = "P2CRC"
inputs_dir = Path("./inputs", study_name)
outputs_dir = "./model_weights/"

#%% load image data
img_adata = sc.read_h5ad(Path(inputs_dir, "visium_P2CRC_img_hvi256.h5ad"))
top_img_matrix = img_adata.X.toarray()
top_img_names = img_adata.var.index.to_numpy()

#%% load gene data
adata = sc.read_h5ad(Path(inputs_dir, "P2CRC_hvg3000.h5ad"))
raw_genes = adata.X.toarray()
top_gene_names = adata.var_names.to_numpy()
annotations = adata.obs["DeconvolutionLabel1"].to_numpy()

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K=10
topics_list = [f'topic{i+1}' for i in range(K)]
n_batches=1
model = InSTaPath(num_topics=K, n_batches=n_batches,
              vocab_size_gene = len(top_gene_names), vocab_size_img = len(top_img_names), 
              t_hidden_size_gene=256, t_hidden_size_img=32,
              rho_size=256, 
              enc_drop_gene=0, enc_drop_img=0, enc_norm = "layer",
              normalize_beta=True,
              enable_gene_batch_bias=False, enable_img_batch_bias=False,
              enable_gene_global_bias=False, enable_img_global_bias=False).to(device)

num_epochs = 1000 #100
optimizer = Adam(model.parameters(), lr=5e-3, weight_decay=1.2e-6) # 5e-3

#%%
# uncomment to train the model
# batch_idx_list = np.zeros(len(annotations))
# dataset = ETM_MutiModal_Dataset(raw_genes, top_img_matrix, batch_idx_list)

# # Create DataLoaders
# batch_size = 4000 
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # Initialize performance tracking array
# perf_train = torch.zeros((num_epochs, 5), dtype=torch.float32)

# for i in range(num_epochs):
#     train_loss, train_loss_recon_gene, train_loss_recon_img, train_loss_kld = process_epoch_multimodal(train_loader, model, optimizer, device, train=True)
#     perf_train[i] = torch.tensor([i, train_loss, train_loss_recon_gene, train_loss_recon_img, train_loss_kld], device=device)
#     print(f"Epoch [{i+1}/{num_epochs}]: {train_loss:.4f}")
# print('Done Training...')

# torch.save(model.state_dict(), Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs_model_weights.pth"))
# recon = get_reconstruction_multimodal(model, full_loader, device)
# save_pickle(recon, Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs.pickle"))
#%%
model.load_state_dict(torch.load(Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs_model_weights.pth"), map_location="cpu"))
model.to(device)
model.eval()

#%% get reconstruction
batch_idx_list = np.zeros(len(annotations))
dataset = ETM_MutiModal_Dataset(raw_genes, top_img_matrix, batch_idx_list)
batch_size = 4000 
full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
recon = get_reconstruction_multimodal(model, full_loader, device)

#%% beta: topic-to-gene probabilities, topic-to-image words probabilities
top_n_words = 5
# plot top 5 genes under each topic
title = r"Topic-gene Dist. ($\beta: gene$)"
df_beta_gene, top_words_list_gene = plot_beta_heatmap(recon['beta_gene'], top_gene_names, topics_list, title, top_n=top_n_words, vmax=0.08) 
# plot top 5 image words under each topic
title = r"Topic-image words Dist. ($\beta: image$)"
df_beta_img, top_words_list_img = plot_beta_heatmap(recon['beta_img'], top_img_names, topics_list, title, top_n=top_n_words, vmax=0.08) 

#%%
# get img recon from gene inputs
img_recon = get_reconstruction_perturb_gene(model, full_loader, device)
img_data = img_recon['preds_img']

#%%
# in silico perturbation
top_n_KO = 500
topic_idx = 5
genes_to_zero = df_beta_gene[f'topic{topic_idx+1}'].nlargest(top_n_KO).index.tolist()
indices = [list(top_gene_names).index(g) for g in genes_to_zero]
gene_data = raw_genes.copy()
gene_data[:,indices]=0

# get perturbed img recon
dataset = ETM_MutiModal_Dataset(gene_data, top_img_matrix, batch_idx_list)
perturb_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
perturb_recon = get_reconstruction_perturb_gene(model, perturb_loader, device)
img_data_p = perturb_recon['preds_img']
    
#%% calculate the effect of perturbation using the top 5 image words under topic k
img_word_list = top_words_list_img[top_n_words*topic_idx:top_n_words*(topic_idx+1)]
index = [list(top_img_names).index(img_word) for img_word in img_word_list]
d_sum = img_data[:,index].sum(axis=1)
d_sum_p = img_data_p[:,index].sum(axis=1)
diff = d_sum_p - d_sum

#%% visualize the effect of perturbation using the top 5 image words under topic k
plt.figure(figsize=(8, 8))
# coordinates
pos_x = adata.obs['pxl_col_in_fullres'].to_numpy()
pos_y = adata.obs['pxl_row_in_fullres'].to_numpy()
sc = plt.scatter(pos_x, pos_y, c=diff, s=1, cmap="coolwarm")
plt.title(f"Perturbation Effect (Topic {topic_idx+1}, Top {top_n_KO} KO genes)", fontsize=14)
# cbar = plt.colorbar(sc)
# cbar.set_label("Δ Intensity (Perturbed - Original)", fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.axis("off")
plt.tight_layout()
plt.show()

#%% calculate the most differentially expressed image words
from instapath.utils_analysis import compute_de_image_words

de_df = compute_de_image_words(img_data, img_data_p, top_img_names)
# get the most suppressed image words under perturbation 
suppressed = de_df.sort_values("log2FC")
print(suppressed.head(top_n_words))
img_word_list = suppressed['img_word'].iloc[:top_n_words].tolist()

#%% plot top DE image words on WSI scale
from instapath.utils_plot import plot_image_words_on_WSI_scale
# plotting can take a while when WSI is large
wsi_signal = plot_image_words_on_WSI_scale(wsi_token_file_path=Path(inputs_dir,"P2CRC_codebook512&64_token_level.npz"),
                                                img_word_list=img_word_list,
                                                H_patches = 119,
                                                W_patches=183, 
                                                title=f"KO{top_n_KO} (topic{topic_idx+1}): Top {top_n_words} DE image words"
                                            )
