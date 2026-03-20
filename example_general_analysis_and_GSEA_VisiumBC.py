'''
paper
'''
#%%
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from instapath.utils_model import ETM_MutiModal_Dataset, InSTaPath
from instapath.utils_train import process_epoch_multimodal, monitor_perf
from instapath.utils_analysis import get_reconstruction_multimodal
from instapath.utils_general import set_seed, save_pickle, load_pickle
from instapath.utils_clustering import eval_metrics
from instapath.utils_plot import plot_theta_scatter, plot_beta_heatmap, plot_theta_clustermap, plot_topic_and_top_words, plot_spatial_clusters_v2
from instapath.utils_plot import plot_image_words_on_WSI_scale
from instapath.utils_clustering import topic_coherence_npmi_numpy, calculate_td
from pathlib import Path
import gseapy as gp
#%%
set_seed(2025) 
study_name = "VisiumBC"
inputs_dir = Path("./inputs", study_name)
outputs_dir = "./model_weights/"

#%% load image data
img_adata = sc.read_h5ad(Path(inputs_dir, "visium_BC_img_hvi256.h5ad"))
top_img_matrix = img_adata.X.toarray()
top_img_names = img_adata.var.index.to_numpy()

#%% load gene data
adata = sc.read_h5ad(Path(inputs_dir, "VisiumBC_hvg3000.h5ad"))
raw_genes = adata.X.toarray()
top_gene_names = adata.var_names.to_numpy()
annotations = adata.obs["Annotation"].to_numpy()
category = list(adata.obs["Annotation"].cat.categories)
palette = load_pickle(Path(inputs_dir, "visium_color_palette.pickle"))
special_color_map = {cat: palette[i] for i, cat in enumerate(category)}

#%%
labels_to_keep = ['DCIS #1', 'DCIS #2', 'adipocytes', 'immune', 'invasive','stromal']
adata_sub = adata[adata.obs['Annotation'].isin(labels_to_keep)].copy()

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K=10
topics_list = [f'topic{i+1}' for i in range(K)]
n_batches=1
model = InSTaPath(num_topics=K, n_batches=n_batches,
              vocab_size_gene = len(top_gene_names), vocab_size_img = len(top_img_names), 
              t_hidden_size_gene=256, t_hidden_size_img=32,
              rho_size=256, 
              enc_drop_gene=0, enc_drop_img=0, enc_norm = "batch",
              normalize_beta=True,
              enable_gene_batch_bias=False, enable_img_batch_bias=False,
              enable_gene_global_bias=False, enable_img_global_bias=False).to(device)

num_epochs = 10000 #100
optimizer = Adam(model.parameters(), lr=5e-3, weight_decay=1.2e-6) # 5e-3

#%%uncomment to train the model
# # Create DataLoaders
# batch_idx_list = np.zeros(len(annotations))
# dataset = ETM_MutiModal_Dataset(raw_genes, top_img_matrix, batch_idx_list)
# batch_size = len(dataset) 
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # Initialize performance tracking array
# perf_train = torch.zeros((num_epochs, 5), dtype=torch.float32)

# for i in range(num_epochs):

#     train_loss, train_loss_recon_gene, train_loss_recon_img, train_loss_kld = process_epoch_multimodal(train_loader, model, optimizer, device, train=True)
    
#     perf_train[i] = torch.tensor([i, train_loss, train_loss_recon_gene, train_loss_recon_img, train_loss_kld], device=device)

#     print(f"Epoch [{i+1}/{num_epochs}]: {train_loss:.4f}")

# print('Done Training...')

# # plot loss
# titles = [
#     "Total Loss",
#     "Reconstruction loss (gene)",
#     "Reconstruction loss (img)",
#     "KLD loss"
# ]
# monitor_perf(perf_train, titles)

# # save 
# torch.save(model.state_dict(), Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs_model_weights.pth"))
# recon = get_reconstruction_multimodal(model, full_loader, device)
# save_pickle(recon, Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs.pickle"))

#%% load model weights
model.load_state_dict(torch.load(Path(outputs_dir, f"InSTaPath_{study_name}_K{K}_{num_epochs}epochs_model_weights.pth"), map_location="cpu"))
model.to(device)
model.eval()

#%% get reconstruction
batch_idx_list = np.zeros(len(annotations))
dataset = ETM_MutiModal_Dataset(raw_genes, top_img_matrix, batch_idx_list)
batch_size = 4000 
full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
recon = get_reconstruction_multimodal(model, full_loader, device)
theta_list = recon['theta_list']

#%% clustering results
theta_list_sub = theta_list[adata.obs['Annotation'].isin(labels_to_keep)]
labels_sub = adata_sub.obs['Annotation'].tolist()
clustering_method="louvain"
cluster_results_0 = eval_metrics(theta_list_sub, labels_sub, method=clustering_method)
ari_0 = cluster_results_0['ari']
asw_0 = cluster_results_0['asw']
nmi_0 = cluster_results_0['nmi']
print(f"InSTaPath theta ( ARI = {ari_0:.4f}, ASW = {asw_0:.4f}, NMI = {nmi_0:.4f})")

#%% theta clustermap
title=f"{study_name}:theta"
clustermap_order = plot_theta_clustermap(theta_list_sub, adata_sub, special_color_map, topics_list, title)
topic_usage = theta_list_sub.mean(axis=0)          # average proportion per topic
top_topics = np.argsort(topic_usage)[-8:][::-1]
print("Top topics:", [f"Topic {k+1}" for k in top_topics])

#%% theta scatter map
n_rows = K//10
plot_theta_scatter(theta_list, adata, n_rows=n_rows, figsize=(24, 3*n_rows), title= f"Theta topics ({K} topics)")

#%% assign annotations to topics
from instapath.utils_plot import topic_target_pointbiserial_heatmap
rmat, pmat, fig, ax = topic_target_pointbiserial_heatmap(
    theta_list=theta_list_sub,
    annotations_sub=adata_sub.obs['Annotation'].tolist(),
    labels_to_keep=labels_to_keep,
    topics_list=topics_list,
    alpha=0.05,
    multiple_testing="bonferroni",
)

#%% theta spatial map
fig, axes = plot_spatial_clusters_v2(
    x = adata_sub.obs["pxl_row_in_fullres"].to_numpy(),
    y = adata_sub.obs["pxl_col_in_fullres"].to_numpy(),
    annotations=adata_sub.obs["Annotation"].to_list(),
    theta_cluster=cluster_results_0["cluster"],
    special_color_map=special_color_map,
    ari=ari_0, asw=asw_0, nmi=nmi_0,
    figsize=(10, 4),
    point_size=1,
    title = "Theta"
)

#%% beta heatmap
top_n_words = 5
# beta: topic-to-gene probabilities
title = r"Topic-gene Dist. ($\beta: gene$)"
df_beta_gene, top_words_list_gene = plot_beta_heatmap(recon['beta_gene'], top_gene_names, topics_list, title, top_n=top_n_words, vmax=0.08) 

title = r"Topic-image words Dist. ($\beta: image$)"
df_beta_img, top_words_list_img = plot_beta_heatmap(recon['beta_img'], top_img_names, topics_list, title, top_n=top_n_words, vmax=0.08) 

#%% plot top genes on ST scale
fig, axes = plot_topic_and_top_words(
    adata=adata,
    df_beta=df_beta_gene,
    top_words_list=top_words_list_gene,
    hvg_names = list(top_gene_names),
    theta=theta_list,
    X_data=raw_genes,
    n_topics=K,
    top_n_per_topic=5,
    figsize=(15, 3*K)
)

#%% plot top image words on ST scale
fig, axes = plot_topic_and_top_words(
    adata=adata,
    df_beta=df_beta_img,
    top_words_list=top_words_list_img,
    hvg_names = list(top_img_names),
    theta=theta_list,
    X_data=top_img_matrix,
    n_topics=K,
    top_n_per_topic=5,
    figsize=(15, 3*K)
)
    
#%% plot top image words on WSI scale
topic_idx = 6
img_word_list = top_words_list_img[top_n_words*topic_idx:top_n_words*(topic_idx+1)]
# plotting can take a while when WSI is large
wsi_signal = plot_image_words_on_WSI_scale(wsi_token_file_path=Path(inputs_dir,"VisiumBC_codebook512&64_token_level.npz"),
                                            img_word_list=img_word_list,
                                            H_patches =74,
                                            W_patches=84, 
                                            title=f"topic{topic_idx+1}: Top {top_n_words} image words"
                                            )

#%% calculate TC: topoic coherence, TD: topic diversity for the gene modality
tc_gene = topic_coherence_npmi_numpy(raw_genes, recon['beta_gene'], top_s=5, eps=np.finfo(float).eps, return_per_topic=False)
td_gene = calculate_td(recon['beta_gene'], top_gene_names, topics_list)
print("TC (gene): ", tc_gene)
print("TD (gene): ", td_gene)

#%% calculate TC: topoic coherence, TD: topic diversity for the image modality
tc_img = topic_coherence_npmi_numpy(top_img_matrix, recon['beta_img'], top_s=5, eps=np.finfo(float).eps, return_per_topic=False)
td_img = calculate_td(recon['beta_img'], top_img_names, topics_list)
print("TC: ", tc_img)
print("TD: ", td_img)

#%% GSEA
with torch.no_grad():
    rho_gene = model.rho_gene.detach().cpu().numpy()
    beta_gene_raw = model.alphas(model.rho_gene).detach().cpu().numpy().T

topic_idx=7 # topic 8
gene_value_pairs = list(zip(top_gene_names, beta_gene_raw[topic_idx,:]))
sorted_pairs = sorted(gene_value_pairs, key=lambda x: x[1], reverse=True)
ranked_genes = {gene: value for gene, value in sorted_pairs}

# Convert to a DataFrame (or load from a file)
rnk_df = pd.DataFrame({
    'gene_symbol': ranked_genes.keys(),
    'rank': ranked_genes.values()
})

# Run GSEA pre-rank analysis
prerank_res = gp.prerank(
    rnk=rnk_df,
    gene_sets= ["./GSEA_library/h.all.v2025.1.Hs.symbols.gmt", "./GSEA_library/c2.all.v2025.1.Hs.symbols.gmt", "./GSEA_library/c4.all.v2025.1.Hs.symbols.gmt", "./GSEA_library/c5.all.v2025.1.Hs.symbols.gmt", "./GSEA_library/c6.all.v2025.1.Hs.symbols.gmt"],  # Or use `gene_sets=embryo_terms` for custom list
    processes=4,          # Number of CPU cores
    permutation_num=1000, # Default permutations
    outdir=f'./GSEA_results/gsea_results_topic{topic_idx+1}',  # Output directory
    format='png',         # Plot format
    seed=42,             # Reproducibility
    min_size=15,         # Min genes in a gene set
    max_size=500         # Max genes in a gene set
)


plt.figure()
terms = prerank_res.res2d.Term
axs = prerank_res.plot(terms=terms[0:10], show_ranking=True, legend_kws={'loc': (1.2, 0)})
plt.title(f"topic {topic_idx+1}") 
plt.show()

