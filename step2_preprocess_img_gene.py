# -*- coding: utf-8 -*-
"""
add annotation to Visium data if there is any
calculate spot level image word count
save: 3000 highly variable genes, 256 highly variable image words, token level image word count.
"""
import scanpy as sc
import anndata as ad
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from instapath.config import cfg
from instapath.utils_patchify import get_spot_level_image_count
from instapath.utils_general import set_seed, save_pickle, load_pickle

set_seed(2025) 
id_list = ['NCBI776']
studyID = id_list[0]
inputs_dir = Path("./inputs/VisiumBC")
inputs_dir.mkdir(parents=True, exist_ok=True)
#%%
# gene info 
adata = sc.read_h5ad(Path(cfg['dir_st'], f"{studyID}.h5ad"))
adata = adata[adata.obs["in_tissue"] == 1].copy()
pos_x, pos_y = adata.obs['pxl_col_in_fullres'].to_numpy(), adata.obs['pxl_row_in_fullres'].to_numpy()

# add annotations
annotation_path = "./data/annotations/Cell_Barcode_Type_Matrices.xlsx" # This annotation file is from the VisiumBC paper, not from HEST dataset. 
ann = pd.read_excel(annotation_path, sheet_name="Visium")
adata.obs = adata.obs.join(
    ann.set_index("Barcode"),  # choose your columns
    how="left"
)
# find the top 3000 high variable genes
adata.var_names_make_unique()
adata = adata[:, ~adata.var_names.str.startswith('MT-')]
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
top_gene_matrix = adata[:, adata.var.highly_variable].X.toarray()

#%%
# Create new AnnData object
new_adata = ad.AnnData(
    X=csr_matrix(top_gene_matrix),
    obs = pd.DataFrame(index=adata.obs.index)
)

# Add gene names (variables)
new_adata.var_names = adata.var_names[adata.var['highly_variable']].to_numpy()

# Add selected obs columns
new_adata.obs["Annotation"] = adata.obs["Annotation"].to_numpy()
new_adata.obs["pxl_col_in_fullres"] = adata.obs["pxl_col_in_fullres"].to_numpy()
new_adata.obs["pxl_row_in_fullres"] = adata.obs["pxl_row_in_fullres"].to_numpy()
new_adata.obs["array_col"] = adata.obs["array_col"].to_numpy()
new_adata.obs["array_row"] = adata.obs["array_row"].to_numpy()

# Save
new_adata.write(Path(inputs_dir, "VisiumBC_hvg3000.h5ad"))

#%%
# image info
vq_results = load_pickle(Path(cfg['dir_uni_vq_features'], f"{studyID}.pickle"))
patch_paras = load_pickle(Path(cfg['dir_wsi_crop_paras'], f"{studyID}.pickle"))
center_list = load_pickle(Path(cfg['dir_patch_center_coords'], f"{studyID}.pickle"))
X_count_token = vq_results["X_count_token"]
X_count_patch = vq_results["X_count_patch"].toarray()
encodings_idx = vq_results["encodings_idx"]
n_row = patch_paras["n_row"]
n_col = patch_paras["n_col"]

# calculate spot level image word count
spot_dist = load_pickle(Path(cfg['dir_spot_distance'], f'{studyID}.pickle'))
spot_rad = (spot_dist/100) * (55/2) # Visium has spot diameter of 55um, and interspot distance of 100 um 
raw_img = get_spot_level_image_count(center_list, adata, X_count_token, spot_rad)

# identify patch level highly variable image words
W = X_count_patch.shape[1]  # number of words
img_adata_0 = ad.AnnData(X=X_count_patch, var=pd.DataFrame(index=[f"meta_{i}" for i in range(W)])) 
sc.pp.highly_variable_genes(img_adata_0, flavor="seurat_v3", n_top_genes=256)

#%% save
img_adata = ad.AnnData(X=raw_img, obs = pd.DataFrame(index=adata.obs.index), var=pd.DataFrame(index=[f"meta_{i}" for i in range(W)])) 
img_adata = img_adata[:, img_adata_0.var['highly_variable']].copy()
top_img_matrix = img_adata.X
top_img_names = img_adata.var.index.to_numpy()

img_adata_hvi = ad.AnnData(X=top_img_matrix, obs = pd.DataFrame(index=adata.obs.index), var=pd.DataFrame(index=top_img_names)) 
img_adata_hvi.X = csr_matrix(img_adata_hvi.X)
img_adata_hvi.write(Path(inputs_dir, "visium_BC_img_hvi256.h5ad"))

#%%
from scipy.sparse import save_npz
save_npz(Path(inputs_dir,"VisiumBC_codebook512&64_token_level.npz"), X_count_token)