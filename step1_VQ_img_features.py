# -*- coding: utf-8 -*-
'''
HEST-1K: https://github.com/mahmoodlab/HEST/tree/main
UNI weights: https://huggingface.co/MahmoodLab/UNI2-h/tree/main
'''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
torch.use_deterministic_algorithms(True)
from instapath.utils_general import create_directories, download_hest, delete_file
from instapath.utils_patchify import read_wsi_region, show_image, initialize_uni_model, get_spot_distance, plot_downsampled_image, plot_spots_on_image, get_image_tiles, get_uni_features, get_vq_features
from instapath.config import cfg
from pathlib import Path

#%%
create_directories(cfg)

from huggingface_hub import login
login(token="your huggingface login token")
id_list = ['NCBI776'] # This is the VisiumBC slide
patterns = []
for sid in id_list:
    patterns.append(f"st/{sid}.h5ad")
    patterns.append(f"wsis/{sid}.tif")
download_dir='./data'
download_hest(patterns, download_dir)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uni_path = "Path to uni2-h model weights" # UNI2-h model weights: https://huggingface.co/MahmoodLab/UNI2-h/tree/main
model = initialize_uni_model(uni_path, device)

#%%
import gc
def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()  # doesn’t free to OS, but helps fragmentation sometimes
    
#%%
studyID = id_list[0]
# output to: f'./processed/spot_distance/{studyID}.pickle'
path = Path(f"./processed/spot_distance/{studyID}.pickle")
if path.exists():
    print("File exists: ", f"./processed/spot_distance/{studyID}.pickle")
else:
    print("File does not exist: ", f"./processed/spot_distance/{studyID}.pickle")
    spot_distance = get_spot_distance(studyID)
    print("spot_distance (px): ", spot_distance)

#%%
# output to: f"./processed/plot_downsampled_image_x20/{studyID}.png"
path = Path(f"./processed/plot_downsampled_image_x20/{studyID}.png")
if path.exists():
    print("File exists: ", f"./processed/plot_downsampled_image_x20/{studyID}.png")
else:
    print("File does not exist: ", f"./processed/plot_downsampled_image_x20/{studyID}.png")
    plot_downsampled_image(studyID, k=20)
    
#%%
# output to: f"./processed/plot_spots_on_image/{studyID}.png"
path = Path(f"./processed/plot_spots_on_image/{studyID}.png")
if path.exists():
    print("File exists: ", f"./processed/plot_spots_on_image/{studyID}.png")
else:
    print("File does not exist: ", f"./processed/plot_spots_on_image/{studyID}.png")
    plot_spots_on_image(studyID)
    
#%% crop the region with tissue
import numpy as np
side = np.round(224*0.5*spot_distance/100)  # target: 224 px, 0.5 um per px, spot distance: 100 um
side = int(side)
anchor = (2160, 1940)
n_row = 84
n_col = 74
tissue_region = read_wsi_region(
    img_path=Path(cfg['dir_wsis'], f"{studyID}.tif"),
    anchor=anchor,
    n_row=n_row,
    n_col=n_col,
    side=side,
)

show_image(tissue_region, invert_y=True, k=1)
img_paras = {'anchor': anchor, 'side': side, 'n_row': n_row, 'n_col': n_col}

#%%
# output to: 
# plot_patch_anchors: f"./processed/plot_patch_anchors/{studyID}.png"
# tiles: f'./processed/visium_tiles/{studyID}.pickle'
# patch_paras: f'./processed/uni_patch_paras/{studyID}.pickle'
path = Path(f'./processed/uni_patch_paras/{studyID}.pickle')
if path.exists():
    print("File exists: ", f'./processed/uni_patch_paras/{studyID}.pickle')
else:
    print("File does not exist: ", f'./processed/uni_patch_paras/{studyID}.pickle')
    _, _ = get_image_tiles(studyID, anchor=anchor, n_row=n_row, n_col=n_col, side=side)
    
#%%
path = Path(f'./processed/uni_features/{studyID}.pickle')
if path.exists():
    print("File exists: ", f'./processed/uni_features/{studyID}.pickle')
else:
    print("File does not exist: ", f'./processed/uni_features/{studyID}.pickle')
    _ = get_uni_features(studyID, model, batch_size=512) # change to smaller batch_size if there is memory error
    
#%%
path = Path(f'./processed/uni_vq_features/{studyID}.pickle')
if path.exists():
    print("File exists: ", f'./processed/uni_vq_features/{studyID}.pickle')
else:
    print("File does not exist: ", f'./processed/uni_vq_features/{studyID}.pickle')
    get_vq_features(studyID, device)

cleanup_cuda()

#%%
