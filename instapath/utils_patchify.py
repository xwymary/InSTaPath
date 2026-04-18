import scanpy as sc
import numpy as np
import scipy.sparse as sp
from .utils_general import save_pickle, load_pickle
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms
import torch
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from .config import cfg
import timm
from scipy.spatial import cKDTree
from sklearn.metrics import pairwise_distances_argmin
from scipy.sparse import csr_matrix

def initialize_uni_model(uni_path, device):
    uni_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2, # mlp_dim: embed_dim x mlp_ratio = 8192
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }

    model = timm.create_model(**uni_kwargs)
    model.load_state_dict(torch.load(uni_path, map_location="cpu"), strict=True)
    model.to(device)
    return model
    
    
def get_spot_distance(studyID):
    # ST (adata):
    adata = sc.read_h5ad(Path(cfg['dir_st'], f"{studyID}.h5ad"))
    df = adata.obs[['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']].copy()

    # sample 10 random centers
    random_rows = df.sample(n=10, replace=False, random_state=0)
    distance_list = []

    for _, random_row in random_rows.iterrows():
        row = random_row['array_row']
        col = random_row['array_col']
        ctr = np.array([random_row['pxl_row_in_fullres'], random_row['pxl_col_in_fullres']])

        coor = []
        r1_neighbors = [
            (row, col-2), (row, col+2),
            (row-1, col-1), (row-1, col+1),
            (row+1, col-1), (row+1, col+1)
        ]

        for i, j in r1_neighbors:
            neighbor = df[ (df['array_row'] == i) & (df['array_col'] == j)]
            if not neighbor.empty:
                coor.append(np.array([neighbor['pxl_row_in_fullres'].iloc[0], neighbor['pxl_col_in_fullres'].iloc[0]]))

        # skip if no valid neighbors
        if len(coor) == 0:
            continue

        distances = [np.linalg.norm(ctr - pt) for pt in coor]
        mean_distance = np.mean(distances)
        distance_list.append(mean_distance)

    distance_agg_mean = np.array(distance_list).mean()
    save_pickle(distance_agg_mean, Path(cfg['dir_spot_distance'], f'{studyID}.pickle'))
    return distance_agg_mean


def plot_downsampled_image(studyID, k=20):
    '''
    :param k: scale_factor
    '''
    Image.MAX_IMAGE_PIXELS = None  # e.g., allow up to 600M px

    # read tif image
    img_pil = Image.open(Path(cfg['dir_wsis'], f"{studyID}.tif")) 
    W, H = img_pil.size
    img_pil = img_pil.resize((W//k, H//k), resample=Image.Resampling.BILINEAR)
    img_pil = np.asarray(img_pil)

    # plot spots on image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    plt.gca().invert_yaxis() 
    plt.title(f"Downscaled x{k}")
    plt.savefig(Path(cfg['dir_plot_downsampled_image_x20'], f"{studyID}.png"), dpi=300, bbox_inches="tight")
    plt.close() 


def plot_spots_on_image(studyID, k=20):
    '''
    :param k: scale_factor
    '''
    Image.MAX_IMAGE_PIXELS = None  # e.g., allow up to 600M px

    # read tif image
    img_pil = Image.open(Path(cfg['dir_wsis'], f"{studyID}.tif")) 
    W, H = img_pil.size
    img_pil = img_pil.resize((W//k, H//k), resample=Image.Resampling.BILINEAR)
    img_pil = np.asarray(img_pil)

    # get spot positions
    adata = sc.read_h5ad(Path(cfg['dir_st'], f"{studyID}.h5ad"))
    adata = adata[adata.obs["in_tissue"] == 1].copy()
    pos_x, pos_y = adata.obs['pxl_col_in_fullres'].to_numpy()/k, adata.obs['pxl_row_in_fullres'].to_numpy()/k
     
    # plot spots on image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    plt.scatter(pos_x, pos_y, s=2)
    plt.gca().invert_yaxis() 
    plt.title(f"Downscaled x{k}")
    plt.savefig(Path(cfg['dir_plot_spots_on_image'], f"{studyID}.png"), dpi=300, bbox_inches="tight")
    plt.close() 


def read_wsi_region(
    img_path,
    anchor=None,
    n_row=None,
    n_col=None,
    side=None,
    flip_ud=False,
    max_image_pixels=None,
):
    """
    Read a whole image or a cropped region.

    Parameters
    ----------
    img_path : str or Path
        Path to TIFF/WSI image.
    anchor : tuple[int, int] or None
        Upper-left corner (x, y) of crop in full-resolution pixels.
    n_row, n_col : int or None
        Number of patch rows/cols for crop.
    side : int or None
        Patch side length in pixels.
    flip_ud : bool
        Whether to vertically flip the image.
    max_image_pixels : int or None
        If not None, assign to Image.MAX_IMAGE_PIXELS before reading.

    Returns
    -------
    np.ndarray
    """
    if max_image_pixels is not None:
        Image.MAX_IMAGE_PIXELS = max_image_pixels
    else:
        Image.MAX_IMAGE_PIXELS = None

    img = np.asarray(Image.open(img_path))

    if flip_ud:
        img = np.flipud(img)

    if anchor is None:
        return img

    if n_row is None or n_col is None or side is None:
        raise ValueError("anchor requires n_row, n_col, and side.")

    x0, y0 = anchor
    return img[y0:y0 + n_row * side, x0:x0 + n_col * side, :]


def show_image(
    img,
    invert_y=True,
    axis_off=False,
    figsize=None,
    grid_step=None,
    title=None,
    k=None,
):
    """
    Image display helper.

    Parameters
    ----------
    img : np.ndarray
        Image array (H, W, C)
    invert_y : bool
        Whether to invert y-axis
    axis_off : bool
        Hide axes if True
    figsize : tuple or None
        Matplotlib figure size
    grid_step : int or None
        Step size for grid overlay (in pixels, BEFORE downsampling)
    title : str or None
        Plot title
    k : int or None
        Downsampling factor (e.g. k=20 means show image at 1/20 resolution)
    """

    # ---- optional downsampling
    if k is not None and k > 1:
        img = img[::k, ::k]

        # adjust grid spacing accordingly
        if grid_step is not None:
            grid_step = grid_step // k

    # ---- figure setup
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()

    plt.imshow(img)

    if invert_y:
        plt.gca().invert_yaxis()

    if axis_off:
        plt.axis("off")

    # ---- draw grid
    if grid_step is not None and grid_step > 0:
        h, w = img.shape[:2]
        for y in range(grid_step, h, grid_step):
            plt.axhline(y=y, color="k", lw=1, alpha=0.7)
        for x in range(grid_step, w, grid_step):
            plt.axvline(x=x, color="k", lw=1, alpha=0.7)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()
    
    
def get_image_tiles(studyID, anchor=None, n_row=None, n_col=None, side=None):
    Image.MAX_IMAGE_PIXELS = None  # e.g., allow up to 600M px
    spot_dist = load_pickle(Path(cfg['dir_spot_distance'], f'{studyID}.pickle'))

    # 1. read tif image, crop the region we need
    img = Image.open(Path(cfg['dir_wsis'], f"{studyID}.tif"))  
    img = np.asarray(img)
    # default anchor if not provided
    if anchor is None:
        anchor = (0, 0)
    
    # compute side only if not provided
    if side is None:
        side = int(np.round(224 * 0.5 * spot_dist / 100))
    
    # compute n_row only if not provided
    if n_row is None:
        n_row = (img.shape[0] - anchor[1]) // side
    
    # compute n_col only if not provided
    if n_col is None:
        n_col = (img.shape[1] - anchor[0]) // side

    tissue_region = img[anchor[1]:anchor[1]+n_row*side, anchor[0]:anchor[0]+n_col*side, :]

    # 2. extract patches
    tiles = rearrange(tissue_region, '(r h) (c w) ch -> (r c) h w ch', r=n_row, c=n_col)
        
    # upper left corner coords
    anchor_list = []
    for i in range(n_row):
        for j in range(n_col):
            anchor_x = anchor[0] + j * side
            anchor_y = anchor[1] + i * side
            anchor_list.append((anchor_x, anchor_y))

    anchor_list = np.array(anchor_list)

    # patch center coords (ONE)
    patch_side = side/16
    patch_centers=[]
    for i in range(16):
        for j in range (16):
            anchor_x = j * patch_side + patch_side/2
            anchor_y = i * patch_side + patch_side/2
            patch_centers.append((anchor_x,anchor_y))
            
    patch_centers = np.array(patch_centers)

    # patch center coords (ALL)
    center_list=[]
    for coor in anchor_list:
        center_list.append(coor+patch_centers)
        
    center_list = np.array(center_list)

    # save
    save_pickle(tiles, Path(cfg['dir_visium_tiles'], f'{studyID}.pickle'))

    img_paras = {'anchor': anchor, 'side': side, 'n_row': n_row, 'n_col': n_col}
    save_pickle(img_paras, Path(cfg['dir_wsi_crop_paras'], f'{studyID}.pickle'))
    save_pickle(center_list, Path(cfg['dir_patch_center_coords'], f'{studyID}.pickle'))

    return tiles, center_list


def get_uni_features(studyID, model, batch_size=8):
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    image_list = load_pickle(Path(cfg['dir_visium_tiles'], f'{studyID}.pickle'))

    model.eval()
    outputs = {}

    def hook_fn(module, inputs, output):
        outputs["last_layer"] = output.detach()

    # register ONCE
    handle = model.blocks[-1].register_forward_hook(hook_fn)

    feats = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_list), batch_size)):
            batch = image_list[i:i + batch_size]

            # preprocess batch
            x = torch.stack([
                transform(Image.fromarray(arr).convert("RGB"))
                for arr in batch
            ]).to(device)                     # [B, 3, 224, 224]

            _ = model(x)

            feat = outputs["last_layer"]      # [B, 265, 1536]
            feats.append(feat.cpu().numpy())

    handle.remove()

    feats = np.concatenate(feats, axis=0)     # [N, 265, 1536]
    print(studyID, ", UNI_features:", feats.shape)

    save_pickle(feats, Path(cfg['dir_uni_features'], f"{studyID}.pickle"))
    return feats


def get_uni_features_old(studyID, model):
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    image_list = load_pickle(Path(cfg['dir_visium_tiles'], f'{studyID}.pickle'))

    model.eval()
    outputs = {}

    def hook_fn(module, inputs, output):
        outputs["last_layer"] = output.detach()

    # register ONCE
    handle = model.blocks[-1].register_forward_hook(hook_fn)

    feats = []
    with torch.no_grad():
        for arr in tqdm(image_list):
            image = Image.fromarray(arr).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            _ = model(x)
            feat = outputs["last_layer"]          # expected [1, 265, 1536]
            feats.append(feat.cpu().numpy())

    # remove hook
    handle.remove()

    feats = np.concatenate(feats, axis=0)         # [N, 265, 1536]
    print(studyID, ", UNI_features:", feats.shape)
    save_pickle(feats, Path(cfg['dir_uni_features'], f"{studyID}.pickle"))
    return feats


def get_vq_features(studyID, device):
    # Parameters
    num_embeddings = 512   # size of codebook
    embedding_dim = 64   # dimensionality of each vector (must match input dim)
    batch_size = num_embeddings*100
    num_head = 24
    head_dim = 64
    num_token = 256

    last_layer_list = load_pickle(Path(cfg['dir_uni_features'], f"{studyID}.pickle"))
    N = len(last_layer_list)
    last_layer_list = last_layer_list[:,9:,:] # [N, 256, 1532], 265 = 1cls + 8reg + 256token
    last_layer_list = last_layer_list.reshape(N, num_token, num_head, head_dim//embedding_dim, embedding_dim)
    last_layer_list = last_layer_list.reshape(-1, embedding_dim)

    # find cluster center
    mbk = MiniBatchKMeans(n_clusters=num_embeddings, batch_size=batch_size, verbose=1, random_state=42)
    mbk.fit(last_layer_list)
    centers = torch.tensor(mbk.cluster_centers_, dtype=torch.float32)
    centers = centers.to(device)

    # VQ step
    print("Find closest codebook indices")
    min_encoding_indices = []

    with torch.no_grad():
        for i in range(0, last_layer_list.shape[0], batch_size):
            X = torch.as_tensor(last_layer_list[i:i+batch_size], dtype=torch.float32, device=device)
            dist = torch.cdist(X, centers)                 # (B, num_embeddings)
            min_encoding_indices.append(dist.argmin(dim=-1))

    min_encoding_indices = torch.cat(min_encoding_indices, dim=0)  # (total_tokens,)

    def block_bincount(indices_1d: torch.Tensor, block_size: int, num_classes: int) -> np.ndarray:
        """Split indices_1d into blocks of block_size, return (n_blocks, num_classes) counts."""
        n_blocks = indices_1d.numel() // block_size
        x = indices_1d[: n_blocks * block_size].view(n_blocks, block_size)
        counts = torch.stack(
            [torch.bincount(row, minlength=num_classes) for row in x],
            dim=0
        )
        return counts.cpu().numpy()

    print("prepare encodings_idx")
    encodings_idx = min_encoding_indices.view(N, 256, 24) 

    print("prepare token level count data")
    token_block = num_head * (head_dim // embedding_dim) 
    X_count_token = block_bincount(min_encoding_indices, token_block, num_embeddings)

    print("prepare patch level count data")
    patch_block = num_token * num_head * (head_dim // embedding_dim)
    X_count_patch = block_bincount(min_encoding_indices, patch_block, num_embeddings)

    # save
    save = {
        "encodings_idx": encodings_idx.cpu().numpy(),
        "X_count_token": sp.csr_matrix(X_count_token),
        "X_count_patch": sp.csr_matrix(X_count_patch),
        "code_book": centers.cpu().numpy(),
    }

    save_pickle(save, Path(cfg['dir_uni_vq_features'], f"{studyID}.pickle"))


def get_spot_level_image_count(center_list, adata, X_count_token, spot_rad):
    # note: X_count_token = vq_results["X_count_token"]   # CSR matrix (n_tokens, n_vocab)
    # centers: (n_tokens, 2)
    center_list = center_list.reshape(-1,2)
    centers = center_list.astype(np.float32)

    # spots: (n_spots, 2)
    pos_x, pos_y = adata.obs['pxl_col_in_fullres'].to_numpy(), adata.obs['pxl_row_in_fullres'].to_numpy()
    spots = np.column_stack([pos_x, pos_y]).astype(np.float32)

    # radius per spot: make sure it's (n_spots,)
    # If your spot_rad is scalar, this still works (broadcast below)
    r = np.asarray(spot_rad, dtype=np.float32)
    if r.ndim == 0:
        r = np.full(spots.shape[0], float(r), dtype=np.float32)

    # build KD-tree once
    tree = cKDTree(centers)

    # query neighbors (multi-threaded)
    # returns a python list of index arrays, length = n_spots
    nbrs = tree.query_ball_point(spots, r, workers=-1)

    # aggregate
    raw_img = np.zeros((spots.shape[0], X_count_token.shape[1]), dtype=np.float32)

    for i, idx in enumerate(tqdm(nbrs, total=len(nbrs))):
        if len(idx):
            # sum sparse rows -> (1, V); convert to 1d
            raw_img[i] = np.asarray(X_count_token[idx].sum(axis=0)).ravel()

    return raw_img


def get_global_codebook_and_maps(
    codebooks,                 # list of 8 arrays, each (512, 64)
    n_global=512,
    random_state=42,
    batch_size=2048,
    n_init="auto",
    max_iter=200,
):
    """
    Returns
    -------
    global_codebook : (n_global, 64) float32
    maps            : list of N arrays, each (512,) int64
                      maps[i][k] = global index for old key k in codebook i
    kmeans          : fitted MiniBatchKMeans object
    """

    D = codebooks[0].shape[1]
    for cb in codebooks:
        assert cb.shape == (512, D), f"Expected (512,{D}), got {cb.shape}"

    # 1) Stack all codewords: (8*512, 64) = (4096, 64)
    X = np.vstack([cb.astype(np.float32, copy=False) for cb in codebooks])

    # 2) KMeans -> global codebook
    kmeans = MiniBatchKMeans(
        n_clusters=n_global,
        batch_size=batch_size,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0,
    )
    kmeans.fit(X)
    global_codebook = kmeans.cluster_centers_.astype(np.float32)

    # 3) Build maps: nearest centroid index for each old codeword
    maps = []
    for cb in codebooks:
        Y = cb.astype(np.float32, copy=False)

        # argmin over distances to centroids => (512,)
        idx = pairwise_distances_argmin(Y, global_codebook, metric="euclidean")
        maps.append(idx.astype(np.int64))

    return global_codebook, maps


def remap_counts_sparse(counts, old2new, n_global=1024):
    """
    counts   : (N, 512)
    old2new  : (512,)
    returns  : (N, 1024)
    """
    N, _ = counts.shape

    # Build aggregation matrix A: (512 → 1024)
    A = csr_matrix(
        (np.ones(len(old2new)),
         (np.arange(len(old2new)), old2new)),
        shape=(len(old2new), n_global)
    )

    # (N,512) @ (512,1024) → (N,1024)
    return counts @ A