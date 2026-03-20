# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 21:23:57 2026

@author: xwyma
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import squidpy as sq
from einops import rearrange
from pathlib import Path
from scipy.sparse import load_npz
import matplotlib as mpl
def plot_tsne_visualization(data, labels, title, special_color_map):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        mask = labels == label
        color = special_color_map[label]   # get color from your dict

        plt.scatter(
            data[mask, 0], data[mask, 1],
            c=[color],            # wrap in list to avoid matplotlib warnings
            label=label,
            s=1,
            alpha=0.7
        )

    # Add legend
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Class",
        markerscale=4,
        frameon=False
    )

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)
    plt.show()

def plot_tsne_two_panel(
    data,
    gt_labels,
    cluster_labels,
    special_color_map,
    title_left="Ground truth",
    title_right="Clustering",
    figsize=(14, 6),
    point_size=1
):
    gt_labels = np.asarray(gt_labels)
    cluster_labels = np.asarray(cluster_labels)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # ---------------- Left: Ground truth (special color map) ----------------
    ax = axes[0]
    unique_gt = np.unique(gt_labels)

    for label in unique_gt:
        mask = gt_labels == label
        color = special_color_map[label]

        ax.scatter(
            data[mask, 0], data[mask, 1],
            c=[color],
            s=point_size,
            alpha=0.7,
            label=label
        )

    ax.set_title(title_left)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Class",
        markerscale=4,
        frameon=False
    )

    # ---------------- Right: Clustering (default matplotlib colors) ----------------
    ax = axes[1]
    unique_cluster = np.unique(cluster_labels)

    for label in unique_cluster:
        mask = cluster_labels == label

        ax.scatter(
            data[mask, 0], data[mask, 1],
            s=point_size,
            alpha=0.7,
            label=label
        )

    ax.set_title(title_right)
    ax.set_xlabel("t-SNE Component 1")
    ax.grid(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Cluster",
        markerscale=4,
        frameon=False
    )

    plt.tight_layout()
    plt.show()

    return fig, axes

    
    
#%%
def plot_theta_scatter_10topics(adata, special_color_map, theta, study_name, title):
    pos_x = adata.obs['pxl_row_in_fullres'].to_numpy()
    pos_y = adata.obs['pxl_col_in_fullres'].to_numpy()

    category = list(adata.obs["Annotation"].cat.categories)
    annotations = adata.obs["Annotation"].to_numpy()
    
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(1, 11, figsize=(18, 3))  # 1 row, 11 columns

    for i in range(11):
        if i == 0:
            # Plot categorical annotations using special_color_map
            for cat in category:
                idx = (annotations == cat)
                color = special_color_map[cat]

                axes[i].scatter(
                    pos_x[idx],
                    pos_y[idx],
                    s=1,
                    color=color,
                    label=cat
                )
            axes[i].set_title(f'{study_name}')

        else:
            # Plot theta values
            scatter = axes[i].scatter(
                pos_x,
                pos_y,
                s=2,
                c=theta[:, i-1],
                cmap='coolwarm'
            )
            axes[i].set_title(f'topic {i}')

        axes[i].axis('off')
        axes[i].set_aspect('equal')

    fig.suptitle(title, x=0.51, y=0.9, fontsize=14)
    plt.show()
    
    
def plot_theta_scatter(theta, adata, n_rows=2, figsize=(20, 6), title=None, vmin_list=None, vmax_list=None):
    """
    Plot all topic spatial maps in a grid.

    theta : array (N_spots, K)
        Topic proportions.
    adata : AnnData
        Contains spatial coordinates in obs.
    n_rows : int
        Number of rows in the subplot grid (e.g. 2 for 20 topics, 3 for 30 topics).
    figsize : tuple
        Figure size.
    title : str or None
        Optional global title.
    """

    pos_x = adata.obs['pxl_row_in_fullres'].to_numpy()
    pos_y = adata.obs['pxl_col_in_fullres'].to_numpy()

    K = theta.shape[1]                  # number of topics
    n_cols = int(np.ceil(K / n_rows))   # columns automatically
    
    if vmin_list is None:
        vmin_list = [None]*K
        
    if vmax_list is None:
        vmax_list = [None]*K
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)   # flatten in case of 2D grid

    for i in range(K):
        ax = axes[i]
        sc = ax.scatter(
            pos_x,
            pos_y,
            s=2,
            c=theta[:, i],
            cmap="coolwarm",
            vmin=vmin_list[i], 
            vmax=vmax_list[i]
        )
        ax.set_title(f"Topic {i+1}", fontsize=10)
        ax.axis("off")
        ax.set_aspect("equal")

    # Turn off unused subplots
    for j in range(K, len(axes)):
        axes[j].axis("off")

    if title is not None:
        fig.suptitle(title, y=0.98, fontsize=14)

    plt.tight_layout()
    plt.show()
#%%
def plot_beta_heatmap(beta, word_list, topics_list, title, top_n=5, plot=True, vmax=0.10, figsize=(8, 12)):

    # beta: topic-to-word probabilities
    # recon['beta'] is assumed shape (K, num_words), so transpose to (num_words, K)
    df_beta = pd.DataFrame(
        beta.T,
        index=word_list,
        columns=topics_list
    )

    # find top-N words per topic
    top_words_list = []
    for c in topics_list:
        top_words_list += df_beta[c].nlargest(top_n).index.tolist()

    # keep unique words but preserve order
    # top_words_list = list(dict.fromkeys(top_words_list))

    top_words_data = df_beta.loc[top_words_list]

    # plot heatmap
    if plot: 
        plt.figure(figsize=figsize)
        cmap = sns.color_palette("Reds", as_cmap=True)
        sns.heatmap(
            top_words_data,
            linewidths=2,
            cmap=cmap,
            vmax=vmax,
            cbar_kws={"shrink": 0.5}
        )
        plt.title(title, fontsize=14)
        plt.xlabel("Topics")
        plt.ylabel("Words")
        plt.tight_layout()
        plt.show()

    return df_beta, top_words_list

#%%
def plot_theta_clustermap(theta, adata, special_color_map, topics_list, title="spot-topic dist."):
    # Annotation list in the same order as rows of theta
    spot_list = adata.obs["Annotation"].to_list()

    # Map each spot to its color using your predefined map
    row_colors = [special_color_map[spot] for spot in spot_list]
    
    # Plot clustermap
    g = sns.clustermap(
        theta,
        cmap=sns.color_palette("Reds", as_cmap=True),
        row_colors=row_colors,
        xticklabels=topics_list,
        yticklabels=False
    )
    g.fig.suptitle(title, y=1.01)
    g.ax_heatmap.grid(False)

    # Create legend from special_color_map
    category = list(adata.obs["Annotation"].cat.categories)
    handles = [
        Patch(facecolor=special_color_map[cat], label=cat)
        for cat in category
        if cat in special_color_map
    ]

    # Place legend outside the heatmap
    g.ax_heatmap.legend(
        handles=handles,
        title="Cell Types",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False
    )

    plt.show()
    col_order = g.dendrogram_col.reordered_ind
    row_order = g.dendrogram_row.reordered_ind
    clustermap_order = {"col_order": col_order,
                        "row_order": row_order,
                        "row_colors": row_colors}
    return clustermap_order
    
#%%
def plot_top_doc_barcode_plus_heatmap(
    theta, adata, special_color_map, topics_list,
    figsize=(8, 20), line_every=50, vmax=None
):
    """
    Plot barcode (categorical) + theta heatmap (continuous) in one combined plot.

    theta: (N, K) array, full theta for all docs/spots
    top_indices: indices (length = rows) selecting subset to plot (same order for barcode+theta)
    layer_list: list of labels for all docs/spots (len = N)
    cluster_cat: list of categories for barcode columns (order matters)
    special_color_map: dict {category_name: RGB or RGBA}
    topics_list: list of topic names (length = K)
    vmax: optional float for Reds scaling
    """
    
    top_indices = np.argsort(theta, axis=0)[-50:][::-1]
    top_indices = top_indices.T.flatten()
    layer_list = adata.obs['Annotation'].to_list()
    cluster_cat = sorted(adata.obs['Annotation'].unique())
    
    # -----------------------------
    # 1) Build barcode color_matrix (rows x C) with ints in {0..C}
    # -----------------------------
    label_to_idx = {cat: i for i, cat in enumerate(cluster_cat)}
    top_labels = [layer_list[i] for i in top_indices]

    # Make sure labels exist in cluster_cat
    missing = sorted(set(top_labels) - set(cluster_cat))
    if missing:
        raise ValueError(f"These labels are missing from cluster_cat: {missing[:10]}")

    top_label_idx = np.asarray([label_to_idx[lbl] for lbl in top_labels], dtype=int)

    rows = len(top_label_idx)
    C = len(cluster_cat)
    barcode = np.zeros((rows, C), dtype=int)
    barcode[np.arange(rows), top_label_idx] = top_label_idx + 1  # 1..C (0 is background)

    # -----------------------------
    # 2) Build theta_sub (rows x K)
    # -----------------------------
    theta_sub = np.asarray(theta)[np.asarray(top_indices)]
    K = theta_sub.shape[1]
    if len(topics_list) != K:
        raise ValueError(f"topics_list length ({len(topics_list)}) != theta columns ({K})")

    # -----------------------------
    # 3) Build discrete cmap for barcode
    # -----------------------------
    palette = [(1, 1, 1, 1)]  # 0 -> white background
    for cat in cluster_cat:
        col = special_color_map.get(cat, (0.7, 0.7, 0.7, 1))
        if len(col) == 3:
            col = (*col, 1.0)
        palette.append(col)

    cmap_bar = ListedColormap(palette)
    bounds = np.arange(len(palette) + 1) - 0.5
    norm_bar = BoundaryNorm(bounds, cmap_bar.N)

    # -----------------------------
    # 4) Concatenate matrices (for tick placement / shape consistency)
    # -----------------------------
    combined = np.concatenate([barcode.astype(float), theta_sub], axis=1)

    # -----------------------------
    # 5) Plot: theta first (Reds) then overlay barcode on left columns
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Draw theta heatmap on the right region only using a mask
    mask_theta = np.zeros_like(combined, dtype=bool)
    mask_theta[:, :C] = True  # mask barcode region
    sns.heatmap(
        combined,
        ax=ax,
        mask=mask_theta,
        cmap=sns.color_palette("Reds", as_cmap=True),
        cbar=True,
        yticklabels=False,
        xticklabels=False,
        vmax=vmax
    )

    # Overlay barcode heatmap on the left region only
    mask_barcode = np.zeros_like(combined, dtype=bool)
    mask_barcode[:, C:] = True  # mask theta region
    sns.heatmap(
        combined,
        ax=ax,
        mask=mask_barcode,
        cmap=cmap_bar,
        norm=norm_bar,
        cbar=False,
        yticklabels=False,
        xticklabels=False
    )

    # -----------------------------
    # 6) X tick labels: cluster columns + topic columns
    # -----------------------------
    xticks = np.arange(C + K) + 0.5
    xlabels = list(cluster_cat) + list(topics_list)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)

    # Vertical separator between barcode and theta
    ax.axvline(C, color="black", linewidth=1.0)

    # Horizontal separator lines
    if line_every is not None and line_every > 0:
        for i in range(line_every, rows, line_every):
            ax.axhline(i, color="black", linewidth=0.8)

    ax.set_title("Barcode (left) + Topic proportions (right)")
    plt.tight_layout()
    plt.show()

    return fig, ax

#%%
def plot_top_doc_barplot(theta, adata, special_color_map, K, title="Category Composition of Top 50 Spots per Topic"):
    top_indices = np.argsort(theta, axis=0)[-50:][::-1]
    top_indices = top_indices.T.flatten()
    layer_list = adata.obs['Annotation'].to_list()
    cluster_cat = sorted(adata.obs['Annotation'].unique())
    
    # Convert top indices to category indices
    top_image_label_str = np.asarray(
        [cluster_cat.index(layer_list[i]) for i in top_indices], dtype=int
    )

    # Synthetic structure: 50 docs per topic
    categories = top_image_label_str
    topics = np.repeat(np.arange(K), 50)

    # Calculate proportions: (K topics) x (num categories)
    proportions = np.zeros((K, len(cluster_cat)))
    for topic in range(K):
        topic_indices = topics == topic
        for category in range(len(cluster_cat)):
            proportions[topic, category] = (
                np.sum(categories[topic_indices] == category) / 50
            )

    # Plot
    x = np.arange(K)
    bar_width = 0.8
    fig, ax = plt.subplots(figsize=(12, 6))

    bottoms = np.zeros(K)

    for category in range(len(cluster_cat)):
        cat_name = cluster_cat[category]
        color = special_color_map[cat_name]   # <-- use your special colormap

        ax.bar(
            x,
            proportions[:, category],
            width=bar_width,
            bottom=bottoms,
            color=color,
            label=cat_name
        )
        bottoms += proportions[:, category]

    # Customize
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(title, fontsize=12, y=1.01)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Topic {i+1}" for i in range(K)], fontsize=12, rotation=45, ha="right")

    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=12,
        ncol=1,
        frameon=False
    )

    ax.grid(False)
    plt.tight_layout()
    plt.show()
    
#%%
def plot_spatial_clusters(
    adata_sub,
    theta_cluster,
    special_color_map,
    ari=None, asw=None, nmi=None,
    figsize=(10, 4),
    point_size=1,
    title="theta"
):
    """
    Plot Ground Truth (Annotation) and theta_cluster side-by-side (1x2 subplot)
    using special_color_map for Annotation.

    Parameters
    ----------
    adata_sub : AnnData
        Subset AnnData object.
    theta_cluster : list or array
        Cluster labels for each spot (same order as adata_sub.obs).
    special_color_map : dict
        {category_name: color} mapping for Annotation.
    ari, asw, nmi : float, optional
        Metrics to show in the title of the theta plot.
    figsize : tuple
        Figure size.
    point_size : int
        Dot size in spatial plot.
    """

    adata_plot = adata_sub.copy()

    # ---- spatial coordinates
    pos_x = adata_plot.obs["pxl_row_in_fullres"].to_numpy()
    pos_y = adata_plot.obs["pxl_col_in_fullres"].to_numpy()
    adata_plot.obsm["spatial"] = np.asarray([pos_x, -pos_y]).T

    # ---- add theta cluster
    adata_plot.obs["theta_cluster"] = theta_cluster

    # ---- prepare Annotation categories and palette
    adata_plot.obs["Annotation"] = adata_plot.obs["Annotation"].astype("category")
    ann_cats = list(adata_plot.obs["Annotation"].cat.categories)

    # reorder to match special_color_map
    ordered_cats = [c for c in special_color_map.keys() if c in ann_cats]
    adata_plot.obs["Annotation"] = adata_plot.obs["Annotation"].cat.reorder_categories(
        ordered_cats, ordered=True
    )

    # build ListedColormap
    palette_list = [special_color_map[c] for c in adata_plot.obs["Annotation"].cat.categories]
    ann_cmap = ListedColormap(palette_list)

    # ---- create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Ground truth
    sq.pl.spatial_scatter(
        adata_plot,
        color="Annotation",
        palette=ann_cmap,
        size=point_size,
        img=False,
        ax=axes[0]
    )
    axes[0].set_title("Ground truth")

    # Right: theta cluster
    if ari is not None and asw is not None and nmi is not None:
        title_theta = f"{title}, ARI = {ari:.4f}, ASW = {asw:.4f}, NMI = {nmi:.4f}"
    else:
        title_theta = "{title} cluster"

    sq.pl.spatial_scatter(
        adata_plot,
        color="theta_cluster",
        size=point_size,
        img=False,
        ax=axes[1]
    )
    axes[1].set_title(title_theta)

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_spatial_clusters_v2(
    x, y,
    annotations,
    theta_cluster,
    special_color_map,
    ari=None, asw=None, nmi=None,
    figsize=(10, 4),
    point_size=1,
    title="theta"
):
    # ---- process annotation
    ann_cat = pd.Categorical(annotations)
    ordered_cats = [c for c in special_color_map.keys() if c in ann_cat.categories]
    ann_cat = ann_cat.reorder_categories(ordered_cats, ordered=True)

    ann_codes = ann_cat.codes
    ann_cmap = ListedColormap([special_color_map[c] for c in ann_cat.categories])

    # ---- process theta cluster
    theta_cat = pd.Categorical(theta_cluster)
    theta_codes = theta_cat.codes
    theta_cats = theta_cat.categories

    base_cmap = mpl.colormaps["tab10"]
    colors_theta = [base_cmap(i) for i in range(len(theta_cats))]
    cmap_theta = ListedColormap(colors_theta)

    # ---- plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Ground truth
    axes[0].scatter(x, y, c=ann_codes, cmap=ann_cmap, s=point_size)
    axes[0].set_aspect("equal")
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    # legend (annotation)
    handles0 = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=special_color_map[c], label=c, markersize=5)
        for c in ann_cat.categories
    ]
    axes[0].legend(handles=handles0, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    # Right: theta cluster
    if ari is not None and asw is not None and nmi is not None:
        title_theta = f"{title}, ARI={ari:.3f}, ASW={asw:.3f}, NMI={nmi:.3f}"
    else:
        title_theta = f"{title} cluster"

    axes[1].scatter(x, y, c=theta_codes, cmap=cmap_theta, s=point_size)
    axes[1].set_aspect("equal")
    axes[1].set_title(title_theta)
    axes[1].axis("off")

    # legend (theta)
    handles1 = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=cmap_theta(i), label=str(cat), markersize=5)
        for i, cat in enumerate(theta_cats)
    ]
    axes[1].legend(handles=handles1, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    plt.show()

    return fig, axes

#%%
def plot_topic_word_occurrence_maps(
    encodings_idx, 
    top_words_list,
    K,
    H_patches=74,
    W_patches=84,
    S=16,
    nrows=2,
    ncols=5,
    figsize=(20, 8),
    crop=None
):
    """
    Plot spatial occurrence maps of top words for each topic.

    Parameters
    ----------
    vq_results : dict
        Must contain "encodings_idx", a 1D or 2D array of token indices.
    top_words_list : list
        List of top words per topic, concatenated (e.g. 5 per topic).
        Example: [word_t1_1, ..., word_t1_5, word_t2_1, ..., word_t2_5, ...]
    K : int
        Number of topics to plot.
    H_patches, W_patches : int
        Number of patch rows and columns in the spatial grid.
    S : int
        Patch size (each patch expands to S x S pixels).
    nrows, ncols : int
        Layout of the subplot grid.
    figsize : tuple
        Size of the matplotlib figure.
    """

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for topic_idx in range(K):
        print(f"topic {topic_idx+1}!")

        # Select top words for this topic (assuming 5 words per topic)
        target_words = top_words_list[topic_idx * 5 : (topic_idx + 1) * 5]
        # Extract integer ids from strings like "word_123" or "token_123"
        target_words = [int(word[5:]) for word in target_words]

        # Count occurrences
        counts = np.isin(encodings_idx, target_words).sum(axis=1)
        counts = counts.reshape(H_patches*W_patches, S*S)  # same shape as your pipeline

        # Stitch into big image
        big = rearrange(
            counts,
            "(r c) (h w) -> (r h) (c w)",
            r=H_patches,
            c=W_patches,
            h=S,
            w=S
        )

        # Ensure valid range for visualization
        if big.dtype != np.uint8:
            big = np.clip(big, 0, 1)

        # Plot
        if crop is not None:
            r0, nr, c0, nc = crop
            big = big[r0*S:r0*S+nr*S, c0*S:c0*S+nc*S]
        ax = axes[topic_idx]
        ax.imshow(big)
        ax.set_title(f"Topic {topic_idx+1}")
        ax.invert_yaxis()
        ax.axis("off")

    # Hide unused subplots if K < nrows*ncols
    for i in range(K, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return fig, axes


#%%
def plot_topic_and_top_words(
    adata,
    df_beta,
    top_words_list,
    hvg_names,
    theta,
    X_data,
    n_topics=10,
    top_n_per_topic=5,
    figsize=(15, 30),
    topic_cmap="coolwarm",
    word_cmap="viridis",
    point_size=2,
):
    """
    Plot for each topic:
      - Column 0: theta spatial map
      - Columns 1..top_n_per_topic: spatial expression of top genes

    Layout: n_topics rows × (1 + top_n_per_topic) columns

    Parameters
    ----------
    adata : AnnData
        Contains spatial coordinates in obs and HVG info in var['highly_variable'].
    df_beta : DataFrame
        Topic-image words/ Topic–gene matrix.
    top_words_list : list
        Concatenated list of top image words/ genes per topic (length = n_topics * top_n_per_topic).
    theta : array
        'theta_list' of shape (n_spots, n_topics).
    X_data : array (n_spots, n_hvgs)
        Raw gene expression matrix corresponding to HVGs.
    n_topics : int
        Number of topics to plot (rows).
    top_n_per_topic : int
        Number of top genes per topic.
    figsize : tuple
        Figure size.
    topic_cmap : str
        Colormap for theta.
    word_cmap : str
        Colormap for gene expression.
    point_size : int
        Size of scatter points.
    """

    # Spatial coordinates
    pos_x = adata.obs['pxl_row_in_fullres'].to_numpy()
    pos_y = adata.obs['pxl_col_in_fullres'].to_numpy()

    # Select top genes from df_beta
    top_gene_data = df_beta.loc[top_words_list]
    gene_list = top_gene_data.index.tolist()

    # Create subplot grid
    n_cols = 1 + top_n_per_topic
    fig, axes = plt.subplots(n_topics, n_cols, figsize=figsize)

    for i in range(n_topics):
        for j in range(n_cols):
            ax = axes[i, j]

            if j == 0:
                # Topic theta plot
                ax.scatter(
                    pos_x,
                    pos_y,
                    s=point_size,
                    c=theta[:, i],
                    cmap=topic_cmap
                )
                ax.set_title(f"Topic {i+1}", fontsize=20)

            else:
                # Gene expression plot
                gene = gene_list[i * top_n_per_topic + (j - 1)]
                idx = hvg_names.index(gene)

                ax.scatter(
                    pos_x,
                    pos_y,
                    s=point_size,
                    c=X_data[:, idx],
                    cmap=word_cmap
                )
                ax.set_title(gene, fontsize=20)

            ax.axis("off")
            ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    return fig, axes

#%%
from scipy.stats import pointbiserialr


def topic_target_pointbiserial_heatmap(
    theta_list,
    annotations_sub,
    labels_to_keep,
    topics_list=None,
    *,
    alpha=0.05,
    multiple_testing="bonferroni",  # "bonferroni" or None
    figsize=(10, 8),
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    linecolor="gray",
    cbar_label="Correlation",
    title="Topic vs Annotations",
    xlabel="Annotations",
    ylabel="Topics",
    xtick_rotation=-45,
    xtick_ha="left",
    mask_color="gray",
    mask_alpha=0.5,
    ax=None,
):
    """
    Compute point-biserial correlations between each topic (continuous)
    and each target label (binary indicator), then plot a heatmap.
    
    Returns
    -------
    correlation_matrix_r : (K, L) ndarray
    correlation_matrix_p : (K, L) ndarray
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    theta_list = np.asarray(theta_list)
    K = theta_list.shape[1]
    L = len(labels_to_keep)

    if topics_list is None:
        topics_list = [f"Topic {i+1}" for i in range(K)]
    if len(topics_list) != K:
        raise ValueError(f"topics_list must have length {K}, got {len(topics_list)}")

    correlation_matrix_r = np.zeros((K, L), dtype=float)
    correlation_matrix_p = np.ones((K, L), dtype=float)

    annotations_sub = np.asarray(annotations_sub)

    # Compute correlations
    for k in range(K):
        topic_data = theta_list[:, k]
        for l, lab in enumerate(labels_to_keep):
            target_data = (annotations_sub == lab).astype(int)
            r, p = pointbiserialr(topic_data, target_data)
            correlation_matrix_r[k, l] = r
            correlation_matrix_p[k, l] = p

    # Multiple testing thresholding
    if multiple_testing == "bonferroni":
        p_thresh = alpha / (K * L)
    elif multiple_testing is None:
        p_thresh = alpha
    else:
        raise ValueError("multiple_testing must be 'bonferroni' or None")

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.heatmap(
        correlation_matrix_r,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        xticklabels=labels_to_keep,
        yticklabels=topics_list,
        cbar_kws={"label": cbar_label},
        linewidths=linewidths,
        linecolor=linecolor,
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=14)
    # Overlay mask where p > threshold
    for i in range(K):
        for j in range(L):
            if correlation_matrix_p[i, j] > p_thresh:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1,
                        fill=True,
                        color=mask_color,
                        alpha=mask_alpha,
                        linewidth=0,
                    )
                )

    ax.set_title(title, fontsize=18)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
    return correlation_matrix_r, correlation_matrix_p, fig, ax


#%%
def plot_image_words_on_WSI_scale(wsi_token_file_path, img_word_list, 
                     H_patches, W_patches, S=16,
                     cmap="viridis", title=None):
    """
    Load VQ data, aggregate selected image words, and plot reconstructed image.

    Parameters
    ----------
    vq_file_path : str or Path
        Path to VQ pickle file
    img_word_list : list
        List of image words to aggregate (e.g., ["meta_1", "meta_5", ...])
    H_patches, W_patches : int
        Patch grid dimensions
    S : int
        Patch size
    cmap : str
        Matplotlib colormap
    title : str or None
        Optional plot title

    Returns
    -------
    big : np.ndarray
        Reconstructed image array
    """

    # load data
    X_count_token = load_npz(wsi_token_file_path).toarray()

    # build full word list
    W = X_count_token.shape[1]
    full_img_words = [f"meta_{i}" for i in range(W)]

    # map words → indices
    word_to_idx = {w: i for i, w in enumerate(full_img_words)}
    idx = [word_to_idx[w] for w in img_word_list if w in word_to_idx]

    if len(idx) == 0:
        raise ValueError("None of the img_word_list entries were found in VQ vocabulary.")

    # aggregate expression
    expr_img = X_count_token[:, idx].sum(axis=1)

    # reshape into image
    wsi_signal = rearrange(
        expr_img,
        "(r c h w) -> (r h) (c w)",
        r=H_patches,
        c=W_patches,
        h=S,
        w=S
    )

    # plot
    plt.figure(figsize=(6, 6))
    plt.imshow(wsi_signal, cmap=cmap)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return wsi_signal
