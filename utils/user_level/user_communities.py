"""
User community detection utilities for YouTube audience analysis.

Implements two approaches:
1. Feature-based clustering (K-means on user activity features)
2. Graph-based clustering (k-NN graph + Louvain community detection)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import networkx as nx
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
from tqdm.auto import tqdm


# ============================================================================
# APPROACH 1: Feature-based Clustering
# ============================================================================

def load_and_merge_data(comments_path, channels_path, timeseries_path, verbose=True):
    """
    Load and merge comment, channel, and timeseries data.
    
    Parameters
    ----------
    comments_path : str
        Path to comments TSV file
    channels_path : str
        Path to channels TSV file
    timeseries_path : str
        Path to timeseries TSV file
    verbose : bool
        Print loading progress
        
    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """
    if verbose:
        print("Loading data...")
    
    comments = pd.read_csv(comments_path, sep="\t", compression="gzip")
    channels = pd.read_csv(channels_path, sep="\t", compression="gzip")
    timeseries = pd.read_csv(timeseries_path, sep="\t", compression="gzip")
    
    if verbose:
        print(f"  Comments: {len(comments):,} rows")
        print(f"  Channels: {len(channels):,} rows")
        print(f"  Timeseries: {len(timeseries):,} rows")
    
    # Aggregate timeseries data
    ts_agg = timeseries.groupby('channel').agg({
        'delta_views': 'mean',
        'delta_subs': 'mean',
        'delta_videos': 'mean',
        'activity': 'mean'
    }).reset_index()
    
    # Merge channels with timeseries
    channels = channels.merge(ts_agg, on='channel', how='left')
    
    # Merge comments with channels
    merged = comments.merge(channels, on="name_cc", how="left")
    
    # Add derived features
    merged['log_subs'] = np.log1p(merged['subscribers_cc'])
    merged['join_year'] = pd.to_datetime(merged['join_date'], errors='coerce').dt.year
    
    if verbose:
        print(f"  Merged: {len(merged):,} rows")
    
    return merged


def compute_user_features(merged_df, top_n_categories=10, verbose=True):
    """
    Compute user-level features from comment data.
    
    Features include:
    - Activity: total comments, number of channels, avg comments per channel
    - Diversity: category entropy, number of unique categories
    - Category profile: proportion of comments in top categories
    - Channel characteristics: mean/std of log(subscribers), join year
    - Upload activity: mean delta_videos, activity
    - Channel growth: mean delta_subs, delta_views
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged comment/channel data
    top_n_categories : int
        Number of top categories to include as features
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        User feature vectors (one row per user)
    """
    if verbose:
        print(f"\nComputing user features for {merged_df['author'].nunique():,} users...")
    
    top_categories = merged_df['category_cc'].value_counts().nlargest(top_n_categories).index.tolist()
    
    def user_features(g):
        total_comments = g['num_comments'].sum()
        n_channels = g['name_cc'].nunique()
        weights = g['num_comments'] / total_comments
        
        f = {}
        
        # Activity
        f['total_comments'] = total_comments
        f['n_channels'] = n_channels
        f['avg_comments_per_channel'] = total_comments / n_channels if n_channels else 0
        
        # Diversity
        cat_counts = g.groupby('category_cc')['num_comments'].sum()
        p = cat_counts / cat_counts.sum()
        f['category_entropy'] = -(p * np.log(p)).sum()
        f['num_unique_categories'] = len(p)
        
        # Category profile (top categories)
        for c in top_categories:
            f[f'cat_{c}'] = p.get(c, 0)
        
        # Channel size
        f['mean_log_subs'] = np.average(g['log_subs'], weights=weights)
        f['std_log_subs'] = np.sqrt(np.cov(g['log_subs'], aweights=weights)) if len(g) > 1 else 0
        
        # Channel age
        if g['join_year'].notna().any():
            yrs = g['join_year'].dropna()
            w = weights.loc[yrs.index]
            f['mean_join_year'] = np.average(yrs, weights=w)
            f['std_join_year'] = np.sqrt(np.cov(yrs, aweights=w)) if len(yrs) > 1 else 0
        else:
            f['mean_join_year'] = f['std_join_year'] = np.nan
        
        # Upload activity
        for col in ['delta_videos', 'activity']:
            if col in g:
                f[f'mean_{col}'] = np.average(g[col].fillna(0), weights=weights)
            else:
                f[f'mean_{col}'] = np.nan
        
        # Channel growth
        for col in ['delta_subs', 'delta_views']:
            if col in g:
                f[f'mean_{col}'] = np.average(g[col].fillna(0), weights=weights)
            else:
                f[f'mean_{col}'] = np.nan
        
        return pd.Series(f)
    
    user_features_df = merged_df.groupby('author', group_keys=False).apply(user_features).fillna(0).reset_index()
    
    if verbose:
        print(f"  Generated {len(user_features_df.columns)-1} features per user")
    
    return user_features_df


def cluster_users_kmeans(user_features_df, n_clusters=8, n_components_pca=10, 
                         random_state=0, verbose=True):
    """
    Cluster users using K-means on PCA-reduced features.
    
    Parameters
    ----------
    user_features_df : pd.DataFrame
        User features (from compute_user_features)
    n_clusters : int
        Number of clusters for K-means
    n_components_pca : int
        Number of PCA components to use for clustering
    random_state : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (clustered_df, pca_2d_coords, cluster_summary)
    """
    if verbose:
        print(f"\nClustering {len(user_features_df):,} users into {n_clusters} clusters...")
    
    user_ids = user_features_df["author"]
    X = user_features_df.drop(columns=["author"])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for visualization (2D)
    pca_2d = PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X_scaled)
    
    # PCA for clustering (higher dimensional)
    pca_nd = PCA(n_components=n_components_pca, random_state=random_state)
    X_nd = pca_nd.fit_transform(X_scaled)
    
    if verbose:
        print(f"  Explained variance ({n_components_pca} components): {pca_nd.explained_variance_ratio_.sum():.2%}")
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_nd)
    
    # Create result dataframe
    result_df = user_features_df.copy()
    result_df["cluster"] = labels
    
    # Compute cluster summary
    cluster_summary = result_df.groupby("cluster")[X.columns].mean().round(2).sort_index()
    
    if verbose:
        print(f"  ✓ Clustering complete")
        print(f"  Cluster sizes: {result_df['cluster'].value_counts().sort_index().to_dict()}")
    
    return result_df, X_2d, cluster_summary


def evaluate_clustering(user_features_df, k_range=range(4, 12), 
                        n_components_pca=10, random_state=0, verbose=True):
    """
    Evaluate clustering quality using silhouette scores for different k values.
    
    Parameters
    ----------
    user_features_df : pd.DataFrame
        User features
    k_range : iterable
        Range of k values to test
    n_components_pca : int
        PCA components to use
    random_state : int
        Random seed
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Mapping of k to silhouette score
    """
    if verbose:
        print("\nEvaluating clustering quality...")
    
    user_ids = user_features_df["author"]
    X = user_features_df.drop(columns=["author"])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        scores[k] = score
        if verbose:
            print(f"  k={k}: silhouette={score:.3f}")
    
    return scores


def plot_user_clusters(X_2d, labels, output_dir='data', 
                       filename='user_clusters_kmeans.png', 
                       figsize=(10, 8), dpi=300, verbose=True):
    """
    Visualize user clusters in 2D PCA space.
    
    Parameters
    ----------
    X_2d : np.ndarray
        2D PCA coordinates
    labels : array-like
        Cluster labels
    output_dir : str
        Output directory
    filename : str
        Output filename
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    verbose : bool
        Print progress
        
    Returns
    -------
    str
        Path to saved file
    """
    import os
    
    if verbose:
        print("\nGenerating cluster visualization...")
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", 
                         s=10, alpha=0.7, edgecolors='none')
    plt.xlabel("PC1 (First Principal Component)", fontsize=12)
    plt.ylabel("PC2 (Second Principal Component)", fontsize=12)
    plt.title("User Clusters (K-means on PCA-reduced features)", fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"  ✓ Saved: {filepath}")
    
    return filepath


# ============================================================================
# APPROACH 2: Graph-based Clustering
# ============================================================================

def build_user_similarity_graph(df, k_neighbors=5, sim_threshold=0.8, 
                                min_component_size=100, verbose=True):
    """
    Build user similarity graph using k-NN on cosine similarity of channel vectors.
    
    Parameters
    ----------
    df : pd.DataFrame
        User-channel comment data with columns: author, name_cc, num_comments
    k_neighbors : int
        Number of nearest neighbors
    sim_threshold : float
        Minimum similarity to create an edge
    min_component_size : int
        Remove components smaller than this
    verbose : bool
        Print progress
        
    Returns
    -------
    nx.Graph
        User similarity graph
    """
    if verbose:
        print(f"\nBuilding user similarity graph from {len(df):,} rows...")
    
    # Map authors and channels to indices
    users = pd.Index(df["author"].unique(), name="author")
    channels = pd.Index(df["name_cc"].unique(), name="name_cc")
    u_idx = pd.Series(np.arange(len(users)), index=users)
    c_idx = pd.Series(np.arange(len(channels)), index=channels)
    
    if verbose:
        print(f"  Users: {len(users):,}")
        print(f"  Channels: {len(channels):,}")
    
    # Build user-channel sparse matrix
    rows = df["author"].map(u_idx).to_numpy()
    cols = df["name_cc"].map(c_idx).to_numpy()
    vals = np.ones_like(rows, dtype=np.float32)
    X = coo_matrix((vals, (rows, cols)), shape=(len(users), len(channels))).tocsr()
    
    if verbose:
        print(f"  Sparsity: {100 * (1 - X.nnz / (X.shape[0] * X.shape[1])):.2f}%")
    
    # k-NN with cosine similarity
    if verbose:
        print(f"  Computing {k_neighbors}-NN with cosine similarity...")
    
    nn = NearestNeighbors(n_neighbors=min(k_neighbors+1, X.shape[0]), 
                         metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    
    # Convert distances to similarities
    sims = 1.0 - distances
    
    # Build graph
    if verbose:
        print(f"  Building graph (similarity threshold: {sim_threshold})...")
    
    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    
    for i in tqdm(range(indices.shape[0]), desc="  Adding edges", disable=not verbose):
        for j in range(1, indices.shape[1]):  # skip self at j=0
            v = int(indices[i, j])
            w = float(sims[i, j])
            if w >= sim_threshold:
                if G.has_edge(i, v):
                    if w > G[i][v]["weight"]:
                        G[i][v]["weight"] = w
                else:
                    G.add_edge(i, v, weight=w)
    
    if verbose:
        print(f"  ✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    if verbose:
        print(f"  Removed {len(isolated):,} isolated nodes")
    
    # Remove small components
    small_comps = [comp for comp in nx.connected_components(G) if len(comp) < min_component_size]
    G.remove_nodes_from(set().union(*small_comps) if small_comps else set())
    G.remove_edges_from(nx.selfloop_edges(G))
    
    if verbose:
        print(f"  Removed {len(small_comps):,} small components (< {min_component_size} nodes)")
        print(f"  Final graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    return G


def detect_communities_louvain(G, verbose=True):
    """
    Detect communities using Louvain algorithm.
    
    Parameters
    ----------
    G : nx.Graph
        User similarity graph
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Node to community mapping
    """
    if community_louvain is None:
        raise ImportError("python-louvain not installed. Install with: pip install python-louvain")
    
    if verbose:
        print("\nDetecting communities with Louvain algorithm...")
    
    partition = community_louvain.best_partition(G, weight="weight")
    nx.set_node_attributes(G, partition, "community")
    
    n_communities = len(set(partition.values()))
    
    if verbose:
        print(f"  ✓ Found {n_communities} communities")
        comm_sizes = pd.Series(partition.values()).value_counts()
        print(f"  Largest community: {comm_sizes.max()} users")
        print(f"  Smallest community: {comm_sizes.min()} users")
    
    return partition


def plot_community_graph(G, partition, output_dir='data',
                        filename='user_communities_graph.png',
                        figsize=(70, 60), seed=42, verbose=True):
    """
    Visualize user community graph.
    
    Parameters
    ----------
    G : nx.Graph
        User similarity graph
    partition : dict
        Node to community mapping
    output_dir : str
        Output directory
    filename : str
        Output filename
    figsize : tuple
        Figure size
    seed : int
        Layout seed
    verbose : bool
        Print progress
        
    Returns
    -------
    str
        Path to saved file
    """
    import os
    
    if verbose:
        print("\nGenerating community visualization...")
        print(f"  Computing spring layout (this may take a while)...")
    
    pos = nx.spring_layout(G, seed=seed, k=0.2)
    colors = [partition[n] for n in G.nodes()]
    
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=40, 
                          cmap=plt.cm.turbo, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title("User Communities (k-NN graph + Louvain clustering)", 
             fontsize=48, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')  # Lower DPI for large plots
    plt.show()
    
    if verbose:
        print(f"  ✓ Saved: {filepath}")
    
    return filepath

