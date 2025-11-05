"""
Network construction utilities for YouTube video-video audience network.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import combinations
from tqdm.notebook import tqdm


def prepare_network_data(df, verbose=True):
    """
    Prepare filtered comment data for network construction.
    
    Removes duplicates and unused category levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered comment data with columns: author, video_id
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Prepared data with unique author-video pairs
    """
    if verbose:
        print("Step 1: Preparing data...")
        print(f"  Input shape: {df.shape}")
        print(f"  Unique videos: {df['video_id'].nunique():,}")
        print(f"  Unique users: {df['author'].nunique():,}")
    
    # Use distinct author-video pairs
    df_prepared = df[['author', 'video_id']].drop_duplicates().copy()
    
    # CRITICAL: Remove unused category levels
    # (filtered data may still have all original category levels)
    if df_prepared['video_id'].dtype.name == 'category':
        df_prepared['video_id'] = df_prepared['video_id'].cat.remove_unused_categories()
    if df_prepared['author'].dtype.name == 'category':
        df_prepared['author'] = df_prepared['author'].cat.remove_unused_categories()
    
    if verbose:
        print(f"  → {len(df_prepared):,} unique author-video pairs")
        print(f"  → {df_prepared['video_id'].nunique():,} unique videos")
        print(f"  → {df_prepared['author'].nunique():,} unique users")
    
    return df_prepared


def build_video_mappings(df, verbose=True):
    """
    Build video→users and user→videos mappings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared data from prepare_network_data()
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    dict
        video_to_users: {video_id: set of users}
    dict
        video_commenter_counts: {video_id: count}
    dict
        user_to_videos: {user_id: list of videos}
    list
        video_ids: list of all video IDs
    """
    if verbose:
        print("\nStep 2: Building video↔users mappings...")
    
    # Video → users mapping
    video_to_users = df.groupby('video_id', observed=True)['author'].apply(set).to_dict()
    video_ids = list(video_to_users.keys())
    
    # Commenter counts per video
    video_commenter_counts = {vid: len(users) for vid, users in video_to_users.items()}
    
    # User → videos mapping
    user_to_videos = df.groupby('author', observed=True)['video_id'].apply(list).to_dict()
    
    if verbose:
        print(f"  → {len(video_ids):,} videos")
        print(f"  → {len(user_to_videos):,} users")
    
    return video_to_users, video_commenter_counts, user_to_videos, video_ids


def compute_video_edges(user_to_videos, video_commenter_counts, verbose=True):
    """
    Compute video-video edges from shared commenters.
    
    For each user, creates edges between all pairs of videos they commented on.
    
    Parameters
    ----------
    user_to_videos : dict
        Mapping from user_id to list of video_ids
    video_commenter_counts : dict
        Mapping from video_id to commenter count
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Edges with columns: video_i, video_j, overlap, w_jaccard, w_cosine
    """
    if verbose:
        print("\nStep 3: Computing video-video edges...")
        print(f"  Processing {len(user_to_videos):,} users...")
    
    # Build edges by iterating through users
    edge_overlaps = defaultdict(int)
    
    for user, vids in tqdm(user_to_videos.items(), desc="Building edges", disable=not verbose):
        if len(vids) >= 2:  # User must comment on 2+ videos to create edges
            for v1, v2 in combinations(sorted(vids), 2):  # sorted ensures consistent ordering
                edge_overlaps[(v1, v2)] += 1
    
    if verbose:
        print(f"  → Found {len(edge_overlaps):,} video pairs with shared commenters")
    
    # Compute edge weights
    if verbose:
        print("\nStep 4: Computing edge weights...")
    
    edge_data = []
    for (v1, v2), overlap_count in tqdm(edge_overlaps.items(), desc="Computing weights", disable=not verbose):
        n1 = video_commenter_counts[v1]
        n2 = video_commenter_counts[v2]
        
        # Jaccard similarity: overlap / union
        jaccard = overlap_count / (n1 + n2 - overlap_count)
        
        # Cosine similarity: overlap / sqrt(size1 * size2)
        cosine = overlap_count / np.sqrt(n1 * n2)
        
        edge_data.append({
            'video_i': v1,
            'video_j': v2,
            'overlap': overlap_count,
            'w_jaccard': jaccard,
            'w_cosine': cosine
        })
    
    edges = pd.DataFrame(edge_data)
    
    if verbose:
        print(f"  → Created {len(edges):,} edges")
    
    return edges


def prune_edges(edges, min_overlap=3, min_jaccard=0.02, verbose=True):
    """
    Prune weak edges to keep only meaningful connections.
    
    Parameters
    ----------
    edges : pd.DataFrame
        Edge DataFrame from compute_video_edges()
    min_overlap : int, default=3
        Minimum number of shared commenters
    min_jaccard : float, default=0.02
        Minimum Jaccard similarity
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Pruned edges
    """
    if verbose:
        print("\nStep 5: Pruning edges...")
        print(f"  Before pruning: {len(edges):,} edges")
    
    edges_pruned = edges[
        (edges['overlap'] >= min_overlap) & 
        (edges['w_jaccard'] >= min_jaccard)
    ].reset_index(drop=True)
    
    if verbose:
        print(f"  After pruning:  {len(edges_pruned):,} edges")
        print(f"  Reduction:      {100 * (1 - len(edges_pruned) / len(edges)):.1f}%")
    
    return edges_pruned


def build_networkx_graph(edges, video_ids, verbose=True):
    """
    Build NetworkX graph from edge list.
    
    Parameters
    ----------
    edges : pd.DataFrame
        Pruned edges with columns: video_i, video_j, overlap, w_jaccard, w_cosine
    video_ids : list
        List of all video IDs (nodes)
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    nx.Graph
        NetworkX graph with edge attributes: weight, overlap, w_cosine
    """
    if verbose:
        print("\nStep 6: Building NetworkX graph...")
    
    G = nx.Graph()
    
    # Add all videos as nodes
    G.add_nodes_from(video_ids)
    
    # Add edges with attributes
    if verbose:
        print(f"  Adding {len(edges):,} edges...")
    
    for r in tqdm(edges.itertuples(index=False), total=len(edges), 
                  desc="  Building graph", disable=not verbose):
        G.add_edge(
            r.video_i, r.video_j,
            weight=float(r.w_jaccard),
            overlap=int(r.overlap),
            w_cosine=float(r.w_cosine)
        )
    
    if verbose:
        print(f"  → Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        print(f"  → Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.1f}")
    
    return G


def build_audience_network(df, min_overlap=3, min_jaccard=0.02, verbose=True):
    """
    End-to-end pipeline to build video-video audience network.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered comment data with columns: author, video_id
    min_overlap : int, default=3
        Minimum shared commenters for an edge
    min_jaccard : float, default=0.02
        Minimum Jaccard similarity for an edge
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    nx.Graph
        Video-video network
    pd.DataFrame
        Pruned edges
    dict
        Metadata: video_commenter_counts, video_ids, etc.
    """
    # Step 1: Prepare data
    df_prepared = prepare_network_data(df, verbose=verbose)
    
    # Step 2: Build mappings
    video_to_users, video_commenter_counts, user_to_videos, video_ids = \
        build_video_mappings(df_prepared, verbose=verbose)
    
    # Step 3-4: Compute edges and weights
    edges = compute_video_edges(user_to_videos, video_commenter_counts, verbose=verbose)
    
    # Step 5: Prune edges
    edges_pruned = prune_edges(edges, min_overlap=min_overlap, 
                              min_jaccard=min_jaccard, verbose=verbose)
    
    # Step 6: Build NetworkX graph
    G = build_networkx_graph(edges_pruned, video_ids, verbose=verbose)
    
    # Collect metadata
    metadata = {
        'video_commenter_counts': video_commenter_counts,
        'video_ids': video_ids,
        'video_to_users': video_to_users,
        'n_videos': len(video_ids),
        'n_edges': len(edges_pruned)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ NETWORK CONSTRUCTION COMPLETE")
        print(f"{'='*60}")
        print(f"Videos (nodes):  {G.number_of_nodes():>15,}")
        print(f"Edges (pruned):  {G.number_of_edges():>15,}")
        print(f"Avg degree:      {2 * G.number_of_edges() / G.number_of_nodes():>15.1f}")
        print(f"{'='*60}")
    
    return G, edges_pruned, metadata

