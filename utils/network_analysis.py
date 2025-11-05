"""
Network analysis utilities: centrality, communities, statistics.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


def compute_betweenness_centrality(G, k=100, seed=42, verbose=True):
    """
    Compute betweenness centrality on the largest connected component.
    
    Uses k-sample approximation for large graphs.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    k : int, default=100
        Number of nodes to sample for approximation
    seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    dict
        Betweenness centrality scores {node: score}
    nx.Graph
        Largest connected component subgraph
    """
    if verbose:
        print("\nStep 7: Computing betweenness centrality...")
    
    # Get largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    
    if verbose:
        print(f"  Largest component: {len(largest_cc):,} nodes")
        print(f"  Computing with k={k} sample (seed={seed})...")
    
    # Compute betweenness on main component
    betw = nx.betweenness_centrality(
        G_main, 
        weight='weight', 
        k=k, 
        normalized=True, 
        seed=seed
    )
    
    if verbose:
        print("  ✓ Done!")
    
    return betw, G_main


def get_top_nodes(centrality_dict, top_n=20, verbose=True):
    """
    Get top N nodes by centrality score.
    
    Parameters
    ----------
    centrality_dict : dict
        Centrality scores {node: score}
    top_n : int, default=20
        Number of top nodes to return
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    list
        List of (node, score) tuples, sorted descending
    """
    top_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if verbose:
        print(f"\n  Top {top_n} nodes:")
        for node, score in top_nodes:
            print(f"    {node}\t{score:.4f}")
    
    return top_nodes


def detect_communities(G, method='louvain', weight='weight', verbose=True):
    """
    Detect communities in the network.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    method : str, default='louvain'
        Community detection method ('louvain' only for now)
    weight : str, default='weight'
        Edge attribute to use as weight
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    dict or None
        Community assignments {node: community_id}, or None if failed
    dict or None
        Community statistics, or None if failed
    """
    if verbose:
        print("\nStep 8: Running community detection...")
    
    try:
        import community as community_louvain
        
        partition = community_louvain.best_partition(G, weight=weight)
        n_communities = len(set(partition.values()))
        
        # Compute community size distribution
        comm_sizes = Counter(partition.values())
        
        stats = {
            'n_communities': n_communities,
            'largest_community': max(comm_sizes.values()),
            'smallest_community': min(comm_sizes.values()),
            'avg_community_size': np.mean(list(comm_sizes.values())),
            'community_sizes': comm_sizes
        }
        
        if verbose:
            print(f"  ✓ Found {n_communities} communities!")
            print(f"  → Largest community:  {stats['largest_community']:,} videos")
            print(f"  → Smallest community: {stats['smallest_community']:,} videos")
            print(f"  → Avg community size: {stats['avg_community_size']:.1f} videos")
        
        return partition, stats
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Community detection failed: {e}")
        return None, None


def analyze_network_structure(G, verbose=True):
    """
    Compute basic network structure statistics.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    verbose : bool, default=True
        Print results
        
    Returns
    -------
    dict
        Network statistics
    """
    if verbose:
        print("\n" + "="*80)
        print("NETWORK STRUCTURE ANALYSIS")
        print("="*80)
    
    # Connected components
    ccs = list(nx.connected_components(G))
    cc_sizes = sorted([len(cc) for cc in ccs], reverse=True)
    
    # Degree statistics
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # Density
    density = nx.density(G)
    
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_components': len(ccs),
        'largest_component_size': cc_sizes[0] if cc_sizes else 0,
        'largest_component_pct': 100 * cc_sizes[0] / G.number_of_nodes() if cc_sizes else 0,
        'isolated_nodes': sum(1 for d in degree_values if d == 0),
        'avg_degree': np.mean(degree_values),
        'median_degree': np.median(degree_values),
        'max_degree': max(degree_values) if degree_values else 0,
        'density': density
    }
    
    if verbose:
        print(f"\n1. CONNECTIVITY:")
        print(f"   Total connected components: {stats['n_components']:,}")
        print(f"   Largest component: {stats['largest_component_size']:,} nodes ({stats['largest_component_pct']:.1f}%)")
        if len(cc_sizes) > 1:
            print(f"   2nd largest: {cc_sizes[1]:,} nodes")
        if len(cc_sizes) > 2:
            print(f"   3rd largest: {cc_sizes[2]:,} nodes")
        print(f"   Isolated nodes (size=1): {stats['isolated_nodes']:,}")
        
        print(f"\n2. DEGREE DISTRIBUTION:")
        print(f"   Average degree: {stats['avg_degree']:.2f}")
        print(f"   Median degree: {stats['median_degree']:.0f}")
        print(f"   Max degree: {stats['max_degree']:,} (hub video)")
        print(f"   Nodes with degree 0: {sum(1 for d in degree_values if d == 0):,}")
        print(f"   Nodes with degree 1: {sum(1 for d in degree_values if d == 1):,}")
        print(f"   Nodes with degree ≥5: {sum(1 for d in degree_values if d >= 5):,}")
        
        print(f"\n3. NETWORK DENSITY:")
        print(f"   Density: {density:.8f} ({100*density:.6f}%)")
        print(f"   Network is {'VERY SPARSE' if density < 0.001 else 'SPARSE' if density < 0.01 else 'MODERATE'}")
        
        print("="*80)
    
    return stats


def export_network(G, edges_df, video_commenter_counts, betweenness=None, 
                   partition=None, output_dir='models', verbose=True):
    """
    Export network to CSV and GEXF formats.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    edges_df : pd.DataFrame
        Edge DataFrame
    video_commenter_counts : dict
        Commenter counts per video
    betweenness : dict, optional
        Betweenness centrality scores
    partition : dict, optional
        Community assignments
    output_dir : str, default='.'
        Output directory
    verbose : bool, default=True
        Print progress messages
    """
    if verbose:
        print("\nStep 9: Exporting network files...")
    
    import os
    
    # Export edges
    edge_file = os.path.join(output_dir, 'video_network_edges.csv')
    edges_df[['video_i', 'video_j', 'overlap', 'w_jaccard', 'w_cosine']].to_csv(
        edge_file, index=False
    )
    if verbose:
        print(f"  ✓ Saved: {edge_file}")
    
    # Export nodes with attributes
    video_ids = list(G.nodes())
    nodes_df = pd.DataFrame({
        'video_id': video_ids,
        'commenter_count': [video_commenter_counts.get(v, 0) for v in video_ids],
        'degree': [G.degree(v) for v in video_ids],
        'betweenness': [betweenness.get(v, 0.0) if betweenness else 0.0 for v in video_ids],
        'community': [partition.get(v) if partition else None for v in video_ids]
    })
    
    node_file = os.path.join(output_dir, 'video_network_nodes.csv')
    nodes_df.to_csv(node_file, index=False)
    if verbose:
        print(f"  ✓ Saved: {node_file}")
    
    # Export GEXF for Gephi
    gexf_file = os.path.join(output_dir, 'video_network.gexf')
    nx.write_gexf(G, gexf_file)
    if verbose:
        print(f"  ✓ Saved: {gexf_file}")
    
    if verbose:
        print("\n" + "="*80)
        print("✓ NETWORK EXPORT COMPLETE")
        print("="*80)
        print(f"Files saved to: {output_dir}")
        print("="*80)

