"""
Network visualization utilities for YouTube audience network analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import Counter


def plot_degree_distribution(G, output_dir='data', filename='network_degree_distribution.png', 
                             dpi=300, verbose=True):
    """
    Plot degree distribution, log-log plot, and component sizes.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    output_dir : str, default='data'
        Output directory for plot
    filename : str
        Output filename
    dpi : int, default=300
        Resolution for saved figure
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    str
        Path to saved file
    """
    if verbose:
        print("\nGenerating degree distribution plots...")
    
    # Get degree and component data
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    # Get connected components
    ccs = list(nx.connected_components(G))
    component_sizes = sorted([len(cc) for cc in ccs], reverse=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Degree histogram
    ax = axes[0]
    non_zero_degrees = [d for d in degree_values if d > 0]
    ax.hist(non_zero_degrees, bins=50, edgecolor='black', color='steelblue')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title('Degree Distribution (non-zero)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    # Plot 2: Log-log degree distribution (power law check)
    ax = axes[1]
    degree_counts = Counter(degree_values)
    degrees_sorted = sorted(degree_counts.items())
    x_deg = [d for d, count in degrees_sorted if d > 0]
    y_count = [count for d, count in degrees_sorted if d > 0]
    ax.loglog(x_deg, y_count, 'o-', markersize=6, color='darkblue')
    ax.set_xlabel('Degree (log scale)', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Log-Log Degree Distribution', fontsize=14)
    ax.grid(alpha=0.3)
    
    # Plot 3: Component size distribution (improved)
    ax = axes[2]
    
    # Show top 10 largest components as a bar chart
    top_components = component_sizes[:min(10, len(component_sizes))]
    
    if len(top_components) > 0:
        x_pos = np.arange(len(top_components))
        bars = ax.bar(x_pos, top_components, edgecolor='black', color='coral', alpha=0.8)
        
        # Color the largest component differently
        if len(bars) > 0:
            bars[0].set_color('darkred')
            bars[0].set_label('Largest Component')
        
        ax.set_xlabel('Component Rank', fontsize=12)
        ax.set_ylabel('Component Size (# Videos)', fontsize=12)
        ax.set_title(f'Top {len(top_components)} Connected Components', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'#{i+1}' for i in x_pos], fontsize=10)
        ax.set_yscale('log')
        ax.grid(alpha=0.3, axis='y')
        
        # Add text annotations for top 3
        for i in range(min(3, len(top_components))):
            height = top_components[i]
            ax.text(i, height * 1.1, f'{height:,}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if len(bars) > 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")
    
    return filepath


def plot_ego_network(G, hub_video, output_dir='data', filename=None, 
                     figsize=(20, 20), dpi=200, verbose=True):
    """
    Plot ego network for a single hub video.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    hub_video : str
        Video ID of the hub
    output_dir : str, default='data'
        Output directory
    filename : str, optional
        Output filename (auto-generated if None)
    figsize : tuple, default=(20, 20)
        Figure size in inches
    dpi : int, default=200
        Resolution
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    str
        Path to saved file
    """
    degree = G.degree(hub_video)
    
    if verbose:
        print(f"\nGenerating ego network for hub: {hub_video} (degree: {degree})")
    
    # Get ego network (1-hop neighborhood)
    ego = nx.ego_graph(G, hub_video, radius=1)
    n_neighbors = ego.number_of_nodes() - 1
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Layout with adaptive spacing
    k_value = 3 / np.sqrt(ego.number_of_nodes())
    pos = nx.spring_layout(ego, k=k_value, iterations=100, seed=42)
    
    # Colors and sizes
    node_colors = ['#FF4444' if n == hub_video else '#88CCFF' for n in ego.nodes()]
    node_sizes = [1000 if n == hub_video else 150 for n in ego.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(ego, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, 
                          edgecolors='black', linewidths=1.5, ax=ax)
    
    # Draw edges with varying thickness
    edges = ego.edges()
    weights = [ego[u][v].get('w_jaccard', 0.1) * 3 for u, v in edges]
    nx.draw_networkx_edges(ego, pos, alpha=0.25, width=weights, ax=ax)
    
    # Label only the hub video
    labels = {hub_video: str(hub_video)[:11]}
    nx.draw_networkx_labels(ego, pos, labels, font_size=14, 
                           font_weight='bold', font_color='white', ax=ax)
    
    ax.set_title(f'Hub Video: {str(hub_video)}\nDegree: {degree:,} | Neighbors: {n_neighbors:,}', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add statistics box
    textstr = f'Network: {ego.number_of_nodes():,} nodes, {ego.number_of_edges():,} edges\n'
    textstr += f'Hub shown in red, neighbors in blue'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f'network_ego_{hub_video}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")
    
    return filepath


def plot_top_hubs_ego_networks(G, top_n=3, output_dir='data', 
                                figsize_individual=(20, 20), 
                                figsize_overview=(30, 10),
                                dpi=200, verbose=True):
    """
    Plot ego networks for top N hub videos (by degree).
    
    Creates individual plots for each hub plus a combined overview.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    top_n : int, default=3
        Number of top hubs to visualize
    output_dir : str, default='data'
        Output directory
    figsize_individual : tuple, default=(20, 20)
        Size for individual plots
    figsize_overview : tuple, default=(30, 10)
        Size for combined overview
    dpi : int, default=200
        Resolution
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    list
        List of saved file paths
    """
    if verbose:
        print(f"\nVisualizing ego networks of top {top_n} hub videos...")
    
    # Find top hubs
    degrees = dict(G.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    saved_files = []
    
    # Create individual plots
    for idx, (video, degree) in enumerate(top_hubs):
        if degree == 0:
            continue
        
        filename = f'network_hub_{idx+1}_{video}.png'
        filepath = plot_ego_network(G, video, output_dir=output_dir, 
                                    filename=filename, figsize=figsize_individual,
                                    dpi=dpi, verbose=verbose)
        saved_files.append(filepath)
    
    # Create combined overview
    if verbose:
        print(f"\nCreating combined overview...")
    
    fig, axes = plt.subplots(1, top_n, figsize=figsize_overview)
    if top_n == 1:
        axes = [axes]
    
    for idx, (video, degree) in enumerate(top_hubs[:top_n]):
        if degree == 0:
            continue
        
        ego = nx.ego_graph(G, video, radius=1)
        ax = axes[idx]
        
        # Layout
        k_value = 2.5 / np.sqrt(ego.number_of_nodes())
        pos = nx.spring_layout(ego, k=k_value, iterations=80, seed=42)
        
        # Colors and sizes
        node_colors = ['#FF4444' if n == video else '#88CCFF' for n in ego.nodes()]
        node_sizes = [600 if n == video else 80 for n in ego.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(ego, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, 
                              edgecolors='black', linewidths=1, ax=ax)
        nx.draw_networkx_edges(ego, pos, alpha=0.2, width=1, ax=ax)
        
        # Label hub
        labels = {video: str(video)[:11]}
        nx.draw_networkx_labels(ego, pos, labels, font_size=9, 
                               font_weight='bold', font_color='white', ax=ax)
        
        ax.set_title(f'Hub: {str(video)[:20]}\nDegree: {degree:,}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save overview
    os.makedirs(output_dir, exist_ok=True)
    overview_filename = 'network_hub_videos_overview.png'
    overview_filepath = os.path.join(output_dir, overview_filename)
    plt.savefig(overview_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {overview_filepath}")
    
    saved_files.append(overview_filepath)
    
    # Print summary
    if verbose:
        print(f"\nTop {top_n} Hub Videos:")
        for video, degree in top_hubs:
            print(f"  {str(video):<40} → Degree: {degree}")
    
    return saved_files


def plot_main_component(G, output_dir='data', filename='network_main_component.png',
                       max_nodes_for_viz=1000, dpi=300, verbose=True):
    """
    Visualize the largest connected component (if not too large).
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    output_dir : str, default='data'
        Output directory
    filename : str
        Output filename
    max_nodes_for_viz : int, default=1000
        Max nodes to attempt visualization
    dpi : int, default=300
        Resolution
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    str or None
        Path to saved file, or None if component too large
    """
    if verbose:
        print("\nVisualizing largest connected component...")
    
    # Get largest component
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    
    if verbose:
        print(f"Largest component: {G_main.number_of_nodes():,} nodes, {G_main.number_of_edges():,} edges")
    
    # Only visualize if reasonably sized
    if G_main.number_of_nodes() > max_nodes_for_viz:
        if verbose:
            print(f"  Component is too large ({G_main.number_of_nodes():,} nodes) for matplotlib visualization")
            print(f"  → Use Gephi instead! Open the GEXF file for interactive exploration")
        return None
    
    if verbose:
        print("  Generating layout (this may take a moment)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # Color nodes by degree
    node_colors = [G_main.degree(n) for n in G_main.nodes()]
    
    # Layout 1: Spring layout
    ax = axes[0]
    pos = nx.spring_layout(G_main, k=0.5, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G_main, pos, node_color=node_colors, 
                          node_size=30, cmap='plasma', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_main, pos, alpha=0.3, width=0.8, ax=ax)
    ax.set_title(f'Spring Layout\nColor by Degree', fontsize=14)
    ax.axis('off')
    
    # Layout 2: Kamada-Kawai (if not too large)
    if G_main.number_of_nodes() <= 500:
        ax = axes[1]
        pos2 = nx.kamada_kawai_layout(G_main)
        nx.draw_networkx_nodes(G_main, pos2, node_color=node_colors, 
                              node_size=30, cmap='plasma', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G_main, pos2, alpha=0.3, width=0.8, ax=ax)
        ax.set_title(f'Kamada-Kawai Layout\nColor by Degree', fontsize=14)
        ax.axis('off')
    else:
        ax = axes[1]
        ax.text(0.5, 0.5, f'Component too large ({G_main.number_of_nodes()} nodes)\nfor Kamada-Kawai layout\n\nUse Gephi for better visualization!', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")
    
    return filepath


def plot_community_sizes(partition, output_dir='data', filename='network_community_sizes.png',
                         top_n=20, dpi=300, verbose=True):
    """
    Plot community size distribution.
    
    Parameters
    ----------
    partition : dict
        Community assignments {node: community_id}
    output_dir : str, default='data'
        Output directory
    filename : str
        Output filename
    top_n : int, default=20
        Number of top communities to show
    dpi : int, default=300
        Resolution
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    str
        Path to saved file
    """
    if verbose:
        print("\nPlotting community size distribution...")
    
    # Count community sizes
    comm_sizes = Counter(partition.values())
    n_communities = len(comm_sizes)
    
    # Sort by size
    sorted_communities = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Top N communities
    ax = axes[0]
    top_communities = sorted_communities[:top_n]
    comm_ids = [f"C{c}" for c, size in top_communities]
    sizes = [size for c, size in top_communities]
    
    ax.barh(comm_ids[::-1], sizes[::-1], color='steelblue', edgecolor='black')
    ax.set_xlabel('Community Size (# Videos)', fontsize=12)
    ax.set_ylabel('Community ID', fontsize=12)
    ax.set_title(f'Top {top_n} Communities by Size', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Size distribution (histogram)
    ax = axes[1]
    all_sizes = list(comm_sizes.values())
    ax.hist(all_sizes, bins=min(50, n_communities), edgecolor='black', color='coral')
    ax.set_xlabel('Community Size', fontsize=12)
    ax.set_ylabel('Number of Communities', fontsize=12)
    ax.set_title(f'Community Size Distribution\n({n_communities} total communities)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")
        print(f"  Total communities: {n_communities}")
        print(f"  Largest: {max(all_sizes):,} videos")
        print(f"  Smallest: {min(all_sizes):,} videos")
        print(f"  Average: {np.mean(all_sizes):.1f} videos")
    
    return filepath


def plot_top_videos(G, video_metadata, top_n=20, metric='degree', 
                   output_dir='data', filename='top_hub_videos.png', 
                   dpi=300, figsize=(14, 10), verbose=True):
    """
    Create a visualization of top hub videos by degree or betweenness.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph
    video_metadata : dict
        Dictionary mapping video_id to metadata (from data_loader)
    top_n : int, default=20
        Number of top videos to display
    metric : str, default='degree'
        Metric to sort by ('degree' or dict of betweenness values)
    output_dir : str, default='data'
        Output directory for plot
    filename : str
        Output filename
    dpi : int, default=300
        Resolution for saved figure
    figsize : tuple, default=(14, 10)
        Figure size (width, height)
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    str
        Path to saved figure
    """
    if verbose:
        print(f"\nGenerating top {top_n} videos visualization...")
    
    # Get metric values
    if metric == 'degree':
        values = dict(G.degree())
        metric_name = 'Network Degree'
        title = f'Top {top_n} Hub Videos by Network Degree'
    else:
        values = metric  # Assume it's a betweenness dict
        metric_name = 'Betweenness Centrality'
        title = f'Top {top_n} Videos by Betweenness Centrality'
    
    # Get top videos
    top_videos = sorted(values.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Prepare data for plotting
    video_ids = [vid for vid, _ in top_videos]
    metric_values = [val for _, val in top_videos]
    
    # Get metadata
    titles = []
    channels = []
    views_list = []
    
    for vid in video_ids:
        meta = video_metadata.get(vid, {})
        title_text = meta.get('title', 'Unknown')[:40] + '...' if len(meta.get('title', '')) > 40 else meta.get('title', 'Unknown')
        titles.append(title_text)
        channels.append(meta.get('channel', 'Unknown'))
        views_list.append(meta.get('views', 0))
    
    # Create color map by channel
    unique_channels = list(set(channels))
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_channels)))
    channel_colors = {ch: colors_palette[i] for i, ch in enumerate(unique_channels)}
    bar_colors = [channel_colors[ch] for ch in channels]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(video_ids))
    bars = ax.barh(y_pos, metric_values, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    # Add view count as text on bars
    from data.data_loader import format_number
    for i, (bar, views) in enumerate(zip(bars, views_list)):
        width = bar.get_width()
        # Add views text inside bar (right side)
        ax.text(width * 0.98, bar.get_y() + bar.get_height()/2, 
                f'{format_number(views)} views',
                ha='right', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, linewidth=0))
    
    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {title}" for i, title in enumerate(titles)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend for all channels (sorted by frequency)
    channel_counts = Counter(channels)
    sorted_channels = [ch for ch, _ in channel_counts.most_common()]
    legend_patches = [mpatches.Patch(color=channel_colors[ch], label=ch) 
                     for ch in sorted_channels if ch in channel_colors]
    
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8, 
             title='Channels', framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")
        print(f"  Top channel: {channel_counts.most_common(1)[0][0]} ({channel_counts.most_common(1)[0][1]} videos)")
    
    return filepath

