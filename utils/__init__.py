"""
Utility functions for network analysis.
"""

from .filtering import iterative_filter

# UPDATED: Import only the new sparse builder function
from .network_builder import build_video_projection_sparse

from .network_analysis import (
    compute_betweenness_centrality,
    get_top_nodes,
    detect_communities,
    analyze_network_structure,
    export_network
)
from .visualization import (
    plot_degree_distribution,
    plot_ego_network,
    plot_top_hubs_ego_networks,
    plot_main_component,
    plot_community_sizes
)
from .data_exploration import (
    explore_initial_data
)

__all__ = [
    # Filtering
    'iterative_filter',
    
    # Network building (Updated)
    'build_video_projection_sparse',
    
    # Network analysis
    'compute_betweenness_centrality',
    'get_top_nodes',
    'detect_communities',
    'analyze_network_structure',
    'export_network',
    
    # Visualization
    'plot_degree_distribution',
    'plot_ego_network',
    'plot_top_hubs_ego_networks',
    'plot_main_component',
    'plot_community_sizes',
    
    # Data exploration
    'explore_initial_data'
]
