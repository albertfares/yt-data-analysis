"""
Network construction utilities for YouTube video-video audience network.
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

def build_video_projection_sparse(df, min_shared_users=5, jaccard_threshold=0.1):
    """
    Step 1.1: Constructs a Video-Video graph using Sparse Matrix Multiplication.
    This is 100x faster than iterating through rows.
    """
    print(f"--- Building Network from {len(df)} comments ---")
    
    # 1. Map string IDs to integer indices for matrix operations
    # We need to ensure we can map back later!
    video_ids = df['video_id'].unique()
    user_ids = df['author_id'].unique()
    
    vid_to_idx = {vid: i for i, vid in enumerate(video_ids)}
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    
    print(f"Nodes (Videos): {len(video_ids)}")
    print(f"Connectors (Users): {len(user_ids)}")
    
    # 2. Create the Sparse Matrix (Rows=Videos, Cols=Users)
    # This matrix represents the Bipartite Graph efficiently
    rows = df['video_id'].map(vid_to_idx).values
    cols = df['author_id'].map(user_to_idx).values
    data = np.ones(len(df), dtype=int)
    
    # Shape: (Num Videos, Num Users)
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(video_ids), len(user_ids)))
    
    # Convert to CSR (Compressed Sparse Row) for fast math
    sparse_matrix_csr = sparse_matrix.tocsr()
    
    # 3. The Magic Step: Matrix Multiplication
    # M * M_transpose = Intersection Matrix (Shared Users)
    # Entry (i, j) is the number of users shared by Video i and Video j
    print("Calculating intersections (Sparse Dot Product)...")
    intersection_matrix = sparse_matrix_csr.dot(sparse_matrix_csr.T)
    
    # 4. Calculate Jaccard Similarity
    # Jaccard(A, B) = Intersection(A, B) / (Degree(A) + Degree(B) - Intersection(A, B))
    
    # Get degrees (number of unique users per video) - this is the diagonal
    video_degrees = intersection_matrix.diagonal()
    
    # Convert intersection to Coordinate format to iterate only non-zero edges
    intersection_coo = intersection_matrix.tocoo()
    
    edges = []
    print("Filtering edges and calculating Jaccard...")
    
    # Zip lets us iterate through existing connections only
    # Note: intersection_coo includes (A, B) and (B, A). We only need A < B.
    for i, j, intersection in zip(intersection_coo.row, intersection_coo.col, intersection_coo.data):
        if i < j: # Upper triangle only
            if intersection >= min_shared_users: # Pre-filter by count
                
                degree_a = video_degrees[i]
                degree_b = video_degrees[j]
                
                union = degree_a + degree_b - intersection
                
                if union > 0:
                    jaccard = intersection / union
                    
                    if jaccard >= jaccard_threshold:
                        # Append tuple: (VideoID_A, VideoID_B, Attributes)
                        edges.append((video_ids[i], video_ids[j], {'weight': jaccard, 'shared_users': int(intersection)}))

    # 5. Create NetworkX Graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    print(f"Graph Built: {G.number_of_nodes()} videos, {G.number_of_edges()} edges")
    return G
