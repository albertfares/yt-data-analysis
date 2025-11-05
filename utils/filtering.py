"""
Iterative filtering utilities for YouTube comment data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def iterative_filter(df, min_videos_per_user=5, min_users_per_video=10, 
                     max_iterations=10, verbose=True, plot=False, output_dir='data'):
    """
    Iteratively filter users and videos until convergence.
    
    Keeps only:
    - Users who commented on >= min_videos_per_user distinct videos
    - Videos that have >= min_users_per_video distinct commenters
    
    Iterates because filtering users changes video counts and vice versa.
    
    Parameters
    ----------
    df : pd.DataFrame
        Comment data with columns: author, video_id
    min_videos_per_user : int, default=5
        Minimum number of distinct videos a user must comment on
    min_users_per_video : int, default=10
        Minimum number of distinct users a video must have
    max_iterations : int, default=10
        Maximum iterations (usually converges in 2-3)
    verbose : bool, default=True
        Print iteration progress
    plot : bool, default=False
        Generate convergence plot showing filtering progression
    output_dir : str, default='data'
        Output directory for plot (if plot=True)
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    dict
        Statistics: original_size, filtered_size, n_users, n_videos, iterations, history
    """
    df_work = df.copy()
    original_size = len(df_work)
    
    # Track history for plotting
    history = {
        'iteration': [0],
        'comments': [len(df_work)],
        'users': [df_work['author'].nunique()],
        'videos': [df_work['video_id'].nunique()]
    }
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"Iteration {iteration + 1}")
        
        # Filter users by number of distinct videos they commented on
        user_counts = df_work.groupby('author', observed=True)['video_id'].nunique()
        valid_users = user_counts[user_counts >= min_videos_per_user].index
        
        # Filter videos by number of distinct valid commenters
        video_counts = df_work[df_work['author'].isin(valid_users)].groupby('video_id', observed=True)['author'].nunique()
        valid_videos = video_counts[video_counts >= min_users_per_video].index
        
        # Apply both filters
        new_df = df_work[
            df_work['author'].isin(valid_users) &
            df_work['video_id'].isin(valid_videos)
        ].reset_index(drop=True)
        
        # Check convergence
        diff = len(df_work) - len(new_df)
        if verbose:
            print(f"   Rows kept: {len(new_df):,}  (dropped {diff:,})")
        
        if diff == 0:
            if verbose:
                print("✓ Converged — filtering complete.")
            break
        
        df_work = new_df
        
        # Track metrics for this iteration
        history['iteration'].append(iteration + 1)
        history['comments'].append(len(df_work))
        history['users'].append(df_work['author'].nunique())
        history['videos'].append(df_work['video_id'].nunique())
    
    # Gather statistics
    stats = {
        'original_size': original_size,
        'filtered_size': len(df_work),
        'n_users': df_work['author'].nunique(),
        'n_videos': df_work['video_id'].nunique(),
        'iterations': iteration + 1,
        'min_videos_per_user': min_videos_per_user,
        'min_users_per_video': min_users_per_video,
        'history': history
    }
    
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Original size:    {stats['original_size']:>15,}")
    print(f"Filtered size:    {stats['filtered_size']:>15,}")
    print(f"Remaining users:  {stats['n_users']:>15,}")
    print(f"Remaining videos: {stats['n_videos']:>15,}")
    print(f"Iterations:       {stats['iterations']:>15}")
    print(f"{'='*60}")

    # Generate convergence plot if requested
    if plot:
        _plot_convergence(history, min_videos_per_user, min_users_per_video, 
                         output_dir, verbose)
    
    return df_work, stats


def _plot_convergence(history, min_videos_per_user, min_users_per_video, 
                      output_dir='data', verbose=True):
    """
    Plot the convergence of iterative filtering.
    
    Parameters
    ----------
    history : dict
        History dictionary with iteration, comments, users, videos
    min_videos_per_user : int
        Filtering threshold for users
    min_users_per_video : int
        Filtering threshold for videos
    output_dir : str
        Output directory
    verbose : bool
        Print progress messages
    """
    if verbose:
        print("\nGenerating convergence plot...")
    
    iterations = history['iteration']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Comments over iterations
    ax = axes[0]
    ax.plot(iterations, history['comments'], 'o-', linewidth=2, 
            markersize=8, color='steelblue', label='Comments')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Comments', fontsize=12)
    ax.set_title('Comments Retained per Iteration', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(iterations)
    
    # Add percentage labels
    for i, (iter_num, count) in enumerate(zip(iterations, history['comments'])):
        if i == 0:
            pct_text = '100%'
        else:
            pct = 100 * count / history['comments'][0]
            pct_text = f'{pct:.1f}%'
        ax.text(iter_num, count, pct_text, ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Users over iterations
    ax = axes[1]
    ax.plot(iterations, history['users'], 'o-', linewidth=2, 
            markersize=8, color='coral', label='Users')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title('Users Retained per Iteration', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(iterations)
    
    # Add percentage labels
    for i, (iter_num, count) in enumerate(zip(iterations, history['users'])):
        if i == 0:
            pct_text = '100%'
        else:
            pct = 100 * count / history['users'][0]
            pct_text = f'{pct:.1f}%'
        ax.text(iter_num, count, pct_text, ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Videos over iterations
    ax = axes[2]
    ax.plot(iterations, history['videos'], 'o-', linewidth=2, 
            markersize=8, color='mediumseagreen', label='Videos')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title('Videos Retained per Iteration', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(iterations)
    
    # Add percentage labels
    for i, (iter_num, count) in enumerate(zip(iterations, history['videos'])):
        if i == 0:
            pct_text = '100%'
        else:
            pct = 100 * count / history['videos'][0]
            pct_text = f'{pct:.1f}%'
        ax.text(iter_num, count, pct_text, ha='center', va='bottom', fontsize=9)
    
    # Overall title
    fig.suptitle(f'Iterative Filtering Convergence\n'
                 f'(Users: ≥{min_videos_per_user} videos | Videos: ≥{min_users_per_video} users)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'filtering_convergence.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    if verbose:
        print(f"✓ Saved: {filepath}")

