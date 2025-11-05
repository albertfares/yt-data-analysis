"""
Data exploration utilities for YouTube comment analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def explore_initial_data(df, output_dir='data', filename_prefix='eda',
                        sample_name='10M', verbose=True):
    """
    Generate comprehensive exploratory analysis of comment data.
    
    Creates 3 separate figures with 2 plots each:
    - Figure 1: User activity and video popularity distributions (power laws)
    - Figure 2: Likes and replies distributions
    - Figure 3: Cumulative distributions (user and video concentration)
    
    Parameters
    ----------
    df : pd.DataFrame
        Comment data with columns: author, video_id, likes, replies
    output_dir : str, default='data'
        Output directory for plots
    filename_prefix : str, default='eda'
        Prefix for output filenames (will create: eda_1_power_laws.png, etc.)
    sample_name : str, default='10M'
        Sample description for title (e.g., '10M', 'Full dataset')
    verbose : bool, default=True
        Print progress and statistics
        
    Returns
    -------
    dict
        Dictionary with statistics and insights
    """
    if verbose:
        print("\nGenerating exploratory data analysis...")
    
    # Basic statistics
    n_comments = len(df)
    n_users = df['author'].nunique()
    n_videos = df['video_id'].nunique()
    
    if verbose:
        print("="*80)
        print("INITIAL DATA OVERVIEW")
        print("="*80)
        print(f"Total comments:     {n_comments:>15,}")
        print(f"Unique users:       {n_users:>15,}")
        print(f"Unique videos:      {n_videos:>15,}")
        print(f"Sample:             {sample_name:>15}")
        print("="*80)
    
    # Comment activity metrics
    comments_per_user = df.groupby('author', observed=True).size()
    comments_per_video = df.groupby('video_id', observed=True).size()
    
    if verbose:
        print(f"\nUser Activity:")
        print(f"  Median comments/user:  {comments_per_user.median():.0f}")
        print(f"  Mean comments/user:    {comments_per_user.mean():.1f}")
        print(f"  Max comments/user:     {comments_per_user.max():,}")
        
        print(f"\nVideo Popularity:")
        print(f"  Median comments/video: {comments_per_video.median():.0f}")
        print(f"  Mean comments/video:   {comments_per_video.mean():.1f}")
        print(f"  Max comments/video:    {comments_per_video.max():,}")
        
        print(f"\nEngagement Metrics:")
        print(f"  Total likes:           {df['likes'].sum():,}")
        print(f"  Total replies:         {df['replies'].sum():,}")
        print(f"  Avg likes/comment:     {df['likes'].mean():.2f}")
        print(f"  Avg replies/comment:   {df['replies'].mean():.2f}")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # =============================================================================
    # FIGURE 1: Power Law Distributions (User Activity + Video Popularity)
    # =============================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Comments per user (log scale)
    data_to_plot = comments_per_user.value_counts().sort_index()
    ax1.loglog(data_to_plot.index, data_to_plot.values, 'o-', markersize=4, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Comments per User', fontsize=12)
    ax1.set_ylabel('Number of Users (log)', fontsize=12)
    ax1.set_title('User Activity Distribution\n(Power Law)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Comments per video (log scale)
    data_to_plot = comments_per_video.value_counts().sort_index()
    ax2.loglog(data_to_plot.index, data_to_plot.values, 'o-', markersize=4, color='coral', alpha=0.7)
    ax2.set_xlabel('Comments per Video', fontsize=12)
    ax2.set_ylabel('Number of Videos (log)', fontsize=12)
    ax2.set_title('Video Popularity Distribution\n(Power Law)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    fig1.suptitle(f'Power Law Distributions ({sample_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filepath1 = os.path.join(output_dir, f'{filename_prefix}_1_power_laws.png')
    fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.show()
    saved_files.append(filepath1)
    if verbose:
        print(f"✓ Saved: {filepath1}")
    
    # =============================================================================
    # FIGURE 2: Engagement Distributions (Likes + Replies)
    # =============================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Likes distribution
    likes_data = df['likes'].value_counts().head(50).sort_index()
    ax1.bar(likes_data.index, likes_data.values, color='gold', edgecolor='black', alpha=0.8)
    ax1.set_xlabel('Likes per Comment', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Comment Likes Distribution\n(Top 50 values)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Replies distribution
    replies_data = df['replies'].value_counts().head(50).sort_index()
    ax2.bar(replies_data.index, replies_data.values, color='orchid', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Replies per Comment', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Comment Replies Distribution\n(Top 50 values)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    
    fig2.suptitle(f'Engagement Distributions ({sample_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filepath2 = os.path.join(output_dir, f'{filename_prefix}_2_engagement.png')
    fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.show()
    saved_files.append(filepath2)
    if verbose:
        print(f"✓ Saved: {filepath2}")
    
    # =============================================================================
    # FIGURE 3: Cumulative Distributions (User + Video Concentration)
    # =============================================================================
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Cumulative distribution - users
    sorted_user_comments = np.sort(comments_per_user.values)
    cumsum = np.cumsum(sorted_user_comments)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    ax1.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2, color='steelblue')
    ax1.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% of comments')
    ax1.axhline(80, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='80% of comments')
    ax1.set_xlabel('Users (sorted by activity)', fontsize=12)
    ax1.set_ylabel('Cumulative % of Comments', fontsize=12)
    ax1.set_title('User Activity: Cumulative Distribution\n(How many users create X% of comments)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Calculate percentiles for annotation
    pct_50_idx = np.searchsorted(cumsum_pct, 50)
    user_concentration_50 = 100 * pct_50_idx / len(cumsum_pct)
    ax1.text(0.95, 0.35, f'{user_concentration_50:.1f}% of users\ncreate 50% of comments', 
             transform=ax1.transAxes, fontsize=10, ha='right', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Cumulative distribution - videos
    sorted_video_comments = np.sort(comments_per_video.values)
    cumsum = np.cumsum(sorted_video_comments)
    cumsum_pct = 100 * cumsum / cumsum[-1]
    ax2.plot(range(len(cumsum_pct)), cumsum_pct, linewidth=2, color='coral')
    ax2.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% of comments')
    ax2.axhline(80, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='80% of comments')
    ax2.set_xlabel('Videos (sorted by popularity)', fontsize=12)
    ax2.set_ylabel('Cumulative % of Comments', fontsize=12)
    ax2.set_title('Video Popularity: Cumulative Distribution\n(How many videos receive X% of comments)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    pct_50_idx = np.searchsorted(cumsum_pct, 50)
    video_concentration_50 = 100 * pct_50_idx / len(cumsum_pct)
    ax2.text(0.95, 0.35, f'{video_concentration_50:.1f}% of videos\nreceive 50% of comments', 
             transform=ax2.transAxes, fontsize=10, ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    fig3.suptitle(f'Cumulative Distributions ({sample_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filepath3 = os.path.join(output_dir, f'{filename_prefix}_3_cumulative.png')
    fig3.savefig(filepath3, dpi=300, bbox_inches='tight')
    plt.show()
    saved_files.append(filepath3)
    if verbose:
        print(f"✓ Saved: {filepath3}")
        print(f"\n✓ Generated {len(saved_files)} exploratory plots")
    
    # Calculate statistics for return
    pct_likes = 100 * df[df['likes'] > 0].shape[0] / len(df)
    pct_replies = 100 * df[df['replies'] > 0].shape[0] / len(df)
    max_possible = n_users * n_videos
    sparsity = 100 * (1 - n_comments / max_possible)
    
    stats = {
        'n_comments': n_comments,
        'n_users': n_users,
        'n_videos': n_videos,
        'user_concentration_50': user_concentration_50,
        'video_concentration_50': video_concentration_50,
        'pct_comments_with_likes': pct_likes,
        'pct_comments_with_replies': pct_replies,
        'sparsity': sparsity,
        'median_comments_per_user': comments_per_user.median(),
        'median_comments_per_video': comments_per_video.median()
    }
    
    # Key insights
    if verbose:
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        print("📊 Distribution: Both user activity and video popularity follow power laws")
        print(f"👥 Concentration: Top {user_concentration_50:.1f}% of users create 50% of all comments")
        print(f"🎥 Concentration: Top {video_concentration_50:.1f}% of videos receive 50% of all comments")
        print(f"💬 Engagement: {pct_likes:.1f}% of comments have likes")
        print(f"💬 Engagement: {pct_replies:.1f}% of comments have replies")
        print(f"🌐 Sparsity: Only {100*(1-sparsity/100):.6f}% of possible user-video pairs have comments")
        print("="*80)
    
    return stats
