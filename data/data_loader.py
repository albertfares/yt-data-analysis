"""
Data loading utilities for YouTube comment analysis.
"""

import json
import pandas as pd
from tqdm.auto import tqdm
import os


def format_number(num):
    """
    Format large numbers in human-readable format (e.g., 16.8M, 1.2B).
    
    Parameters
    ----------
    num : int or float
        Number to format
        
    Returns
    -------
    str
        Formatted number string
        
    Examples
    --------
    >>> format_number(16879028)
    '16.9M'
    >>> format_number(1234567890)
    '1.2B'
    """
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)
    
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.0f}"


def load_comment_data(filepath, nrows=None, verbose=True):
    """
    Load YouTube comment data from TSV file with memory-efficient dtypes.
    
    Parameters
    ----------
    filepath : str
        Path to the TSV file
    nrows : int, optional
        Number of rows to load (None = load all)
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: author, video_id, likes, replies
    """
    if verbose:
        if nrows:
            print(f"Loading first {nrows:,} rows from {filepath}...")
        else:
            print(f"Loading data from {filepath}...")
    
    # Load with string dtypes first (faster than category during load)
    df = pd.read_csv(
        filepath,
        sep='\t',
        nrows=nrows,
        dtype={
            'author': 'str',
            'video_id': 'str',
            'likes': 'int32',
            'replies': 'int32'
        }
    )
    
    if verbose:
        print(f"Data loaded! Converting to efficient dtypes...")
    
    # Convert to category after loading (much faster this way)
    df['author'] = df['author'].astype('category')
    df['video_id'] = df['video_id'].astype('category')
    
    if verbose:
        print(f"\n✓ Data loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        if nrows:
            print(f"  Note: Using first {nrows:,} rows")
    
    return df


def load_channel_data(filepath, verbose=True):
    """
    Load YouTube channel data from TSV file.
    
    Creates a mapping from channel_id to channel information.
    
    Parameters
    ----------
    filepath : str
        Path to the TSV file containing channel data
    verbose : bool, default=True
        Print loading progress
        
    Returns
    -------
    dict
        Dictionary mapping channel_id to dict with keys:
        'name', 'subscribers', 'videos', 'category', 'join_date'
        
    Examples
    --------
    >>> channels = load_channel_data('df_channels_en.tsv')
    >>> print(channels['UC-lHJZR3Gqxm24_Vd_AJ5Yw']['name'])
    'PewDiePie'
    """
    if verbose:
        print(f"Loading channel data from {filepath}...")
    
    df = pd.read_csv(filepath, sep='\t')
    
    # Create channel lookup dictionary
    channels = {}
    for _, row in df.iterrows():
        channel_id = row['channel']
        channels[channel_id] = {
            'name': row.get('name_cc', 'Unknown'),
            'subscribers': row.get('subscribers_cc', 0),
            'videos': row.get('videos_cc', 0),
            'category': row.get('category_cc', 'Unknown'),
            'join_date': row.get('join_date', 'Unknown')
        }
    
    if verbose:
        print(f"✓ Loaded {len(channels):,} channels")
    
    return channels


def load_metadata_for_videos(metadata_path, video_ids, channel_map=None, verbose=True):
    """
    Efficiently load metadata for specific video IDs from a large JSONL file.
    
    Only parses lines that match the requested video IDs and stops early once
    all videos are found. Much faster than loading the entire metadata file.
    
    Parameters
    ----------
    metadata_path : str
        Path to the JSONL metadata file
    video_ids : list or set
        Video IDs to find metadata for
    channel_map : dict, optional
        Dictionary mapping channel_id to channel info (from load_channel_data)
        If provided, replaces channel_id with channel name
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    dict
        Dictionary mapping video_id to metadata dict with keys:
        'title', 'channel', 'channel_id', 'views', 'views_formatted', 
        'upload_date', 'categories', 'duration', 'likes', 'dislikes'
        
    Examples
    --------
    >>> channels = load_channel_data('channels.tsv')
    >>> top_videos = ['video1', 'video2', 'video3']
    >>> metadata = load_metadata_for_videos('metadata.jsonl', top_videos, channels)
    >>> print(metadata['video1']['title'])
    >>> print(metadata['video1']['channel'])  # Now shows channel name, not ID!
    """
    video_ids = set(video_ids)  # Convert to set for O(1) lookup
    video_metadata = {}
    found_count = 0
    total_videos = len(video_ids)
    
    if verbose:
        print(f"\nSearching for metadata for {total_videos} videos...")
        print(f"Scanning: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        # Use tqdm for progress if verbose
        iterator = tqdm(f, desc="Scanning metadata", disable=not verbose, unit=" lines")
        
        for line in iterator:
            # Early exit if all videos found
            if found_count >= total_videos:
                if verbose:
                    print(f"✓ All {total_videos} videos found! Stopping scan.")
                break
            
            try:
                video = json.loads(line.strip())
                # Try different field names for video ID
                video_id = video.get('display_id') or video.get('video_id') or video.get('id')
                
                # Only process if this is one of our target videos
                if video_id and video_id in video_ids:
                    channel_id = video.get('channel_id', 'Unknown')
                    views = video.get('view_count', 0)
                    
                    # Get channel name from map if available
                    if channel_map and channel_id in channel_map:
                        channel_name = channel_map[channel_id]['name']
                    else:
                        channel_name = channel_id  # Fallback to ID
                    
                    video_metadata[video_id] = {
                        'title': video.get('title', 'No title'),
                        'channel': channel_name,
                        'channel_id': channel_id,
                        'views': views,
                        'views_formatted': format_number(views),
                        'upload_date': video.get('upload_date', 'Unknown'),
                        'categories': video.get('categories', []),
                        'duration': video.get('duration', 0),
                        'likes': video.get('like_count', 0),
                        'dislikes': video.get('dislike_count', 0)
                    }
                    found_count += 1
                    
                    if verbose:
                        iterator.set_postfix({'found': f"{found_count}/{total_videos}"})
                        
            except (json.JSONDecodeError, AttributeError):
                continue
    
    if verbose:
        print(f"\n✓ Found metadata for {found_count}/{total_videos} videos")
        if found_count < total_videos:
            missing = video_ids - set(video_metadata.keys())
            print(f"  ⚠ Missing {len(missing)} videos: {list(missing)[:5]}...")
    
    return video_metadata


def load_comments_gz(
    data_path,
    comments_file="youtube_comments.tsv.gz",
    n_chunks=3,
    chunksize=1_000_000,
    usecols=('author', 'video_id', 'likes', 'replies'),
):
    """
    Load YouTube comments from a .tsv.gz file in chunks and return a single DataFrame.
    Prints progress as chunks are loaded.
    """
    comments_path = os.path.join(data_path, comments_file)
    print(f"📄 Loading comments from: {comments_path}")
    print(f"➡️  Chunk size: {chunksize:,} rows | Max chunks: {n_chunks}")

    comments_iter = pd.read_csv(
        comments_path,
        sep="\t",
        compression="gzip",
        chunksize=chunksize,
        usecols=usecols,
    )

    chunks = []
    for i, chunk in enumerate(comments_iter, start=1):
        print(f"   🔹 Loaded chunk {i}")
        chunks.append(chunk)

        if (n_chunks is not None) and (i >= n_chunks):
            print("   ⏹️  Reached chunk limit.")
            break

    print("📌 Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Comments DataFrame shape: {df.shape}")
    return df


def load_videos_gz(
    data_path,
    videos_file="yt_metadata_en.jsonl.gz",
    n_chunks=3,
    chunksize=500_000,
    usecols=None,
):
    """
    Load video metadata from a .jsonl.gz file in chunks and return a single DataFrame.
    Prints progress as chunks are loaded.
    """
    videos_path = os.path.join(data_path, videos_file)
    print(f"📄 Loading videos from: {videos_path}")
    print(f"➡️  Chunk size: {chunksize:,} rows | Max chunks: {n_chunks}")

    videos_iter = pd.read_json(
        videos_path,
        lines=True,
        compression="gzip",
        chunksize=chunksize,
    )

    chunks = []
    for i, chunk in enumerate(videos_iter, start=1):
        print(f"   🔹 Loaded chunk {i}")

        if usecols is not None:
            missing = [c for c in usecols if c not in chunk.columns]
            if missing:
                print(f"     ⚠️ Missing columns in this chunk: {missing}")
            chunk = chunk[[c for c in usecols if c in chunk.columns]]

        chunks.append(chunk)

        if (n_chunks is not None) and (i >= n_chunks):
            print("   ⏹️  Reached chunk limit.")
            break

    print("📌 Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Videos DataFrame shape: {df.shape}")
    return df