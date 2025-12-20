import duckdb
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import os
import json
import time
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
import pickle
from collections import defaultdict


def format_number(n):
    """Format large numbers with commas"""
    return f"{n:,}"

# ==================================================================
# Data Cleaning and Filtering
# ==================================================================

def stream_filter_large_dataset(
    data_path,
    output_path,
    cache_dir=None,
    chunk_size=100_000_000,
    max_chunks=None,
    min_videos_per_user=24,
    min_users_per_video=200,
    min_total_likes_per_user=5,
    use_iterative_filter=False,
    max_iterations=10,
    remove_user_outliers=True,
    remove_video_outliers=False,
    outlier_percentile=99,
    reset_state=False,
    memory_limit='28GB',
    threads=8
):
    """
    Two-pass streaming filter for massive datasets using DuckDB.
    
    Pass 1: Aggregates counts (Videos per User, Users per Video) using DuckDB.
    Pass 2: Streams data in chunks, filtering based on counts from Pass 1.
    
    Args:
        data_path (str): Path to the input TSV/CSV file.
        output_path (str): Path where the filtered CSV will be saved.
        cache_dir (str, optional): Directory to store state/count files. Defaults to output_path's directory.
        chunk_size (int): Number of rows to process per chunk.
        max_chunks (int, optional): Limit processing to N chunks (for testing). None = process all.
        min_videos_per_user (int): Min unique videos a user must comment on.
        min_users_per_video (int): Min unique users a video must have.
        min_total_likes_per_user (int): Min total likes a user must have received.
        use_iterative_filter (bool): If True, loads filtered data into memory for cascading refinement.
        remove_user_outliers (bool): If True, removes top percentile of active users.
        outlier_percentile (int): Percentile cutoff for outliers (e.g., 99).
        reset_state (bool): If True, ignores previous progress and starts over.
        memory_limit (str): DuckDB memory limit (e.g., '28GB').
        threads (int): Number of threads for DuckDB.
    """
    
    # --- Setup Paths ---
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    if cache_dir is None:
        cache_dir = output_path.parent
    else:
        cache_dir = Path(cache_dir)
        
    os.makedirs(cache_dir, exist_ok=True)
    
    state_file = cache_dir / '.filter_state.json'
    counts_file = cache_dir / '.counts_cache_duckdb.json'
    
    print("="*80)
    print("⚡ STREAMING TWO-PASS FILTER (FAST VERSION WITH DUCKDB)")
    print("="*80)
    print(f"Input:        {data_path}")
    print(f"Output:       {output_path}")
    print(f"Cache Dir:    {cache_dir}")
    print(f"Strategy:     DuckDB Aggregation + Streaming {'Iterative ' if use_iterative_filter else ''}Filter")
    print("-" * 80)

    # --- State Management ---
    if reset_state and state_file.exists():
        os.remove(state_file)
        print("🔄 Reset state - starting from beginning")

    state = {'pass1_complete': False, 'pass2_chunks_processed': 0}
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            print(f"📍 Resuming previous session:")
            print(f"   Pass 2 processed: {state['pass2_chunks_processed']} chunks")

    # ============================================================================
    # PASS 1: FAST COUNTING WITH DUCKDB
    # ============================================================================
    
    # Check if we can use cached counts
    cache_valid = False
    skip_chunks = state.get('pass2_chunks_processed', 0)
    
    if skip_chunks > 0:
        print(f"\n📍 Previous session processed {skip_chunks} chunks")
        
    # Try to load counts cache
    user_video_counts = {}
    video_user_counts = {}
    user_total_likes = {}
    
    if counts_file.exists():
        with open(counts_file, 'r') as f:
            cache = json.load(f)
        
        # Validate cache matches current run parameters
        if cache.get('skip_chunks') == skip_chunks and cache.get('max_chunks') == max_chunks:
            print(f"\n📂 Found valid cached counts.")
            user_video_counts = cache['user_video_counts']
            user_total_likes = cache['user_total_likes']
            video_user_counts = cache['video_user_counts']
            cache_valid = True
            print(f"   ✓ Loaded {len(user_video_counts):,} users and {len(video_user_counts):,} videos")
    
    pass1_time = 0
    if not cache_valid:
        print(f"\n[PASS 1] Counting with DuckDB...")
        pass1_start = time.time()
        
        con = duckdb.connect(database=':memory:')
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA threads={threads}")
        con.execute("SET preserve_insertion_order=false")
        con.execute("PRAGMA temp_directory='/tmp'")
        
        # Calculate offsets for incremental counting
        skip_rows = skip_chunks * chunk_size
        max_rows_limit = (max_chunks * chunk_size) if max_chunks else None
        
        limit_clause = f"LIMIT {max_rows_limit}" if max_rows_limit else ""
        offset_clause = f"OFFSET {skip_rows}"
        
        print(f"   Reading data (Skip: {skip_rows:,} rows)...")
        con.execute(f"""
            CREATE TABLE comments AS 
            SELECT * FROM read_csv_auto('{data_path}', 
                delim='\t', header=true, ignore_errors=true,
                columns={{'author': 'VARCHAR', 'video_id': 'VARCHAR', 'likes': 'INTEGER', 'replies': 'INTEGER'}}
            )
            {offset_clause}
            {limit_clause}
        """)
        
        total_rows = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        print(f"   ✓ Registered {total_rows:,} rows for analysis")
        
        # Aggregations
        print("   Aggregating User stats...")
        user_counts_df = con.execute("""
            SELECT author, COUNT(DISTINCT video_id) as video_count, SUM(likes) as total_likes
            FROM comments GROUP BY author
        """).df()
        
        print("   Aggregating Video stats...")
        video_counts_df = con.execute("""
            SELECT video_id, COUNT(DISTINCT author) as user_count
            FROM comments GROUP BY video_id
        """).df()
        
        con.close()
        
        # Convert to dicts
        user_video_counts = dict(zip(user_counts_df['author'], user_counts_df['video_count']))
        user_total_likes = dict(zip(user_counts_df['author'], user_counts_df['total_likes']))
        video_user_counts = dict(zip(video_counts_df['video_id'], video_counts_df['user_count']))
        
        pass1_time = time.time() - pass1_start
        print(f"   ✓ Pass 1 finished in {pass1_time:.1f}s")
        
        # Save cache
        with open(counts_file, 'w') as f:
            json.dump({
                'skip_chunks': skip_chunks,
                'max_chunks': max_chunks,
                'user_video_counts': user_video_counts,
                'user_total_likes': user_total_likes,
                'video_user_counts': video_user_counts
            }, f)

    # --- Outlier Calculation ---
    max_videos_per_user = float('inf')
    max_users_per_video = float('inf')
    
    if remove_user_outliers:
        valid_counts = [c for c in user_video_counts.values() if c >= min_videos_per_user]
        if valid_counts:
            max_videos_per_user = np.percentile(valid_counts, outlier_percentile)
            print(f"   🎯 User Outlier Cutoff (> {outlier_percentile}%): {max_videos_per_user:.0f} videos")

    if remove_video_outliers:
        valid_counts = [c for c in video_user_counts.values() if c >= min_users_per_video]
        if valid_counts:
            max_users_per_video = np.percentile(valid_counts, outlier_percentile)
            print(f"   🎯 Video Outlier Cutoff (> {outlier_percentile}%): {max_users_per_video:.0f} users")

    # ============================================================================
    # PASS 2: FILTERING
    # ============================================================================
    print(f"\n[PASS 2] Streaming & Filtering...")
    pass2_start = time.time()
    
    accumulated_chunks = [] if use_iterative_filter else None
    total_processed = 0
    total_kept = 0
    chunks_read = 0
    
    # DuckDB for Fast Offset Loading
    con2 = duckdb.connect(database=':memory:')
    con2.execute(f"PRAGMA memory_limit='{memory_limit}'") 
    con2.execute(f"PRAGMA threads={threads}")
    
    offset_rows = skip_chunks * chunk_size
    total_rows_to_load = (max_chunks * chunk_size) if max_chunks else None
    limit_clause = f"LIMIT {total_rows_to_load}" if total_rows_to_load else ""
    
    print(f"   Loading data into engine (Offset: {offset_rows:,})...")
    con2.execute(f"""
        CREATE TABLE pass2_data AS 
        SELECT * FROM read_csv_auto('{data_path}', 
            delim='\t', header=true, ignore_errors=true,
            columns={{'author': 'VARCHAR', 'video_id': 'VARCHAR', 'likes': 'INTEGER', 'replies': 'INTEGER'}}
        )
        OFFSET {offset_rows}
        {limit_clause}
    """)
    
    total_loaded = con2.execute("SELECT COUNT(*) FROM pass2_data").fetchone()[0]
    
    if total_loaded == 0:
        print("⚠️ No data found to process.")
        con2.close()
        return

    # Process in chunks
    with tqdm(total=total_loaded, desc="Filtering", unit="rows", unit_scale=True) as pbar:
        chunk_offset = 0
        while True:
            chunk = con2.execute(f"SELECT * FROM pass2_data OFFSET {chunk_offset} LIMIT {chunk_size}").df()
            if len(chunk) == 0:
                break
                
            chunks_read += 1
            total_processed += len(chunk)
            chunk_offset += chunk_size
            
            # Type conversion
            chunk['author'] = chunk['author'].astype(str)
            chunk['video_id'] = chunk['video_id'].astype(str)
            
            # --- APPLY FILTERS ---
            mask = (
                chunk['author'].map(lambda x: min_videos_per_user <= user_video_counts.get(x, 0) <= max_videos_per_user) &
                chunk['video_id'].map(lambda x: min_users_per_video <= video_user_counts.get(x, 0) <= max_users_per_video)
            )
            
            if min_total_likes_per_user > 0:
                mask = mask & chunk['author'].map(lambda x: user_total_likes.get(x, 0) >= min_total_likes_per_user)
                
            chunk_filtered = chunk[mask]
            kept = len(chunk_filtered)
            total_kept += kept
            
            # Handle Output
            if use_iterative_filter:
                if kept > 0:
                    accumulated_chunks.append(chunk_filtered)
            else:
                # Direct Write to CSV
                if kept > 0:
                    mode = 'w' if (chunks_read == 1 and skip_chunks == 0) else 'a'
                    header = (mode == 'w')
                    chunk_filtered.to_csv(output_path, index=False, mode=mode, header=header)
            
            # Update State
            state['pass2_chunks_processed'] = skip_chunks + chunks_read
            with open(state_file, 'w') as f:
                json.dump(state, f)
                
            pbar.update(len(chunk))
            pbar.set_postfix({'kept': f'{total_kept:,}', 'rate': f'{100*total_kept/max(1, total_processed):.1f}%'})

    con2.close()

    # --- Iterative Refinement (Optional) ---
    if use_iterative_filter and accumulated_chunks:
        print(f"\n[Step 2] Iterative Refinement on {total_kept:,} rows...")
        df_combined = pd.concat(accumulated_chunks, ignore_index=True)
        
        for i in range(max_iterations):
            u_counts = df_combined.groupby('author')['video_id'].nunique()
            valid_u = u_counts[u_counts >= min_videos_per_user].index
            
            v_counts = df_combined[df_combined['author'].isin(valid_u)].groupby('video_id')['author'].nunique()
            valid_v = v_counts[v_counts >= min_users_per_video].index
            
            new_df = df_combined[df_combined['author'].isin(valid_u) & df_combined['video_id'].isin(valid_v)]
            
            dropped = len(df_combined) - len(new_df)
            print(f"   Iteration {i+1}: Dropped {dropped:,} rows")
            if dropped == 0:
                break
            df_combined = new_df
            
        print(f"   Saving final output to {output_path}...")
        df_combined.to_csv(output_path, index=False)
        total_kept = len(df_combined)

    total_time = pass1_time + (time.time() - pass2_start)
    print(f"\n{'='*80}")
    print(f"DONE! Kept {total_kept:,} rows ({100*total_kept/max(1, total_processed):.2f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")

# ==================================================================
# Groups and Network Creation
# ==================================================================

def build_video_group_mapping(
    videos_file_path,
    output_dir,
    chunk_size=1_000_000,
    compression="gzip"
):
    """
    Reads video metadata and builds a 'video_id -> group_key' mapping.
    The result is saved as a partitioned Parquet dataset (multiple files).

    Args:
        videos_file_path (str): Path to the input JSONL.GZ file.
        output_dir (str): Directory where Parquet files will be saved.
        chunk_size (int): Number of rows to process at a time.
        compression (str): Compression type for input file (e.g., 'gzip').
    """
    
    # --- Setup Paths ---
    input_path = Path(videos_file_path)
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 BUILDING VIDEO_ID → GROUP_KEY MAPPING")
    print("="*80)
    print(f"Source:     {input_path}")
    print(f"Output Dir: {out_dir}")
    print(f"Chunk size: {chunk_size:,}")
    print("-" * 80)

    usecols = ["display_id", "categories", "channel_id"]
    start_time = time.time()
    total_rows = 0

    # --- Processing Loop ---
    try:
        reader = pd.read_json(
            input_path,
            lines=True,
            compression=compression,
            chunksize=chunk_size
        )
        
        for i, chunk in enumerate(reader, start=1):
            t0 = time.time()
            
            # Filter columns immediately to save memory
            # Note: We check if columns exist to be safe
            available_cols = [c for c in usecols if c in chunk.columns]
            chunk = chunk[available_cols].copy()
            
            if 'categories' not in chunk.columns or 'channel_id' not in chunk.columns:
                print(f"⚠️ Warning: Missing required columns in chunk {i}. Skipping.")
                continue

            # Create Group Key: Category | ChannelID
            chunk["group_key"] = (
                chunk["categories"].astype(str) + "|" + chunk["channel_id"].astype(str)
            )

            # Select only needed columns for mapping
            mapping = chunk[["display_id", "group_key"]]
            
            # Save to Parquet partition
            out_file = out_dir / f"part_{i:04d}.parquet"
            mapping.to_parquet(out_file, index=False)

            # Stats
            rows = len(mapping)
            total_rows += rows
            elapsed = time.time() - start_time
            chunk_time = time.time() - t0

            print(
                f"🔹 Chunk {i:3d} | rows: {rows:>9,} | total: {total_rows:>10,} | "
                f"chunk time: {chunk_time:5.1f}s | elapsed: {elapsed/60:6.1f} min"
            )

    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        raise

    total_time = (time.time() - start_time) / 60
    print("\n" + "="*80)
    print(f"✅ DONE!")
    print(f"📊 Total videos processed: {total_rows:,}")
    print(f"⏱️  Total runtime: {total_time:.1f} minutes")
    print("="*80)

def replace_channel_ids_with_names(
    parquet_dir,
    channel_mapping_tsv,
    output_dir
):
    """
    Replaces channel IDs in the 'group_key' column (format: Category|ChannelID)
    with channel names (format: Category|ChannelName) across a directory of Parquet files.

    Args:
        parquet_dir (str): Directory containing the input Parquet files.
        channel_mapping_tsv (str): Path to the TSV file containing 'channel' and 'name_cc' columns.
        output_dir (str): Directory where the processed Parquet files will be saved.
    """
    
    # --- Setup Paths ---
    input_dir = Path(parquet_dir)
    mapping_file = Path(channel_mapping_tsv)
    out_dir = Path(output_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("="*80)
    print("🔄 REPLACING CHANNEL IDs WITH CHANNEL NAMES IN PARQUET FILES")
    print("="*80)
    print(f"Input Directory:  {input_dir}")
    print(f"Mapping File:     {mapping_file}")
    print(f"Output Directory: {out_dir}")
    print("-" * 80)

    # --- Step 1: Load Channel Mapping ---
    print("\n[1/3] Loading channel mapping...")
    try:
        channels_df = pd.read_csv(mapping_file, sep='\t', usecols=['channel', 'name_cc'])
        # Create a dictionary for fast O(1) lookups
        channel_map = dict(zip(channels_df['channel'], channels_df['name_cc']))
        print(f"   ✓ Loaded {len(channel_map):,} channel mappings")
    except Exception as e:
        print(f"❌ Error loading mapping file: {e}")
        return

    # --- Step 2: Get File List ---
    parquet_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])
    print(f"\n[2/3] Found {len(parquet_files)} parquet files to process")
    
    if not parquet_files:
        print("⚠️ No parquet files found. Aborting.")
        return

    # --- Step 3: Process Files ---
    print("\n[3/3] Processing files...")
    unmapped_channels = set()
    total_rows_processed = 0

    # Helper function for 'apply'
    def replace_id(group_key):
        # Ensure input is a string and has the separator
        if not isinstance(group_key, str) or '|' not in group_key:
            return group_key
        
        # Split from the right to handle potential pipes in category names safely
        category, channel_id = group_key.rsplit('|', 1)
        
        # Look up the name
        channel_name = channel_map.get(channel_id)
        
        if channel_name is None:
            unmapped_channels.add(channel_id)
            return group_key  # Keep original if mapping not found
        
        return f"{category}|{channel_name}"

    # Loop through files
    for p_file in tqdm(parquet_files, desc="Processing Parquet"):
        input_path = input_dir / p_file
        output_path = out_dir / p_file
        
        try:
            df = pd.read_parquet(input_path)
            total_rows_processed += len(df)
            
            # Apply the replacement function
            df['group_key'] = df['group_key'].apply(replace_id)
            
            # Save to new location
            df.to_parquet(output_path, index=False)
            
        except Exception as e:
            print(f"⚠️ Error processing file {p_file}: {e}")

    # --- Summary ---
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed:       {len(parquet_files)}")
    print(f"Total rows processed:  {total_rows_processed:,}")
    print(f"Output directory:      {out_dir}")

    if unmapped_channels:
        print(f"\n⚠️  Warning: {len(unmapped_channels):,} channel IDs were not found in the mapping.")
        print("    (Kept original IDs for these entries)")
        # Show a few examples
        print(f"    Examples: {list(unmapped_channels)[:5]}")
    else:
        print("\n✅ All channel IDs successfully mapped!")

    print("="*80)
    print(f"\n✅ Complete! Results saved to: {out_dir}")

def aggregate_comments_by_author_and_group(
    filtered_comments_csv,
    parquet_mapping_dir,
    output_csv,
    temp_db_path=None,
    chunk_size=50_000_000,
    memory_limit='20GB',
    threads=12
):
    """
    Joins filtered comments with video groups (Parquet) and aggregates by (author, group_key).
    Produces a CSV with columns: author, group_key, comment_count.
    
    Args:
        filtered_comments_csv (str): Path to the filtered comments CSV.
        parquet_mapping_dir (str): Directory containing the video-to-group Parquet files.
        output_csv (str): Path where the aggregated CSV will be saved.
        temp_db_path (str, optional): Path for temporary DuckDB file. Defaults to output folder.
        chunk_size (int): Number of rows to process per chunk.
        memory_limit (str): DuckDB memory limit.
        threads (int): Number of threads for DuckDB.
    """
    
    # --- Setup Paths ---
    comments_path = Path(filtered_comments_csv)
    mapping_dir = Path(parquet_mapping_dir)
    output_path = Path(output_csv)
    
    if temp_db_path is None:
        temp_db_path = output_path.parent / 'aggregate_temp.duckdb'
    else:
        temp_db_path = Path(temp_db_path)
        
    print("="*80)
    print("📊 AGGREGATING COMMENTS BY AUTHOR AND GROUP (DuckDB Chunked)")
    print("="*80)
    print(f"Comments:   {comments_path}")
    print(f"Mapping:    {mapping_dir}")
    print(f"Output:     {output_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print("-" * 80)

    start_time = time.time()

    # --- Step 1: Initialize DuckDB ---
    # Clean up previous temp file if it exists
    if temp_db_path.exists():
        os.remove(temp_db_path)

    print("\n[1/4] Initializing DuckDB...")
    con = duckdb.connect(str(temp_db_path))
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    print(f"   ✓ Configured: {memory_limit} RAM, {threads} Threads")

    try:
        # --- Step 2: Count Rows ---
        print("\n[2/4] Counting input rows...")
        total_rows = con.execute(f"""
            SELECT COUNT(*) FROM read_csv_auto('{comments_path}', sample_size=100000)
        """).fetchone()[0]
        print(f"   ✓ Total rows to process: {total_rows:,}")
        
        num_chunks = (total_rows + chunk_size - 1) // chunk_size

        # --- Step 3: Chunked Processing ---
        print("\n[3/4] Joining and Aggregating...")
        
        # Create storage table
        con.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_results (
                author VARCHAR,
                group_key VARCHAR,
                comment_count BIGINT
            )
        """)
        
        offset = 0
        with tqdm(total=total_rows, unit=" rows", unit_scale=True, desc="Aggregating") as pbar:
            for chunk_idx in range(num_chunks):
                # We select a chunk of comments, JOIN with the Parquet files, 
                # and insert the counts into the results table
                con.execute(f"""
                    INSERT INTO aggregated_results
                    SELECT
                        c.author,
                        m.group_key,
                        COUNT(*) as comment_count
                    FROM (
                        SELECT author, video_id
                        FROM read_csv_auto('{comments_path}', sample_size=100000)
                        LIMIT {chunk_size} OFFSET {offset}
                    ) c
                    INNER JOIN read_parquet('{mapping_dir}/part_*.parquet') m
                        ON c.video_id = m.display_id
                    GROUP BY c.author, m.group_key
                """)
                
                offset += chunk_size
                pbar.update(min(chunk_size, total_rows - (offset - chunk_size)))

        # --- Step 4: Final Merge & Save ---
        print("\n[4/4] Final merge and save...")
        # Join the partial results from chunks and save to CSV
        con.execute(f"""
            COPY (
                SELECT
                    author,
                    group_key,
                    SUM(comment_count) as comment_count
                FROM aggregated_results
                GROUP BY author, group_key
                ORDER BY comment_count DESC
            ) TO '{output_path}' (HEADER, DELIMITER ',')
        """)
        
        # --- Stats ---
        print("\n   Generating stats...")
        sample = con.execute(f"""
            SELECT author, group_key, SUM(comment_count) as cnt
            FROM aggregated_results
            GROUP BY author, group_key
            ORDER BY cnt DESC
            LIMIT 5
        """).fetchall()
        
        print(f"\n   📈 Top 5 pairs:")
        for row in sample:
            print(f"   - {row[0][:15]}... in {row[1][:30]}... ({row[2]:,} comments)")

    finally:
        con.close()
        # Cleanup
        if temp_db_path.exists():
            os.remove(temp_db_path)
            print("\n   ✓ Cleaned up temporary database")

    total_time = (time.time() - start_time) / 60
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total rows processed:  {total_rows:,}")
    print(f"Output saved to:       {output_path}")
    print(f"Total time:            {total_time:.1f} min")
    print("="*80)

def count_unique_commentators(
    input_csv,
    output_csv,
    temp_db_path=None,
    chunk_size=50_000_000,
    memory_limit='20GB',
    threads=12
):
    """
    Counts unique commentators per group using DuckDB with chunked processing.
    Useful for massive datasets where loading everything into pandas fails.

    Args:
        input_csv (str): Path to input CSV (author, group_key).
        output_csv (str): Path to save the counts CSV.
        temp_db_path (str, optional): Path for temporary DuckDB file. Defaults to folder of output_csv.
        chunk_size (int): Rows to process per chunk.
        memory_limit (str): DuckDB memory limit.
        threads (int): Number of threads for DuckDB.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    
    if temp_db_path is None:
        temp_db_path = output_path.parent / 'group_counts_temp.duckdb'
    else:
        temp_db_path = Path(temp_db_path)

    print("="*80)
    print("📊 COUNTING UNIQUE COMMENTATORS PER GROUP (Chunked DuckDB)")
    print("="*80)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print("-" * 80)

    start_time = time.time()

    # --- Step 1: Initialize DuckDB ---
    # We use a file-based DB to handle data larger than RAM
    if temp_db_path.exists():
        os.remove(temp_db_path)
        
    con = duckdb.connect(str(temp_db_path))
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    try:
        # --- Step 2: Count Total Rows ---
        print("[1/4] Counting total rows...")
        total_rows = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{input_path}', sample_size=100000)").fetchone()[0]
        print(f"   ✓ Total rows: {total_rows:,}")
        
        # --- Step 3: Process in Chunks ---
        print("[2/4] Processing chunks...")
        con.execute("""
            CREATE TABLE IF NOT EXISTS group_author_pairs (
                group_key VARCHAR,
                author VARCHAR
            )
        """)
        
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        offset = 0
        
        with tqdm(total=total_rows, unit="rows", unit_scale=True, desc="Processing") as pbar:
            for _ in range(num_chunks):
                con.execute(f"""
                    INSERT INTO group_author_pairs
                    SELECT group_key, author
                    FROM read_csv_auto('{input_path}', sample_size=100000)
                    LIMIT {chunk_size} OFFSET {offset}
                """)
                
                offset += chunk_size
                pbar.update(min(chunk_size, total_rows - (offset - chunk_size)))

        # --- Step 4: Aggregation ---
        print("\n[3/4] Aggregating unique commentators...")
        # Direct copy to CSV is fastest
        con.execute(f"""
            COPY (
                SELECT 
                    group_key,
                    COUNT(DISTINCT author) as unique_commentators
                FROM group_author_pairs
                GROUP BY group_key
                ORDER BY unique_commentators DESC
            ) TO '{output_path}' (HEADER, DELIMITER ',')
        """)
        print(f"   ✓ Saved to {output_path}")

        # --- Stats ---
        print("\n[4/4] Generating stats...")
        stats = con.execute(f"""
            SELECT COUNT(*), MAX(unique_commentators), MEDIAN(unique_commentators), AVG(unique_commentators)
            FROM read_csv_auto('{output_path}')
        """).fetchone()
        
        print(f"   Total Groups: {stats[0]:,}")
        print(f"   Max Commentators: {stats[1]:,}")
        print(f"   Median Commentators: {stats[2]:.0f}")

    finally:
        con.close()
        if temp_db_path.exists():
            os.remove(temp_db_path)
            print("   ✓ Cleaned up temporary database")

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ DONE in {total_time:.1f} min")

def create_group_network_from_overlaps(
    input_csv,
    output_edges_csv,
    temp_db_path=None,
    chunk_size=10_000_000,
    memory_limit='20GB',
    threads=12
):
    """
    Creates a network of groups where edges represent overlapping authors.
    Nodes: group_key
    Edges: pairs of groups (group1, group2)
    Weight: number of shared authors (authors who commented in both groups)

    Args:
        input_csv (str): Path to the author-group-comments CSV.
        output_edges_csv (str): Path where the edge list CSV will be saved.
        temp_db_path (str, optional): Path for temporary DuckDB file.
        chunk_size (int): Number of author-group pairs to process at a time.
        memory_limit (str): DuckDB memory limit.
        threads (int): Number of threads for DuckDB.
    """
    
    # --- Setup Paths ---
    input_path = Path(input_csv)
    output_path = Path(output_edges_csv)
    
    if temp_db_path is None:
        temp_db_path = output_path.parent / 'network_temp.duckdb'
    else:
        temp_db_path = Path(temp_db_path)
        
    print("="*80)
    print("🕸️  CREATING GROUP NETWORK FROM AUTHOR OVERLAPS")
    print("="*80)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Chunk size:  {chunk_size:,} rows")
    print("-" * 80)

    start_time = time.time()

    # --- Step 1: Initialize DuckDB ---
    if temp_db_path.exists():
        os.remove(temp_db_path)

    print("\n[1/5] Initializing DuckDB...")
    con = duckdb.connect(str(temp_db_path))
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    try:
        # --- Step 2: Count Rows ---
        print("\n[2/5] Counting rows...")
        total_rows = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{input_path}')").fetchone()[0]
        print(f"   ✓ Total author-group pairs: {total_rows:,}")
        
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        print(f"   ✓ Will process in {num_chunks} chunks")

        # --- Step 3: Create Edge Table ---
        print("\n[3/5] Creating temporary edge pairs table...")
        con.execute("""
            CREATE TABLE IF NOT EXISTS edge_pairs (
                group1 VARCHAR,
                group2 VARCHAR,
                author VARCHAR
            )
        """)

        # --- Step 4: Process Chunks (Self-Join) ---
        print("\n[4/5] Processing chunks (Self-Join)...")
        offset = 0
        total_pairs_generated = 0
        
        with tqdm(total=total_rows, unit=" rows", unit_scale=True, desc="Generating Edges") as pbar:
            for chunk_idx in range(num_chunks):
                # Load chunk into temp table
                con.execute("DROP TABLE IF EXISTS current_chunk")
                con.execute(f"""
                    CREATE TEMP TABLE current_chunk AS
                    SELECT author, group_key
                    FROM read_csv_auto('{input_path}')
                    LIMIT {chunk_size} OFFSET {offset}
                """)
                
                # Self-join to find groups sharing the same author
                # We use 'group1 < group2' to avoid duplicates (A-B vs B-A) and self-loops (A-A)
                # 
                con.execute("""
                    INSERT INTO edge_pairs
                    SELECT 
                        CASE WHEN a.group_key < b.group_key THEN a.group_key ELSE b.group_key END as group1,
                        CASE WHEN a.group_key < b.group_key THEN b.group_key ELSE a.group_key END as group2,
                        a.author
                    FROM current_chunk a
                    INNER JOIN current_chunk b
                        ON a.author = b.author
                        AND a.group_key != b.group_key
                        AND a.group_key < b.group_key
                """)
                
                # Check progress
                # Note: Counting exact rows in big tables is slow, so we skip it inside the loop for speed
                # unless debugging is needed.
                
                offset += chunk_size
                pbar.update(min(chunk_size, total_rows - (offset - chunk_size)))

        # --- Step 5: Aggregate & Save ---
        print("\n[5/5] Aggregating shared authors and saving...")
        con.execute(f"""
            COPY (
                SELECT 
                    group1,
                    group2,
                    COUNT(DISTINCT author) as shared_authors
                FROM edge_pairs
                GROUP BY group1, group2
                ORDER BY shared_authors DESC
            ) TO '{output_path}' (HEADER, DELIMITER ',')
        """)
        
        # --- Stats ---
        print("\n   Generating stats...")
        edge_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{output_path}')").fetchone()[0]
        stats = con.execute(f"""
            SELECT AVG(shared_authors), MAX(shared_authors), MEDIAN(shared_authors)
            FROM read_csv_auto('{output_path}')
        """).fetchone()
        
        print(f"   Total Edges: {edge_count:,}")
        print(f"   Max Shared:  {stats[1]:,}")
        print(f"   Avg Shared:  {stats[0]:.2f}")

        # Show Top 5
        print("\n📊 Top 5 Edges by Shared Authors:")
        top = con.execute(f"SELECT * FROM read_csv_auto('{output_path}') LIMIT 5").fetchall()
        for row in top:
            g1 = row[0][:30] + "..." if len(row[0]) > 30 else row[0]
            g2 = row[1][:30] + "..." if len(row[1]) > 30 else row[1]
            print(f"   {g1:<35} <--> {g2:<35} ({row[2]:,} shared)")

    finally:
        con.close()
        if temp_db_path.exists():
            os.remove(temp_db_path)
            print("\n   ✓ Cleaned up temporary database")

    total_time = (time.time() - start_time) / 60
    print("\n" + "="*80)
    print("NETWORK SUMMARY")
    print("="*80)
    print(f"Output saved to: {output_path}")
    print(f"Total time:      {total_time:.1f} min")
    print("="*80)

def filter_inter_channel_edges(input_csv, output_csv=None):
    """
    Filters a network edge list to keep ONLY inter-channel edges.
    Removes edges where both groups belong to the same channel.
    
    Args:
        input_csv (str): Path to network edges CSV.
        output_csv (str, optional): Path to save filtered CSV. Defaults to input_name + '_inter_channel.csv'.
    """
    input_path = Path(input_csv)
    if output_csv is None:
        output_csv = input_path.with_name(f"{input_path.stem}_inter_channel{input_path.suffix}")
    
    print("="*80)
    print("🔗 FILTERING TO INTER-CHANNEL EDGES ONLY")
    print("="*80)
    print(f"Input:  {input_path.name}")
    print(f"Output: {Path(output_csv).name}")
    print("-" * 80)

    # Load Data
    print("   Loading network...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"   ✓ Loaded {initial_count:,} edges")

    # Extract Channel IDs (Assumes format "Category|ChannelID")
    # We split by '|' and take the second part (index 1)
    # 
    print("   Extracting channels...")
    channel1 = df['group1'].astype(str).str.split('|').str[1]
    channel2 = df['group2'].astype(str).str.split('|').str[1]

    # Filter
    print("   Filtering...")
    mask_inter = channel1 != channel2
    df_filtered = df[mask_inter].copy()
    
    removed_count = initial_count - len(df_filtered)
    pct_kept = (len(df_filtered) / initial_count) * 100 if initial_count > 0 else 0
    
    print(f"   ✓ Removed {removed_count:,} intra-channel edges")
    print(f"   ✓ Kept {len(df_filtered):,} inter-channel edges ({pct_kept:.1f}%)")

    # Save
    if len(df_filtered) > 0:
        df_filtered.to_csv(output_csv, index=False)
        print(f"\n✅ SAVED: {output_csv}")
        
        # Show top example
        top = df_filtered.iloc[0]
        print(f"   Top Edge: {top['group1']} <--> {top['group2']}")
    else:
        print("\n⚠️ No inter-channel edges found.")

def create_pmi_network(
    edges_csv,
    group_sizes_csv,
    output_csv,
    min_pmi=1.0,
    min_shared=3,
    n_users=40_656_008
):
    """
    Creates a Pointwise Mutual Information (PMI) network from pre-computed edges.
    
    This function is "INSTANT" because it uses the small summary CSVs (edges list 
    and group counts) rather than processing the raw comment data again.

    Args:
        edges_csv (str): Path to the edges CSV (e.g., 'group_network_edges.csv').
        group_sizes_csv (str): Path to group counts CSV (e.g., 'group_commentator_counts.csv').
        output_csv (str): Path to save the filtered network.
        min_pmi (float): Minimum PMI score to keep an edge.
                         1.0 ~= 2.7x stronger than random
                         3.0 ~= 20x stronger than random
        min_shared (int): Minimum shared commentators to keep an edge.
        n_users (int): Total unique users in the ecosystem (used for probability calc).
                       Defaults to ~40.6M.

    Returns:
        pd.DataFrame: The filtered dataframe containing the network edges.
    """
    
    # --- Setup ---
    edges_path = Path(edges_csv)
    sizes_path = Path(group_sizes_csv)
    output_path = Path(output_csv)
    
    print("="*80)
    print("🚀 CREATING GROUP NETWORK WITH PMI (INSTANT VERSION)")
    print("="*80)
    print(f"Edges:       {edges_path.name}")
    print(f"Group Sizes: {sizes_path.name}")
    print(f"Output:      {output_path.name}")
    print("-" * 80)
    print(f"Filters:     PMI >= {min_pmi} ({np.exp(min_pmi):.1f}x random)")
    print(f"             Shared >= {min_shared}")
    print("-" * 80)

    # --- Step 1: Load Group Sizes ---
    print(f"[1/4] Loading group sizes...")
    group_sizes_df = pd.read_csv(sizes_path)
    group_size_map = dict(zip(group_sizes_df['group_key'], group_sizes_df['unique_commentators']))
    print(f"   ✓ Loaded {len(group_sizes_df):,} groups")

    # --- Step 2: Load Edges ---
    print(f"[2/4] Loading edges...")
    edges_df = pd.read_csv(edges_path)
    print(f"   ✓ Loaded {len(edges_df):,} edges")
    
    # Column Detection (Robustness)
    if 'group1' in edges_df.columns:
        g1_col, g2_col = 'group1', 'group2'
    elif 'group_key1' in edges_df.columns:
        g1_col, g2_col = 'group_key1', 'group_key2'
    else:
        raise ValueError(f"Could not find group columns in {edges_path.name}")

    shared_col = next((c for c in ['shared_commentators', 'shared_authors', 'weight'] if c in edges_df.columns), None)
    if not shared_col:
        raise ValueError("Could not find shared commentators/weight column")

    # --- Step 3: Compute PMI ---
    print(f"[3/4] Computing PMI & Scores...")
    
    # Map sizes
    edges_df['group1_size'] = edges_df[g1_col].map(group_size_map)
    edges_df['group2_size'] = edges_df[g2_col].map(group_size_map)
    
    # Drop rows where size is missing (if any)
    initial_len = len(edges_df)
    edges_df = edges_df.dropna(subset=['group1_size', 'group2_size'])
    if len(edges_df) < initial_len:
        print(f"   ⚠️ Dropped {initial_len - len(edges_df)} edges due to missing group sizes.")

    # PMI Calculation: log( (Shared * N) / (SizeA * SizeB) )
    # 
    edges_df['pmi'] = (
        np.log(edges_df[shared_col]) + 
        np.log(n_users) - 
        np.log(edges_df['group1_size']) - 
        np.log(edges_df['group2_size'])
    )

    # Hybrid Score: PMI * log(Shared)
    edges_df['score'] = edges_df['pmi'] * np.log(edges_df[shared_col])
    
    print(f"   ✓ PMI computed (Median: {edges_df['pmi'].median():.2f})")

    # --- Step 4: Filter & Save ---
    print(f"[4/4] Filtering...")
    
    filtered_df = edges_df[
        (edges_df['pmi'] >= min_pmi) & 
        (edges_df[shared_col] >= min_shared)
    ].copy()
    
    # Rename columns for consistency
    output_df = filtered_df[[g1_col, g2_col, shared_col, 'group1_size', 'group2_size', 'pmi', 'score']].copy()
    output_df.columns = ['group1', 'group2', 'shared_commentators', 'group1_size', 'group2_size', 'pmi', 'score']
    
    # Sort by Score (Best edges first)
    output_df = output_df.sort_values('score', ascending=False)
    
    kept_pct = (len(output_df) / len(edges_df)) * 100 if len(edges_df) > 0 else 0
    print(f"   ✓ Kept {len(output_df):,} edges ({kept_pct:.1f}%)")
    
    if len(output_df) > 0:
        output_df.to_csv(output_path, index=False)
        print(f"\n✅ SAVED: {output_path}")
        print(f"   Nodes: {len(set(output_df['group1']) | set(output_df['group2'])):,}")
        print(f"   Top edge: {output_df.iloc[0]['group1']} <-> {output_df.iloc[0]['group2']} (Score: {output_df.iloc[0]['score']:.2f})")
    else:
        print("\n⚠️ No edges met the criteria. Try lowering thresholds.")

    return output_df

# ==================================================================
# Comments & Groups Analysis
# ==================================================================

def analyze_author_group_comments(input_csv, output_txt):
    """
    Analyzes the relationship between authors and groups using DuckDB.

    This function calculates basic file statistics, counts unique authors and groups,
    and performs a category analysis to identify top categories by pair count
    and top groups by author participation.

    Args:
        input_csv (str): Path to the input CSV file containing author-group pairs.
        output_txt (file object): An open file object (writable) to log the analysis results. 
                                  Results are printed to stdout and written to this file.

    Returns:
        None
    """
    def print_both(text, output_file=output_txt):
        """Print to both console and file"""
        print(text)
        if output_file:
            output_file.write(text + "\n")
        """Analyze the author-group comments CSV file"""
    
    print_both("=" * 80)
    print_both("AUTHOR-GROUP COMMENTS ANALYSIS")
    print_both("=" * 80)
    
    start_time = time.time()
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    print_both("\n1. BASIC FILE INFO")
    print_both("-" * 80)
    
    # Get file size
    file_size = Path(input_csv).stat().st_size
    print_both(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Get total row count
    print_both("\nCounting total rows (author-group pairs)...")
    result = con.execute(f"""
        SELECT COUNT(*) as total_rows
        FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchone()
    total_rows = result[0]
    print_both(f"Total pairs: {format_number(total_rows)}")
    
    # Get column info
    print_both("\n2. COLUMN INFORMATION")
    print_both("-" * 80)
    columns = con.execute(f"""
        DESCRIBE SELECT * FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchall()
    
    for col in columns:
        print_both(f"  - {col[0]}: {col[1]}")
    
    column_names = [col[0] for col in columns]
    
    # Get unique counts
    print_both("\n3. UNIQUE VALUE COUNTS")
    print_both("-" * 80)
    
    if 'author' in column_names:
        print_both("\nCounting unique authors...")
        result = con.execute(f"""
            SELECT COUNT(DISTINCT author) as unique_authors
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Unique authors: {format_number(result[0])}")
    
    if 'group_key' in column_names:
        print_both("\nCounting unique groups...")
        result = con.execute(f"""
            SELECT COUNT(DISTINCT group_key) as unique_groups
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Unique groups: {format_number(result[0])}")
    
    # Total comments
    if 'comment_count' in column_names:
        print_both("\nTotal comments across all pairs...")
        result = con.execute(f"""
            SELECT SUM(comment_count) as total_comments
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Total comments: {format_number(result[0])}")
    
    # Category analysis
    if 'group_key' in column_names:
        print_both("\n4. CATEGORY ANALYSIS")
        print_both("-" * 80)
        
        print_both("\nTop 15 categories by number of pairs:")
        results = con.execute(f"""
            SELECT 
                SPLIT_PART(group_key, '|', 1) as category,
                COUNT(*) as pair_count,
                COUNT(DISTINCT author) as unique_authors,
                COUNT(DISTINCT group_key) as unique_channels,
                SUM(comment_count) as total_comments
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            WHERE group_key IS NOT NULL
            GROUP BY category
            ORDER BY pair_count DESC
            LIMIT 15
        """).fetchall()
        
        print_both(f"{'Rank':<6} {'Category':<30} {'Pairs':>12} {'Authors':>10} {'Channels':>10} {'Comments':>12}")
        print_both("-" * 86)
        for i, (category, pairs, authors, channels, comments) in enumerate(results, 1):
            print_both(f"{i:<6} {category:<30} {format_number(pairs):>12} {format_number(authors):>10} {format_number(channels):>10} {format_number(comments):>12}")
        
        print_both("\n\nTop 15 groups by number of commenting authors:")
        results = con.execute(f"""
            SELECT 
                group_key,
                SPLIT_PART(group_key, '|', 1) as category,
                SPLIT_PART(group_key, '|', 2) as channel,
                COUNT(DISTINCT author) as unique_authors,
                SUM(comment_count) as total_comments
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            WHERE group_key IS NOT NULL
            GROUP BY group_key, category, channel
            ORDER BY unique_authors DESC
            LIMIT 15
        """).fetchall()
        
        print_both(f"{'Rank':<6} {'Category':<20} {'Channel':<40} {'Authors':>10} {'Comments':>12}")
        print_both("-" * 90)
        for i, (group_key, category, channel, authors, comments) in enumerate(results, 1):
            print_both(f"{i:<6} {category:<20} {channel:<40} {format_number(authors):>10} {format_number(comments):>12}")
    
    con.close()
    
    elapsed = time.time() - start_time
    print_both("\n" + "=" * 80)
    print_both(f"Analysis completed in {elapsed:.2f} seconds")
    print_both("=" * 80)

def analyze_filtered_streaming(input_csv, output_txt):
    """
    Performs statistical analysis on filtered streaming comment data.

    This function connects to DuckDB to analyze comment distributions. It calculates
    unique counts for videos, authors, and groups, and generates percentile statistics 
    (min, median, 99th%, max) for comment volume per group and per author.

    Args:
        input_csv (str): Path to the input CSV file containing streaming comment data.
        output_txt (file object): An open file object (writable) to log the analysis results.

    Returns:
        None
    """
    def print_both(text, output_file=output_txt):
        """Print to both console and file"""
        print(text)
        if output_file:
            output_file.write(text + "\n")
    """Analyze the filtered streaming CSV file"""
    
    print_both("=" * 80)
    print_both("FILTERED STREAMING DATA ANALYSIS")
    print_both("=" * 80)
    
    start_time = time.time()
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    print_both("\n1. BASIC FILE INFO")
    print_both("-" * 80)
    
    # Get file size
    file_size = Path(input_csv).stat().st_size
    print_both(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Get total row count
    print_both("\nCounting total rows...")
    result = con.execute(f"""
        SELECT COUNT(*) as total_rows
        FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchone()
    total_rows = result[0]
    print_both(f"Total rows: {format_number(total_rows)}")
    
    # Get column info
    print_both("\n2. COLUMN INFORMATION")
    print_both("-" * 80)
    columns = con.execute(f"""
        DESCRIBE SELECT * FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchall()
    
    for col in columns:
        print_both(f"  - {col[0]}: {col[1]}")
    
    # Get unique counts for key columns
    print_both("\n3. UNIQUE VALUE COUNTS")
    print_both("-" * 80)
    
    # Check if common columns exist and analyze them
    column_names = [col[0] for col in columns]
    
    if 'video_id' in column_names or 'display_id' in column_names:
        vid_col = 'video_id' if 'video_id' in column_names else 'display_id'
        print_both(f"\nCounting unique {vid_col}s...")
        result = con.execute(f"""
            SELECT COUNT(DISTINCT {vid_col}) as unique_videos
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Unique videos: {format_number(result[0])}")
    
    if 'author' in column_names or 'author_id' in column_names:
        author_col = 'author' if 'author' in column_names else 'author_id'
        print_both(f"\nCounting unique {author_col}s...")
        result = con.execute(f"""
            SELECT COUNT(DISTINCT {author_col}) as unique_authors
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Unique authors: {format_number(result[0])}")
    
    if 'group_key' in column_names:
        print_both("\nCounting unique group_keys...")
        result = con.execute(f"""
            SELECT COUNT(DISTINCT group_key) as unique_groups
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        print_both(f"Unique groups: {format_number(result[0])}")
    
    # Category analysis if group_key exists
    if 'group_key' in column_names:
        print_both("\n4. CATEGORY ANALYSIS (from group_key)")
        print_both("-" * 80)
        
        print_both("\nTop 10 categories by comment count:")
        results = con.execute(f"""
            SELECT 
                SPLIT_PART(group_key, '|', 1) as category,
                COUNT(*) as comment_count,
                COUNT(DISTINCT SPLIT_PART(group_key, '|', 2)) as unique_channels
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            WHERE group_key IS NOT NULL
            GROUP BY category
            ORDER BY comment_count DESC
            LIMIT 10
        """).fetchall()
        
        for i, (category, count, channels) in enumerate(results, 1):
            print_both(f"  {i:2d}. {category:30s} | Comments: {format_number(count):>12s} | Channels: {format_number(channels):>8s}")
        
        print_both("\nTop 10 channels by comment count:")
        results = con.execute(f"""
            SELECT 
                group_key,
                SPLIT_PART(group_key, '|', 1) as category,
                SPLIT_PART(group_key, '|', 2) as channel,
                COUNT(*) as comment_count
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            WHERE group_key IS NOT NULL
            GROUP BY group_key, category, channel
            ORDER BY comment_count DESC
            LIMIT 10
        """).fetchall()
        
        for i, (group_key, category, channel, count) in enumerate(results, 1):
            print_both(f"  {i:2d}. {category} | {channel:40s} | Comments: {format_number(count):>12s}")
    
    # Distribution analysis
    print_both("\n5. DISTRIBUTION ANALYSIS")
    print_both("-" * 80)
    
    if 'group_key' in column_names:
        print_both("\nComments per group statistics:")
        results = con.execute(f"""
            WITH group_counts AS (
                SELECT 
                    group_key,
                    COUNT(*) as comment_count
                FROM read_csv_auto('{input_csv}', sample_size=2000000)
                WHERE group_key IS NOT NULL
                GROUP BY group_key
            )
            SELECT 
                MIN(comment_count) as min_comments,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY comment_count) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY comment_count) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY comment_count) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY comment_count) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY comment_count) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY comment_count) as p99,
                MAX(comment_count) as max_comments,
                AVG(comment_count) as avg_comments
            FROM group_counts
        """).fetchone()
        
        print_both(f"  Min:     {format_number(int(results[0]))}")
        print_both(f"  10th %:  {format_number(int(results[1]))}")
        print_both(f"  25th %:  {format_number(int(results[2]))}")
        print_both(f"  Median:  {format_number(int(results[3]))}")
        print_both(f"  75th %:  {format_number(int(results[4]))}")
        print_both(f"  90th %:  {format_number(int(results[5]))}")
        print_both(f"  99th %:  {format_number(int(results[6]))}")
        print_both(f"  Max:     {format_number(int(results[7]))}")
        print_both(f"  Average: {format_number(int(results[8]))}")
    
    if author_col in column_names:
        print_both(f"\nComments per author statistics:")
        results = con.execute(f"""
            WITH author_counts AS (
                SELECT 
                    {author_col},
                    COUNT(*) as comment_count
                FROM read_csv_auto('{input_csv}', sample_size=2000000)
                WHERE {author_col} IS NOT NULL
                GROUP BY {author_col}
            )
            SELECT 
                MIN(comment_count) as min_comments,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY comment_count) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY comment_count) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY comment_count) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY comment_count) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY comment_count) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY comment_count) as p99,
                MAX(comment_count) as max_comments,
                AVG(comment_count) as avg_comments
            FROM author_counts
        """).fetchone()
        
        print_both(f"  Min:     {format_number(int(results[0]))}")
        print_both(f"  10th %:  {format_number(int(results[1]))}")
        print_both(f"  25th %:  {format_number(int(results[2]))}")
        print_both(f"  Median:  {format_number(int(results[3]))}")
        print_both(f"  75th %:  {format_number(int(results[4]))}")
        print_both(f"  90th %:  {format_number(int(results[5]))}")
        print_both(f"  99th %:  {format_number(int(results[6]))}")
        print_both(f"  Max:     {format_number(int(results[7]))}")
        print_both(f"  Average: {format_number(int(results[8]))}")
    
    con.close()
    
    elapsed = time.time() - start_time
    print_both("\n" + "=" * 80)
    print_both(f"Analysis completed in {elapsed:.2f} seconds")
    print_both("=" * 80)

def analyze_pmi_network(input_csv, output_txt):
    """
    Analyzes the structure and strength of the PMI (Pointwise Mutual Information) network.

    This function evaluates the network graph where nodes are groups/channels. 
    It calculates:
    - Network density and total edges.
    - Node degree distribution (percentiles).
    - Distribution of PMI scores and 'Score' (PMI * log(shared_commentators)).
    - Connectivity between categories (Inter-category vs Intra-category edges).

    Args:
        input_csv (str): Path to the CSV file containing network edges (group1, group2, scores).
        output_txt (file object): An open file object (writable) to log the analysis results.

    Returns:
        None
    """
    def print_both(text, output_file=output_txt):
        """Print to both console and file"""
        print(text)
        if output_file:
            output_file.write(text + "\n")
    """Analyze the PMI network edges CSV file"""
        
    print_both("=" * 80)
    print_both("PMI NETWORK EDGES ANALYSIS")
    print_both("=" * 80)
    
    start_time = time.time()
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    print_both("\n1. BASIC FILE INFO")
    print_both("-" * 80)
    
    # Get file size
    file_size = Path(input_csv).stat().st_size
    print_both(f"File size: {file_size / (1024*3):.2f} GB ({file_size / (1024*2):.2f} MB)")
    
    # Get total edge count
    print_both("\nCounting total edges...")
    result = con.execute(f"""
        SELECT COUNT(*) as total_edges
        FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchone()
    total_edges = result[0]
    print_both(f"Total edges: {format_number(total_edges)}")
    
    # Get column info
    print_both("\n2. COLUMN INFORMATION")
    print_both("-" * 80)
    columns = con.execute(f"""
        DESCRIBE SELECT * FROM read_csv_auto('{input_csv}', sample_size=2000000)
    """).fetchall()
    
    for col in columns:
        print_both(f"  - {col[0]}: {col[1]}")
    
    column_names = [col[0] for col in columns]
    
    # Get unique node count
    print_both("\n3. NETWORK STRUCTURE")
    print_both("-" * 80)
    
    print_both("\nCounting unique nodes...")
    result = con.execute(f"""
        WITH all_nodes AS (
            SELECT group1 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
            UNION
            SELECT group2 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
        )
        SELECT COUNT(DISTINCT node) as unique_nodes
        FROM all_nodes
    """).fetchone()
    unique_nodes = result[0]
    print_both(f"Unique nodes (groups): {format_number(unique_nodes)}")
    
    # Calculate network density
    max_edges = (unique_nodes * (unique_nodes - 1)) / 2
    density = (total_edges / max_edges) * 100 if max_edges > 0 else 0
    print_both(f"\nNetwork density: {density:.4f}%")
    print_both(f"  (Actual edges: {format_number(total_edges)} / Possible edges: {format_number(int(max_edges))})")
    
    # Degree distribution
    print_both("\n4. DEGREE DISTRIBUTION")
    print_both("-" * 80)
    
    print_both("\nCalculating node degrees...")
    results = con.execute(f"""
        WITH degrees AS (
            SELECT node, COUNT(*) as degree
            FROM (
                SELECT group1 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
                UNION ALL
                SELECT group2 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
            )
            GROUP BY node
        )
        SELECT 
            MIN(degree) as min_degree,
            PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY degree) as p10,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY degree) as p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY degree) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY degree) as p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY degree) as p90,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY degree) as p99,
            MAX(degree) as max_degree,
            AVG(degree) as avg_degree
        FROM degrees
    """).fetchone()
    
    print_both(f"  Min:     {format_number(int(results[0]))}")
    print_both(f"  10th %:  {format_number(int(results[1]))}")
    print_both(f"  25th %:  {format_number(int(results[2]))}")
    print_both(f"  Median:  {format_number(int(results[3]))}")
    print_both(f"  75th %:  {format_number(int(results[4]))}")
    print_both(f"  90th %:  {format_number(int(results[5]))}")
    print_both(f"  99th %:  {format_number(int(results[6]))}")
    print_both(f"  Max:     {format_number(int(results[7]))}")
    print_both(f"  Average: {results[8]:.2f}")
    
    # Top nodes by degree
    print_both("\n\nTop 15 nodes by degree:")
    results = con.execute(f"""
        WITH degrees AS (
            SELECT node, COUNT(*) as degree
            FROM (
                SELECT group1 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
                UNION ALL
                SELECT group2 as node FROM read_csv_auto('{input_csv}', sample_size=2000000)
            )
            GROUP BY node
        )
        SELECT 
            node,
            SPLIT_PART(node, '|', 1) as category,
            SPLIT_PART(node, '|', 2) as channel,
            degree
        FROM degrees
        ORDER BY degree DESC
        LIMIT 15
    """).fetchall()
    
    print_both(f"{'Rank':<6} {'Category':<20} {'Channel':<45} {'Degree':>10}")
    print_both("-" * 85)
    for i, (node, category, channel, degree) in enumerate(results, 1):
        channel_display = (channel[:42] + '...') if len(channel) > 45 else channel
        print_both(f"{i:<6} {category:<20} {channel_display:<45} {format_number(degree):>10}")
    
    # Edge weight analysis (shared_commentators)
    if 'shared_commentators' in column_names:
        print_both("\n5. SHARED COMMENTATORS DISTRIBUTION")
        print_both("-" * 80)
        
        results = con.execute(f"""
            SELECT 
                MIN(shared_commentators) as min_shared,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY shared_commentators) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY shared_commentators) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY shared_commentators) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY shared_commentators) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY shared_commentators) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY shared_commentators) as p99,
                MAX(shared_commentators) as max_shared,
                AVG(shared_commentators) as avg_shared,
                SUM(shared_commentators) as total_shared
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        
        print_both(f"  Min:     {format_number(int(results[0]))}")
        print_both(f"  10th %:  {format_number(int(results[1]))}")
        print_both(f"  25th %:  {format_number(int(results[2]))}")
        print_both(f"  Median:  {format_number(int(results[3]))}")
        print_both(f"  75th %:  {format_number(int(results[4]))}")
        print_both(f"  90th %:  {format_number(int(results[5]))}")
        print_both(f"  99th %:  {format_number(int(results[6]))}")
        print_both(f"  Max:     {format_number(int(results[7]))}")
        print_both(f"  Average: {results[8]:.2f}")
        print_both(f"  Total:   {format_number(int(results[9]))}")
    
    # PMI analysis
    if 'pmi' in column_names:
        print_both("\n6. PMI (POINTWISE MUTUAL INFORMATION) DISTRIBUTION")
        print_both("-" * 80)
        
        results = con.execute(f"""
            SELECT 
                MIN(pmi) as min_pmi,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY pmi) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY pmi) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY pmi) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY pmi) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY pmi) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY pmi) as p99,
                MAX(pmi) as max_pmi,
                AVG(pmi) as avg_pmi
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        
        print_both(f"  Min:     {results[0]:.4f} ({format_number(int(2.718**results[0]))}x stronger than random)")
        print_both(f"  10th %:  {results[1]:.4f} ({format_number(int(2.718**results[1]))}x)")
        print_both(f"  25th %:  {results[2]:.4f} ({format_number(int(2.718**results[2]))}x)")
        print_both(f"  Median:  {results[3]:.4f} ({format_number(int(2.718**results[3]))}x)")
        print_both(f"  75th %:  {results[4]:.4f} ({format_number(int(2.718**results[4]))}x)")
        print_both(f"  90th %:  {results[5]:.4f} ({format_number(int(2.718**results[5]))}x)")
        print_both(f"  99th %:  {results[6]:.4f} ({format_number(int(2.718**results[6]))}x)")
        print_both(f"  Max:     {results[7]:.4f} ({format_number(int(2.718**results[7]))}x)")
        print_both(f"  Average: {results[8]:.4f}")
        
        print_both("\n\nTop 15 edges by PMI:")
        results = con.execute(f"""
            SELECT 
                group1,
                SPLIT_PART(group1, '|', 1) as cat1,
                SPLIT_PART(group1, '|', 2) as chan1,
                group2,
                SPLIT_PART(group2, '|', 1) as cat2,
                SPLIT_PART(group2, '|', 2) as chan2,
                pmi,
                shared_commentators
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            ORDER BY pmi DESC
            LIMIT 15
        """).fetchall()
        
        print_both(f"{'Rank':<6} {'Group 1':<35} {'Group 2':<35} {'PMI':>8} {'Shared':>8}")
        print_both("-" * 95)
        for i, (g1, cat1, chan1, g2, cat2, chan2, pmi, shared) in enumerate(results, 1):
            g1_display = f"{cat1}|{chan1[:25]}..." if len(chan1) > 28 else f"{cat1}|{chan1}"
            g2_display = f"{cat2}|{chan2[:25]}..." if len(chan2) > 28 else f"{cat2}|{chan2}"
            print_both(f"{i:<6} {g1_display:<35} {g2_display:<35} {pmi:>8.4f} {format_number(shared):>8}")
    
    # Score analysis
    if 'score' in column_names:
        print_both("\n7. SCORE (PMI × log(shared_commentators)) DISTRIBUTION")
        print_both("-" * 80)
        
        results = con.execute(f"""
            SELECT 
                MIN(score) as min_score,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY score) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY score) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY score) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY score) as p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY score) as p90,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY score) as p99,
                MAX(score) as max_score,
                AVG(score) as avg_score
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        
        print_both(f"  Min:     {results[0]:.4f}")
        print_both(f"  10th %:  {results[1]:.4f}")
        print_both(f"  25th %:  {results[2]:.4f}")
        print_both(f"  Median:  {results[3]:.4f}")
        print_both(f"  75th %:  {results[4]:.4f}")
        print_both(f"  90th %:  {results[5]:.4f}")
        print_both(f"  99th %:  {results[6]:.4f}")
        print_both(f"  Max:     {results[7]:.4f}")
        print_both(f"  Average: {results[8]:.4f}")
        
        print_both("\n\nTop 15 edges by Score:")
        results = con.execute(f"""
            SELECT 
                group1,
                SPLIT_PART(group1, '|', 1) as cat1,
                SPLIT_PART(group1, '|', 2) as chan1,
                group2,
                SPLIT_PART(group2, '|', 1) as cat2,
                SPLIT_PART(group2, '|', 2) as chan2,
                score,
                pmi,
                shared_commentators
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
            ORDER BY score DESC
            LIMIT 15
        """).fetchall()
        
        print_both(f"{'Rank':<6} {'Group 1':<30} {'Group 2':<30} {'Score':>8} {'PMI':>8} {'Shared':>8}")
        print_both("-" * 95)
        for i, (g1, cat1, chan1, g2, cat2, chan2, score, pmi, shared) in enumerate(results, 1):
            g1_display = f"{cat1}|{chan1[:22]}..." if len(chan1) > 25 else f"{cat1}|{chan1}"
            g2_display = f"{cat2}|{chan2[:22]}..." if len(chan2) > 25 else f"{cat2}|{chan2}"
            print_both(f"{i:<6} {g1_display:<30} {g2_display:<30} {score:>8.4f} {pmi:>8.4f} {format_number(shared):>8}")
    
    # Category-level analysis
    print_both("\n8. CATEGORY-LEVEL ANALYSIS")
    print_both("-" * 80)
    
    print_both("\nEdges between categories:")
    results = con.execute(f"""
        SELECT 
            SPLIT_PART(group1, '|', 1) as cat1,
            SPLIT_PART(group2, '|', 1) as cat2,
            COUNT(*) as edge_count,
            AVG(pmi) as avg_pmi,
            AVG(shared_commentators) as avg_shared,
            SUM(shared_commentators) as total_shared
        FROM read_csv_auto('{input_csv}', sample_size=2000000)
        GROUP BY cat1, cat2
        ORDER BY edge_count DESC
        LIMIT 20
    """).fetchall()
    
    print_both(f"{'Rank':<6} {'Category 1':<20} {'Category 2':<20} {'Edges':>8} {'Avg PMI':>9} {'Avg Shared':>12} {'Total Shared':>13}")
    print_both("-" * 95)
    for i, (cat1, cat2, edges, avg_pmi, avg_shared, total_shared) in enumerate(results, 1):
        print_both(f"{i:<6} {cat1:<20} {cat2:<20} {format_number(edges):>8} {avg_pmi:>9.4f} {avg_shared:>12.2f} {format_number(int(total_shared)):>13}")
    
    print_both("\n\nInter-category vs Intra-category edges:")
    results = con.execute(f"""
        SELECT 
            CASE 
                WHEN SPLIT_PART(group1, '|', 1) = SPLIT_PART(group2, '|', 1) THEN 'Same Category'
                ELSE 'Different Category'
            END as edge_type,
            COUNT(*) as edge_count,
            AVG(pmi) as avg_pmi,
            AVG(shared_commentators) as avg_shared
        FROM read_csv_auto('{input_csv}', sample_size=2000000)
        GROUP BY edge_type
    """).fetchall()
    
    for edge_type, count, avg_pmi, avg_shared in results:
        pct = (count / total_edges) * 100
        print_both(f"  {edge_type:<20}: {format_number(count):>10} ({pct:>5.2f}%) | Avg PMI: {avg_pmi:>6.4f} | Avg Shared: {avg_shared:>8.2f}")
    
    # NULL value analysis
    print_both("\n9. DATA QUALITY (NULL VALUES)")
    print_both("-" * 80)
    
    has_nulls = False
    for col_info in columns:
        col_name = col_info[0]
        result = con.execute(f"""
            SELECT 
                COUNT(*) - COUNT({col_name}) as null_count,
                (COUNT() - COUNT({col_name})) * 100.0 / COUNT() as null_pct
            FROM read_csv_auto('{input_csv}', sample_size=2000000)
        """).fetchone()
        
        if result[0] > 0:
            print_both(f"  {col_name:30s} | NULLs: {format_number(result[0]):>12s} ({result[1]:.2f}%)")
            has_nulls = True
    
    if not has_nulls:
        print_both("  ✓ No NULL values found in any column")
    
    con.close()
    
    elapsed = time.time() - start_time
    print_both("\n" + "=" * 80)
    print_both(f"Analysis completed in {elapsed:.2f} seconds")
    print_both("=" * 80)

# ==================================================================
# Data Generation for the Website Visualizations
# ==================================================================

def generate_violin_json(file_path, output_path):
    """
    Generates a Plotly JSON artifact for a facetted violin chart representing node distributions.

    This function processes network edge data to aggregate node statistics (Size, Degree, Avg_Score).
    It applies Log10 transformations to these metrics and creates an interactive violin plot
    facetted by category. The output includes dropdown buttons to switch between metrics
    and custom HTML tooltips.

    Args:
        file_path (str): Path to the input CSV file containing network edge data.
        output_path (str): Path where the resulting Plotly JSON file will be saved.

    Returns:
        None: The function saves a JSON file to disk.
    """
    print("Generating Violin Chart Data...")

    # --- 2. LOAD DATA ---
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    df = pd.read_csv(file_path)
    if df.empty:
        print("Warning: Input dataframe is empty.")
        return

    # --- 3. PREPROCESS DATA ---
    # Prepare two sides of edges to get full node stats
    df_1 = df[['group1', 'group1_size', 'score']].rename(columns={'group1': 'Node', 'group1_size': 'Size', 'score': 'Edge_Score'})
    df_2 = df[['group2', 'group2_size', 'score']].rename(columns={'group2': 'Node', 'group2_size': 'Size', 'score': 'Edge_Score'})
    all_nodes = pd.concat([df_1, df_2])

    # Aggregate stats per node
    node_stats = all_nodes.groupby('Node').agg(
        Size=('Size', 'mean'), 
        Degree=('Edge_Score', 'count'), 
        Avg_Score=('Edge_Score', 'mean')
    ).reset_index()

    # Create Categories & Log metrics
    def get_category(s):
        return s.split('|')[0] if pd.notna(s) else "Unknown"

    node_stats['Category'] = node_stats['Node'].apply(get_category)
    node_stats['Log_Size'] = np.log10(node_stats['Size'] + 1)
    node_stats['Log_Degree'] = np.log10(node_stats['Degree'] + 1)
    node_stats['Log_Score'] = np.log10(node_stats['Avg_Score'] + 1)

    # Filter: Keep only top 15 categories
    top_cats = node_stats['Category'].value_counts().nlargest(15).index
    data = node_stats[node_stats['Category'].isin(top_cats)].copy()
    data.sort_values('Category', inplace=True)
    categories = data['Category'].unique()

    # --- 4. HTML TOOLTIP GENERATOR ---
    def generate_table_html(cat_name, series_data):
        stats = series_data.describe()
        html = f"""
        <span style='font-size:14px; font-weight:bold;'>{cat_name}</span><br>
        <table style="border: 1px solid black; border-collapse: collapse; background-color: white; font-family: Arial; font-size: 12px; color: black;">
            <tr style="background-color: #f2f2f2; border-bottom: 1px solid #ddd;">
                <th style="padding: 4px; text-align: left;">Statistique</th>
                <th style="padding: 4px; text-align: right;">Valeur</th>
            </tr>
            <tr><td style="padding: 4px;">Moyenne</td><td style="padding: 4px; text-align: right;">{stats['mean']:.2f}</td></tr>
            <tr><td style="padding: 4px;">Médiane</td><td style="padding: 4px; text-align: right;">{stats['50%']:.2f}</td></tr>
            <tr><td style="padding: 4px;">Min</td><td style="padding: 4px; text-align: right;">{stats['min']:.2f}</td></tr>
            <tr><td style="padding: 4px;">Max</td><td style="padding: 4px; text-align: right;">{stats['max']:.2f}</td></tr>
            <tr style="border-top: 1px solid #ddd;"><td style="padding: 4px;"><i>Nbr Noeuds</i></td><td style="padding: 4px; text-align: right;"><i>{int(stats['count'])}</i></td></tr>
        </table>
        <extra></extra>
        """
        return html

    # --- 5. BUILD BASE FIGURE ---
    fig = px.violin(
        data, 
        y="Log_Score", 
        color="Category", 
        facet_col="Category", 
        facet_col_wrap=5, 
        facet_row_spacing=0.1,
        box=True, points=False,
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Apply custom HTML tooltips to initial traces
    for trace in fig.data:
        cat = trace.name
        if cat in categories:
            subset = data[data['Category'] == cat]['Log_Score']
            trace.hovertemplate = generate_table_html(cat, subset)

    # Clean up annotations (Remove "Category=")
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Category=", "")))
    
    # Style the traces
    fig.update_traces(
        name="",  
        meanline_visible=True,
        jitter=0.05,
        width=0.8,
        hoveron="violins"
    )

    # Clean Axes
    fig.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        matches='y', showticklabels=True, showgrid=True, title=None, 
        showspikes=False
    )   
    metrics = ['Log_Score', 'Log_Degree', 'Log_Size']
    labels = ['Score (Log10)', 'Degree (Log10)', 'Size (Log10)']
    
    buttons = []
    for m, label in zip(metrics, labels):
        y_updates = []
        html_updates = []
        
        
        for cat in categories: 
            cat_data = data[data['Category'] == cat]
            y_vals = cat_data[m]
            y_updates.append(y_vals)
            html_updates.append(generate_table_html(cat, y_vals))
            
        buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"y": y_updates, "hovertemplate": html_updates}, 
                {"title": f"Distribution: {label}"}
            ]
        ))
# --- 6.5 FIX: MANUALLY FORCE ALL AXES TO MATCH ---
    # This loop iterates through 'xaxis', 'xaxis2', 'yaxis', 'yaxis2', etc.
    # and forces the style explicitly, preventing the "Master Axis" glitch.
    
    for axis_name in fig.layout:
        if axis_name.startswith('xaxis'):
            # Force X-Axis Clean (No labels, no grid)
            fig.layout[axis_name].showticklabels = False
            fig.layout[axis_name].showgrid = False
            fig.layout[axis_name].zeroline = False
            fig.layout[axis_name].title = None
            
        elif axis_name.startswith('yaxis'):
            # Force Y-Axis Visible (Labels yes, Grid yes)
            fig.layout[axis_name].showticklabels = True
            fig.layout[axis_name].showgrid = True
            fig.layout[axis_name].title = None
            # Ensure the "Master" axis doesn't look different
            fig.layout[axis_name].zeroline = False
            fig.layout[axis_name].showspikes = False

    # Force the background color explicitly so the website CSS doesn't override it
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    
    # --- 7. FINAL LAYOUT & EXPORT ---
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.12, xanchor="left", yanchor="top", bgcolor="white")],
        showlegend=False,
        margin=dict(t=100),
        height=900,
        width=1200,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            namelength=-1 
        )
    )

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    fig.write_json(output_path)
    print(f"Saved JSON data! Open '{output_path}' to use it.")

def create_category_network_from_overlaps(
    input_csv,
    output_edges_csv,
    output_stats_csv,
    temp_db_path=None,
    chunk_size=50_000_000,
    memory_limit='20GB',
    threads=12
):
    """
    Creates a network of CATEGORIES based on shared commentators.
    Extracts category from 'group_key' (format: Category|Channel).
    
    Nodes: Categories
    Edges: Undirected link weighted by number of shared unique commentators.
    
    Args:
        input_csv (str): Path to 'author_group_comments.csv'.
        output_edges_csv (str): Path to save the edge list.
        output_stats_csv (str): Path to save category node stats (size).
        temp_db_path (str, optional): Path for temporary DuckDB file.
        chunk_size (int): Rows to process per chunk.
        memory_limit (str): DuckDB memory limit.
        threads (int): Number of threads for DuckDB.
    """
    
    # --- Setup Paths ---
    input_path = Path(input_csv)
    out_edges_path = Path(output_edges_csv)
    out_stats_path = Path(output_stats_csv)
    
    if temp_db_path is None:
        temp_db_path = out_edges_path.parent / 'category_network_temp.duckdb'
    else:
        temp_db_path = Path(temp_db_path)
        
    print("="*80)
    print("🕸️  CREATING CATEGORY NETWORK FROM SHARED COMMENTATORS")
    print("="*80)
    print(f"Input:       {input_path}")
    print(f"Edges Out:   {out_edges_path}")
    print(f"Stats Out:   {out_stats_path}")
    print(f"Chunk Size:  {chunk_size:,} rows")
    print("-" * 80)

    start_time = time.time()

    # --- Step 1: Initialize DuckDB ---
    if temp_db_path.exists():
        os.remove(temp_db_path)

    print("\n[1/6] Initializing DuckDB...")
    con = duckdb.connect(str(temp_db_path))
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    try:
        # --- Step 2: Count Rows ---
        print("\n[2/6] Counting rows...")
        total_rows = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{input_path}', sample_size=100000)").fetchone()[0]
        print(f"   ✓ Total rows: {total_rows:,}")
        
        num_chunks = (total_rows + chunk_size - 1) // chunk_size

        # --- Step 3: Extract Categories ---
        print("\n[3/6] Extracting categories from group_keys...")
        con.execute("""
            CREATE TABLE IF NOT EXISTS author_categories (
                author VARCHAR,
                category VARCHAR
            )
        """)
        
        offset = 0
        with tqdm(total=total_rows, unit=" rows", unit_scale=True, desc="Extracting") as pbar:
            for chunk_idx in range(num_chunks):
                # Extract category (part before |)
                con.execute(f"""
                    INSERT INTO author_categories
                    SELECT DISTINCT
                        author,
                        SPLIT_PART(group_key, '|', 1) as category
                    FROM read_csv_auto('{input_path}', sample_size=100000)
                    LIMIT {chunk_size} OFFSET {offset}
                """)
                
                offset += chunk_size
                pbar.update(min(chunk_size, total_rows - (offset - chunk_size)))

        # --- Step 4: Analyze Category Sizes ---
        print("\n[4/6] Analyzing categories...")
        # Save node stats to CSV
        con.execute(f"""
            COPY (
                SELECT 
                    category,
                    COUNT(DISTINCT author) as unique_commentators
                FROM author_categories
                GROUP BY category
                ORDER BY unique_commentators DESC
            ) TO '{out_stats_path}' (HEADER, DELIMITER ',')
        """)
        
        # Display top categories
        stats = con.execute(f"SELECT * FROM read_csv_auto('{out_stats_path}') LIMIT 10").fetchall()
        print(f"\n   📊 Top Categories by Unique Commentators:")
        for cat, count in stats:
            print(f"   - {cat:<30} {count:>10,}")

        # --- Step 5: Create Edges (Self-Join) ---
        print("\n[5/6] Creating category network edges...")
        
        # Self-join on author to find category pairs
        # Filter: a.category < b.category ensures distinct pairs and no self-loops
        con.execute("""
            CREATE TABLE IF NOT EXISTS category_edges AS
            SELECT 
                CASE WHEN a.category < b.category THEN a.category ELSE b.category END as category1,
                CASE WHEN a.category < b.category THEN b.category ELSE a.category END as category2,
                a.author
            FROM author_categories a
            INNER JOIN author_categories b
                ON a.author = b.author
                AND a.category != b.category
                AND a.category < b.category
        """)
        
        # --- Step 6: Aggregate & Save ---
        print("\n[6/6] Aggregating shared commentators and saving...")
        
        con.execute(f"""
            COPY (
                SELECT 
                    category1,
                    category2,
                    COUNT(DISTINCT author) as shared_commentators
                FROM category_edges
                GROUP BY category1, category2
                ORDER BY shared_commentators DESC
            ) TO '{out_edges_path}' (HEADER, DELIMITER ',')
        """)

        # --- Summary Stats ---
        edge_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_edges_path}')").fetchone()[0]
        
        if edge_count > 0:
            top_edges = con.execute(f"SELECT * FROM read_csv_auto('{out_edges_path}') LIMIT 5").fetchall()
            print("\n   📊 Top Category Connections:")
            for row in top_edges:
                print(f"   {row[0]:<25} <--> {row[1]:<25} ({row[2]:,} shared)")
        else:
            print("   ⚠️ No edges created.")

    finally:
        con.close()
        if temp_db_path.exists():
            os.remove(temp_db_path)
            print("\n   ✓ Cleaned up temporary database")

    total_time = (time.time() - start_time) / 60
    print("\n" + "="*80)
    print("CATEGORY NETWORK SUMMARY")
    print("="*80)
    print(f"Output Edges: {out_edges_path}")
    print(f"Output Stats: {out_stats_path}")
    print(f"Total time:   {total_time:.1f} min")
    print("="*80)

def generate_sunburst_json(file_path, output_path, CATEGORY_PALETTE, DEFAULT_COLOR="#CCCCCC", COLOR_INTERNAL="#B0BEC5", COLOR_EXTERNAL="#FFCCBC"):
    """
    Generates a Plotly Sunburst chart JSON representing shared readership flow between categories.

    This function builds a hierarchical tree structure from the flat CSV data. The hierarchy is:
    [Category -> Internal/External Flow -> Connected Category]. 
    It aggregates 'shared_commentators' volume to size the slices and applies a color scheme 
    based on the category palette.

    Args:
        file_path (str): Path to the input CSV file containing group-to-group shared commentator data.
        output_path (str): Path where the resulting Plotly JSON file will be saved.
        CATEGORY_PALETTE (dict): A dictionary mapping category names (strings) to hex color codes.
        DEFAULT_COLOR (str, optional): Fallback color for unmatched categories. Defaults to "#CCCCCC".
        COLOR_INTERNAL (str, optional): Color for the 'Internal' branch ring. Defaults to "#B0BEC5".
        COLOR_EXTERNAL (str, optional): Color for the 'External' branch ring. Defaults to "#FFCCBC".

    Returns:
        None: The function saves a JSON file to disk.
    """
    print("Generating Sunburst Data...")

    # --- 2. LOAD & PREPARE DATA ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Helper: Extract category name
    def get_cat(s):
        return s.split('|')[0] if pd.notna(s) else "Unknown"

    df['Cat1'] = df['group1'].apply(get_cat)
    df['Cat2'] = df['group2'].apply(get_cat)

    # Aggregate Volume (Shared Users)
    agg_df = df.groupby(['Cat1', 'Cat2'])['shared_commentators'].sum().reset_index()

    # Filter: Remove noise (Bottom 20%) to keep chart clean
    threshold = agg_df['shared_commentators'].quantile(0.20)
    agg_filtered = agg_df[agg_df['shared_commentators'] > threshold].copy()

    # --- 3. BUILD HIERARCHY (Tree Structure) ---
    tree = {}

    def insert_data(center, neighbor, vol):
        if center not in tree: tree[center] = {'Internal': {}, 'External': {}}
        
        type_label = 'Internal' if center == neighbor else 'External'
        
        # Add to the specific branch
        if neighbor in tree[center][type_label]:
            tree[center][type_label][neighbor] += vol
        else:
            tree[center][type_label][neighbor] = vol

    # Populate Tree (Symmetric)
    for _, row in agg_filtered.iterrows():
        c1, c2, vol = row['Cat1'], row['Cat2'], row['shared_commentators']
        insert_data(c1, c2, vol)
        if c1 != c2: insert_data(c2, c1, vol)

    # --- 4. FLATTEN TO PLOTLY LISTS ---
    ids = []
    labels = []
    parents = []
    values = []
    node_colors = []

    for center, branches in tree.items():
        # Calculate Branch Totals
        internal_sum = sum(branches['Internal'].values())
        external_sum = sum(branches['External'].values())
        center_total = internal_sum + external_sum
        
        if center_total == 0: continue

        # LEVEL 0: CENTER (The Category)
        ids.append(center)
        labels.append(center)
        parents.append("")
        values.append(center_total)
        node_colors.append(CATEGORY_PALETTE.get(center, DEFAULT_COLOR))

        # LEVEL 1: TYPE (Internal vs External)
        for type_label, neighbors in branches.items():
            branch_sum = sum(neighbors.values())
            
            if branch_sum > 0:
                branch_id = f"{center} - {type_label}"
                
                ids.append(branch_id)
                labels.append(type_label)
                parents.append(center)
                values.append(branch_sum)
                
                if type_label == 'Internal':
                    node_colors.append(COLOR_INTERNAL)
                else:
                    node_colors.append(COLOR_EXTERNAL)

                # LEVEL 2: LEAVES (Connected Categories)
                for neighbor, vol in neighbors.items():
                    leaf_id = f"{branch_id} - {neighbor}"
                    
                    ids.append(leaf_id)
                    labels.append(neighbor)
                    parents.append(branch_id)
                    values.append(vol)
                    
                    node_colors.append(CATEGORY_PALETTE.get(neighbor, DEFAULT_COLOR))

    # --- 5. GENERATE FIGURE OBJECT ---
    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=node_colors),
        branchvalues="total",
        
        # DISPLAY: Label + Percentage of Parent Ring
        textinfo="label+percent parent",
        
        # HOVER INFO
        hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>'
    ))

    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        width=1000,
        height=1000,
        font=dict(family="Arial", size=14)
    )

    
    fig.write_json(output_path)
    
    print(f"Saved JSON data! Open '{output_path}' to use it.")

def generate_network_community_json(
    graph_pickle_path,
    output_json_path,
    communities_cache_path=None,
    num_communities=15,
    max_nodes_per_comm=300
):
    """
    Generates a JSON file containing network community data for visualization.
    Detects communities using Louvain, calculates layouts, and exports node/edge positions.

    Args:
        graph_pickle_path (str): Path to the pickled NetworkX graph (.gpickle).
        output_json_path (str): Path to save the resulting JSON file.
        communities_cache_path (str, optional): Path to cache community detection results.
        num_communities (int): Number of top communities (by size) to process.
        max_nodes_per_comm (int): Max nodes to include per community (to keep JSON small).
    """
    
    # --- Setup Paths ---
    graph_path = Path(graph_pickle_path)
    output_path = Path(output_json_path)
    
    if communities_cache_path:
        cache_path = Path(communities_cache_path)
    else:
        cache_path = graph_path.parent / f"{graph_path.stem}_communities.pkl"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📊 GENERATING NETWORK COMMUNITY DATA (JSON)")
    print("="*80)
    print(f"Graph Input:  {graph_path}")
    print(f"JSON Output:  {output_path}")
    print(f"Cache File:   {cache_path}")
    print("-" * 80)

    # --- Step 1: Load Graph ---
    print("\n[1/4] Loading graph...")
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"   ✓ Loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return

    # --- Step 2: Extract Largest Component ---
    print("\n[2/4] Extracting largest connected component...")
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    print(f"   ✓ Component size: {G_main.number_of_nodes():,} nodes")

    # --- Step 3: Load/Compute Communities ---
    print("\n[3/4] Loading communities...")
    
    if cache_path.exists():
        print(f"   📦 Loading from cache...")
        with open(cache_path, 'rb') as f:
            communities = pickle.load(f)
        print(f"   ✓ Loaded {len(communities)} communities")
    else:
        print("   ⚙️  Computing Louvain communities (this might take a moment)...")
        # Try-except block for different NetworkX versions regarding community detection
        try:
            communities = nx.community.louvain_communities(G_main, seed=42)
        except AttributeError:
            # Fallback for older NetworkX or if python-louvain is installed instead
            import community as community_louvain
            partition = community_louvain.best_partition(G_main)
            communities_dict = {}
            for node, comm_id in partition.items():
                communities_dict.setdefault(comm_id, []).append(node)
            communities = list(communities_dict.values())
            
        communities = sorted(communities, key=len, reverse=True)
        
        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(communities, f)
        print(f"   ✓ Computed and cached {len(communities)} communities")

    # --- Step 4: Process & Export ---
    print(f"\n[4/4] Processing top {num_communities} communities...")

    community_data = []

    for idx, comm_nodes in enumerate(tqdm(communities[:num_communities], desc="Processing"), 1):
        comm_size = len(comm_nodes)
        
        # Create subgraph for this community
        G_comm = G_main.subgraph(comm_nodes).copy()
        
        # 1. Calculate Category Counts (on FULL community)
        category_counts = {}
        for node in G_comm.nodes():
            # Format is Category|ChannelID
            parts = str(node).split('|')
            category = parts[0] if len(parts) > 0 else "Unknown"
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 2. Filter Nodes (Keep top degree nodes for visualization)
        if len(G_comm) > max_nodes_per_comm:
            degrees = dict(G_comm.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes_per_comm]
            G_comm = G_comm.subgraph(top_nodes).copy()
        
        # 3. Compute Layout (Force-directed)
        # k=2/sqrt(n) is often good, but fixed k=2 works well for small subgraphs
        pos = nx.spring_layout(G_comm, k=2, iterations=50, seed=42)
        
        # Get degrees of the SUBGRAPH
        degrees = dict(G_comm.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        # 4. Prepare Node Data
        node_data = []
        for node in G_comm.nodes():
            x, y = pos[node]
            degree = degrees[node]
            
            parts = str(node).split('|')
            category = parts[0] if len(parts) > 0 else "Unknown"
            channel = parts[1] if len(parts) > 1 else "Unknown"
            
            # Size calculation for visual importance
            size = 5 + (degree / max_degree) * 25
            
            node_data.append({
                'id': str(node),
                'x': float(x),
                'y': float(y),
                'category': category,
                'channel': channel,
                'degree': degree,
                'size': float(size)
            })
        
        # Sort nodes so larger ones render on top if needed, or listed first
        node_data.sort(key=lambda n: n['degree'], reverse=True)
        
        # 5. Prepare Edge Data
        edge_data = []
        for u, v in G_comm.edges():
            if u in pos and v in pos:
                edge_data.append({
                    'x0': float(pos[u][0]), 
                    'y0': float(pos[u][1]),
                    'x1': float(pos[v][0]), 
                    'y1': float(pos[v][1])
                })
        
        # 6. Identify "Label" channels (top 3 by degree)
        top_channels = [(n['channel'], n['degree']) for n in node_data[:3]]
        
        community_data.append({
            'id': idx,
            'total_nodes': comm_size,
            'loaded_nodes': len(node_data),
            'nodes': node_data,
            'edges': edge_data,
            'top_channels': top_channels,
            'category_counts': category_counts
        })

    # --- Save JSON ---
    print(f"\nSaving to {output_path.name}...")
    with open(output_path, 'w') as f:
        json.dump(community_data, f)

    print("\n" + "="*80)
    print("✅ JSON GENERATION COMPLETE!")
    print(f"📂 File: {output_path}")
    print("="*80)

    import pandas as pd

def generate_recommender_json(
    edges_csv_path,
    output_json_path,
    top_n_channels=200
):
    """
    Generates a JSON file containing pre-computed shortest paths between top channels
    and different categories. Used for the 'Rabbit Hole Recommender' visualization.

    Args:
        edges_csv_path (str): Path to the network edges CSV.
        output_json_path (str): Path to save the output JSON.
        top_n_channels (int): Number of most connected channels to include in the graph.
    """
    
    # --- Setup Paths ---
    input_path = Path(edges_csv_path)
    output_path = Path(output_json_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🚀 GENERATING RECOMMENDER JSON DATA")
    print("="*80)
    print(f"Input Edges:   {input_path}")
    print(f"Output JSON:   {output_path}")
    print(f"Top Channels:  {top_n_channels}")
    print("-" * 80)

    # --- Step 1: Load Network ---
    print("\n[1/4] Loading network...")
    if not input_path.exists():
        # Fallback check for inter-channel specific file
        fallback = input_path.parent / f"{input_path.stem}_inter_channel{input_path.suffix}"
        if fallback.exists():
            print(f"   ⚠️ Input not found, trying fallback: {fallback.name}")
            input_path = fallback
        else:
            raise FileNotFoundError(f"Could not find {input_path}")

    edges_df = pd.read_csv(input_path)
    print(f"   ✓ Loaded {len(edges_df):,} edges")

    # --- Step 2: Find Top Channels ---
    print(f"\n[2/4] Finding top {top_n_channels} channels by degree...")
    
    channel_degree = defaultdict(int)
    for _, row in edges_df.iterrows():
        # Extract channel name from 'Category|Channel' string
        try:
            ch1 = str(row['group1']).split('|')[1]
            ch2 = str(row['group2']).split('|')[1]
            channel_degree[ch1] += 1
            channel_degree[ch2] += 1
        except IndexError:
            continue # Skip malformed rows

    top_channels = sorted(channel_degree.items(), key=lambda x: x[1], reverse=True)[:top_n_channels]
    top_channel_names = set([ch for ch, _ in top_channels])
    
    print(f"   ✓ Selected {len(top_channel_names)} channels")
    print(f"   ✓ Top channel: {top_channels[0][0]} (Degree: {top_channels[0][1]})")

    # --- Step 3: Filter Graph ---
    print("\n[3/4] Filtering graph & building network...")
    
    # Filter edges where BOTH nodes are in the top channel list
    # We do this manually to ensure we split the strings correctly check against our set
    mask = edges_df.apply(lambda x: 
                          str(x['group1']).split('|')[1] in top_channel_names and 
                          str(x['group2']).split('|')[1] in top_channel_names, axis=1)
    
    filtered_df = edges_df[mask].copy()
    print(f"   ✓ Filtered to {len(filtered_df):,} edges")

    # Build Graph
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        # Weight is inverse of score (higher score = shorter distance/stronger link)
        weight = 1.0 / row['score'] if row['score'] > 0 else 1.0
        G.add_edge(row['group1'], row['group2'], weight=weight)
        
    print(f"   ✓ Graph built: {G.number_of_nodes():,} nodes")

    # --- Step 4: Compute Shortest Paths ---
    print("\n[4/4] Computing shortest paths (Rabbit Holes)...")
    
    # 4a. Inventory of Groups & Categories
    groups_info = []
    for group in G.nodes():
        parts = str(group).split('|')
        if len(parts) >= 2:
            category = parts[0]
            channel = parts[1]
            groups_info.append({
                'group': group,
                'category': category,
                'channel': channel
            })
            
    groups_df = pd.DataFrame(groups_info)
    if groups_df.empty:
        print("   ❌ Error: No valid groups found to parse.")
        return

    categories = sorted(groups_df['category'].unique().tolist())
    print(f"   ✓ Found {len(categories)} categories")

    # 4b. Path Calculation
    # Logic: For every group in the graph, find the shortest path to ANY group in every other Category
    paths_lookup = {}
    
    # Pre-group nodes by category for faster lookup
    nodes_by_category = defaultdict(list)
    for _, row in groups_df.iterrows():
        nodes_by_category[row['category']].append(row['group'])

    count = 0
    # Iterate over every node in the graph
    for start_group in G.nodes():
        parts = str(start_group).split('|')
        if len(parts) < 2: continue
        start_cat, start_ch = parts[0], parts[1]
        
        # Calculate distance to every other category
        for target_category in categories:
            if target_category == start_cat:
                continue
            
            # Find the closest node in the target category
            target_nodes = nodes_by_category[target_category]
            
            shortest_path = None
            shortest_length = float('inf')
            
            # This can be slow, but for top 200 channels (small graph) it's acceptable
            # Optimization: could use multi-source Dijkstra if needed, but simple loop is fine here
            for target_group in target_nodes:
                if target_group in G:
                    try:
                        # Calculate weighted shortest path
                        path = nx.shortest_path(G, source=start_group, target=target_group, weight='weight')
                        dist = len(path) # Hop count for simplicity, or sum weights
                        
                        if dist < shortest_length:
                            shortest_path = path
                            shortest_length = dist
                            
                            # Heuristic optimization: if we find a very short path (direct neighbor), stop looking
                            if dist <= 2: 
                                break
                    except nx.NetworkXNoPath:
                        continue
            
            if shortest_path:
                # Format path for JSON
                path_info = [{'category': g.split('|')[0], 'channel': g.split('|')[1]} for g in shortest_path]
                
                # Key format: "StartGroup|TargetCategory" -> "Gaming|PewDiePie|Politics"
                key = f"{start_group}|{target_category}"
                
                paths_lookup[key] = {
                    'path_info': path_info,
                    'length': len(shortest_path) - 1
                }
                count += 1

    print(f"   ✓ Computed {count:,} optimal paths")

    # --- Step 5: Save Output ---
    # Build a dictionary of channels -> categories they belong to
    channels_dict = {}
    for channel in groups_df['channel'].unique():
        channel_groups = groups_df[groups_df['channel'] == channel]
        channels_dict[channel] = {
            'categories': sorted(channel_groups['category'].unique().tolist())
        }

    output_data = {
        'pathsData': paths_lookup,
        'channelsData': channels_dict,
        'allCategories': categories
    }

    print(f"\nSaving JSON to {output_path.name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*80)
    print("✅ JSON GENERATED SUCCESSFULLY")
    print(f"📂 File: {output_path}")
    print("="*80)

    import pandas as pd

def generate_sankey_json(
    edges_csv_path,
    output_json_path,
    top_n_channels=200
):
    """
    Generates a JSON file for the Interactive Sankey Diagram.
    Calculates reachability (shortest paths) from a channel's categories to all other categories.

    Args:
        edges_csv_path (str): Path to the network edges CSV.
        output_json_path (str): Path to save the output JSON.
        top_n_channels (int): Number of most connected channels to include.
    """
    
    # --- Setup Paths ---
    input_path = Path(edges_csv_path)
    output_path = Path(output_json_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📊 GENERATING SANKEY DIAGRAM DATA (JSON)")
    print("="*80)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Top Channels: {top_n_channels}")
    print("-" * 80)

    # --- Step 1: Load Network ---
    print("\n[1/3] Loading and processing network...")
    if not input_path.exists():
         # Fallback check
        fallback = input_path.parent / f"{input_path.stem}_inter_channel{input_path.suffix}"
        if fallback.exists():
            print(f"   ⚠️ Input not found, trying fallback: {fallback.name}")
            input_path = fallback
        else:
            raise FileNotFoundError(f"Could not find {input_path}")
            
    edges_df = pd.read_csv(input_path)
    print(f"   ✓ Loaded {len(edges_df):,} edges")

    # --- Step 2: Filter Top Channels ---
    # Find top channels by degree
    channel_degree = defaultdict(int)
    for _, row in edges_df.iterrows():
        try:
            ch1 = str(row['group1']).split('|')[1]
            ch2 = str(row['group2']).split('|')[1]
            channel_degree[ch1] += 1
            channel_degree[ch2] += 1
        except IndexError:
            continue

    top_channels = sorted(channel_degree.items(), key=lambda x: x[1], reverse=True)[:top_n_channels]
    top_channel_names = set([ch for ch, _ in top_channels])
    
    print(f"   ✓ Selected top {len(top_channel_names)} channels")

    # Filter edges to only those between top channels
    mask = edges_df.apply(lambda x: 
                          str(x['group1']).split('|')[1] in top_channel_names and 
                          str(x['group2']).split('|')[1] in top_channel_names, axis=1)
    
    filtered_df = edges_df[mask].copy()
    print(f"   ✓ Filtered to {len(filtered_df):,} edges")

    # --- Step 3: Build Graph ---
    print("\n[2/3] Building graph and computing reachability...")
    G = nx.Graph()
    for _, row in filtered_df.iterrows():
        # Use simple unweighted edges for BFS shortest path finding
        G.add_edge(row['group1'], row['group2'])
    
    print(f"   ✓ Graph: {G.number_of_nodes():,} nodes")

    # Get all categories
    all_categories = set()
    groups_info = []
    
    for group in G.nodes():
        parts = str(group).split('|')
        if len(parts) >= 2:
            cat = parts[0]
            ch = parts[1]
            all_categories.add(cat)
            groups_info.append({'group': group, 'category': cat, 'channel': ch})
    
    groups_df = pd.DataFrame(groups_info)
    all_categories_list = sorted(list(all_categories))
    print(f"   ✓ {len(all_categories_list)} categories found")

    # --- Step 4: Compute Reachability Paths ---
    # Structure: channel_data[channel] = {'categories': [], 'paths': {source_cat: {target_cat: path}}}
    channel_data = {}
    
    # Pre-calculate target groups by category for speed
    target_groups_by_cat = defaultdict(list)
    for node in G.nodes():
        cat = str(node).split('|')[0]
        target_groups_by_cat[cat].append(node)

    count_processed = 0
    
    for channel in top_channel_names:
        # 1. Identify categories this channel belongs to
        channel_groups = groups_df[groups_df['channel'] == channel]
        channel_categories = sorted(channel_groups['category'].unique().tolist())
        
        if not channel_categories:
            continue
            
        paths_from_category = {}
        
        # 2. For each of its categories (Source Category)
        for source_cat in channel_categories:
            source_group = f"{source_cat}|{channel}"
            
            if source_group not in G:
                continue
            
            paths_to_categories = {}
            
            # 3. Find path to every other Target Category
            for target_cat in all_categories_list:
                # Skip if it's the same category
                if target_cat == source_cat:
                    continue
                
                # Get all possible destination nodes in that category
                potential_targets = target_groups_by_cat[target_cat]
                
                shortest_path = None
                shortest_length = float('inf')
                
                # Find the single shortest path to ANY group in the target category
                # (excluding paths that just loop back to the same channel)
                for target_node in potential_targets:
                    target_ch = target_node.split('|')[1]
                    
                    if target_ch == channel:
                        continue
                        
                    try:
                        # Unweighted shortest path (fewer hops = better for Sankey)
                        path = nx.shortest_path(G, source_group, target_node)
                        
                        if len(path) < shortest_length:
                            shortest_path = path
                            shortest_length = len(path)
                            
                            # Optimization: If we find a direct neighbor (length 2), 
                            # we won't find anything shorter, so break.
                            if shortest_length <= 2:
                                break
                                
                    except nx.NetworkXNoPath:
                        continue
                
                # If a path was found, store it
                if shortest_path:
                    paths_to_categories[target_cat] = shortest_path
            
            paths_from_category[source_cat] = paths_to_categories
        
        channel_data[channel] = {
            "categories": channel_categories,
            "paths": paths_from_category
        }
        count_processed += 1
        
        if count_processed % 50 == 0:
            print(f"     Processed {count_processed}/{len(top_channel_names)} channels...")

    print(f"   ✓ Computed reachability for {len(channel_data)} channels")

    # --- Step 5: Save JSON ---
    print("\n[3/3] Saving JSON...")
    
    output_data = {
        "channelData": channel_data,
        "allCategories": all_categories_list
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4) # Indent for readability, remove if file is too large

    print("\n" + "="*80)
    print("✅ SANKEY DATA GENERATED!")
    print(f"📂 Saved to: {output_path}")
    print("="*80)