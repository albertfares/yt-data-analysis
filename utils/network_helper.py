import duckdb
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import os



def format_number(n):
    """Format large numbers with commas"""
    return f"{n:,}"



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
    
    global output_file
    
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
