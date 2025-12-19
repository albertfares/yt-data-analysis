import duckdb
import time
from pathlib import Path



def format_number(n):
    """Format large numbers with commas"""
    return f"{n:,}"



def analyze_author_group_comments(input_csv, output_txt):

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