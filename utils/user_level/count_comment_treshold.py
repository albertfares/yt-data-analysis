#!/usr/bin/env python3
import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python count_comment_treshold.py <N>")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Error: N must be an integer.")
        sys.exit(1)

    df = pd.read_csv("../../models/user_level/user_total_comments_small.csv")

    count = (df["total_comments"] <= N).sum()
    total_users = len(df)

    print(f"{count:,} users have commented {N} or fewer times "
          f"({count/total_users:.1%} of all users).")

if __name__ == "__main__":
    main()
