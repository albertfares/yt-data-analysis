# utils/community_helper.py
# contains helpers for both Thomas and Matteo

from __future__ import annotations

from pathlib import Path
import os
import csv
import gzip
import json
import math
import subprocess
import heapq
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Iterable, Dict, Sequence
from IPython.display import display


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError:
    StandardScaler = None
    KMeans = None


# 1) Low-level flush helpers (streaming writers)

def flush_author_to_writer(author: str,
                           counts_dict: Dict[str, int],
                           writer: csv.writer) -> None:
    """
    Write (author, channel_id, count) rows for a single author.

    Parameters
    ----------
    author : str
        Author identifier.
    counts_dict : dict
        Mapping channel_id -> num_comments for this author.
    writer : csv.writer
        CSV writer already configured with header.
    """
    if author is None:
        return

    # sort channel_id for deterministic output
    for channel_id in sorted(counts_dict):
        writer.writerow([author, channel_id, counts_dict[channel_id]])


def flush_group_count(group: str,
                      count: int,
                      writer: csv.writer) -> None:
    """
    Write one row (group, count), used for:
    - number of authors per group
    - number of channels per group
    """
    if group is None:
        return
    writer.writerow([group, count])


def flush_group_total_comments(group: str,
                               total: int,
                               writer: csv.writer) -> None:
    """
    Write one row (group, total_comments) for total comments per group.
    """
    if group is None:
        return
    writer.writerow([group, total])


def normalized_entropy_from_counts(counts: Iterable[int]) -> float:
    """
    Normalized entropy H / log(k) for a multiset of counts.

    Returns value in [0, 1]. Empty or degenerate sets -> 0.
    """
    counts = [c for c in counts if c > 0]
    if not counts:
        return 0.0
    total = sum(counts)
    k = len(counts)
    if k <= 1 or total <= 0:
        return 0.0

    H = 0.0
    for c in counts:
        p = c / total
        H -= p * math.log(p)

    return H / math.log(k)


def flush_group_features(group: str,
                         channel_counts: Dict[str, int],
                         cat_counts: Dict[str, int],
                         total_comments: int,
                         writer: csv.writer) -> None:
    """
    Write *one row* of group-level features:

    - group
    - total_comments
    - num_channels
    - fidelity        (currently == exploration; or could be 1 - exploration)
    - cat_entropy
    """
    if group is None:
        return

    num_channels = len(channel_counts)
    exploration = normalized_entropy_from_counts(channel_counts.values())
    fidelity = exploration  # or 1 - exploration 

    cat_entropy = (
        normalized_entropy_from_counts(cat_counts.values())
        if cat_counts else 0.0
    )

    writer.writerow([
        group,
        total_comments,
        num_channels,
        f"{fidelity:.6f}",
        f"{cat_entropy:.6f}",
    ])


def flush_group_category_counts(group: str,
                                cat_counts: Dict[str, int],
                                writer: csv.writer) -> None:
    """
    For a given group, write one row per category:

        group, category, num_comments_in_that_category
    """
    if group is None:
        return

    for cat, cnt in cat_counts.items():
        writer.writerow([group, cat, cnt])


# 2) Generic histogram / heatmap exporters (JSON for the web)

def export_hist(values: Iterable[float],
                bins: int,
                out_dir: str,
                out_name: str) -> str:
    """
    1D histogram → JSON, same schema as in results_community notebook.
    """
    values_arr = np.asarray(list(values))
    values_arr = values_arr[np.isfinite(values_arr)]

    counts, edges = np.histogram(values_arr, bins=bins)
    payload = {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "counts": counts.tolist(),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print("✅ wrote:", out_path)
    return out_path


def export_hist_1d(x: Iterable[float],
                   bins: int,
                   out_dir: str,
                   out_name: str) -> str:
    """
    Simpler 1D histogram → JSON (used in later sections).
    """
    x_arr = np.asarray(list(x))
    counts, edges = np.histogram(x_arr, bins=bins)

    payload = {
        "bin_left": edges[:-1].tolist(),
        "bin_right": edges[1:].tolist(),
        "counts": counts.tolist()
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print("✅ wrote", out_path, "bins:", len(counts))
    return out_path


def export_heatmap_2d(x: Iterable[float],
                      y: Iterable[float],
                      xbins: int,
                      ybins: int,
                      out_dir: str,
                      out_name: str) -> str:
    """
    2D histogram → JSON, storing only non-zero cells:

        {
          "x_edges": [...],
          "y_edges": [...],
          "cells": [[i, j, count], ...]
        }
    """
    x_arr = np.asarray(list(x))
    y_arr = np.asarray(list(y))

    H, xedges, yedges = np.histogram2d(x_arr, y_arr, bins=[xbins, ybins])

    # store only non-zero cells to keep JSON small
    cells: List[List[int]] = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            c = int(H[i, j])
            if c > 0:
                cells.append([i, j, c])  # heatmap uses [xIndex, yIndex, value]

    payload = {
        "x_edges": xedges.tolist(),
        "y_edges": yedges.tolist(),
        "cells": cells
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print("✅ wrote", out_path, "nonzero cells:", len(cells))
    return out_path


# 3) High-level pipeline helpers used by results_community.ipynb

def build_author_channelid_comment_counts(
    comments_path,
    meta_path,
    output_path,
    total_lines_comments=None,
):
    """Section 2.1: build author_channelid_comment_counts.tsv.gz."""
    meta = pd.read_feather(meta_path)

    if "display_id" not in meta.columns and "video_id" in meta.columns:
        meta = meta.rename(columns={"video_id": "display_id"})
    if "channel_id" not in meta.columns and "channel" in meta.columns:
        meta = meta.rename(columns={"channel": "channel_id"})

    meta = meta[["display_id", "channel_id"]].dropna()
    meta = meta.astype({"display_id": "string", "channel_id": "string"})

    video_to_channel = dict(zip(meta["display_id"], meta["channel_id"]))
    del meta

    with gzip.open(comments_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(output_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        if total_lines_comments is not None:
            reader = tqdm(reader, total=total_lines_comments)

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["author", "channel_id", "num_comments"])

        current_author = None
        current_counts = defaultdict(int)

        for row in reader:
            author = row.get("author")
            video_id = row.get("display_id") or row.get("video_id")

            if not author or not video_id:
                continue

            channel_id = video_to_channel.get(video_id)
            if channel_id is None:
                continue

            if current_author is None:
                current_author = author

            if author != current_author:
                flush_author_to_writer(current_author, current_counts, writer)
                current_author = author
                current_counts = defaultdict(int)

            current_counts[channel_id] += 1

        # flush last author
        flush_author_to_writer(current_author, current_counts, writer)

    print("✅ Done, wrote:", output_path)


def build_author_groups_and_group_channels(
    path,
    author_groups_path,
    groups_channelid_numc_path,
    chunksize,
    total_lines=None,
):
    """
    Section 2.2.1: build
      - author_groups.tsv.gz
      - groups_channelid_numc.tsv.gz
    from author_channelid_comment_counts.tsv.gz
    """
    f_auth = gzip.open(author_groups_path, "wt", encoding="utf-8", newline="")
    f_auth.write("author\tgroup\n")

    f_group = gzip.open(groups_channelid_numc_path, "wt", encoding="utf-8", newline="")
    f_group.write("group\tchannel_id\tnum_comments\n")

    current_author = None
    current_pairs = []
    current_channels = []
    current_counts = []

    sig_to_group = {}
    next_group_id = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype={"author": "string", "channel_id": "string"},
        chunksize=chunksize,
    )

    iterator = reader
    if total_lines is not None:
        iterator = tqdm(reader, total=total_lines // chunksize)

    for chunk in iterator:
        chunk["num_comments"] = chunk["num_comments"].astype("int64")

        for row in chunk.itertuples(index=False):
            author = row.author
            channel = row.channel_id
            count = row.num_comments

            # boundary: new author
            if current_author is not None and author != current_author:
                signature = "|".join(current_pairs)
                group = sig_to_group.get(signature)
                if group is None:
                    group = next_group_id
                    sig_to_group[signature] = group
                    next_group_id += 1
                    for ch, cnt in zip(current_channels, current_counts):
                        f_group.write(f"{group}\t{ch}\t{cnt}\n")
                f_auth.write(f"{current_author}\t{group}\n")

                current_pairs = []
                current_channels = []
                current_counts = []

            current_author = author
            current_channels.append(channel)
            current_counts.append(count)
            current_pairs.append(f"{channel}:{count}")

    # flush last author
    if current_author is not None:
        signature = "|".join(current_pairs)
        group = sig_to_group.get(signature)
        if group is None:
            group = next_group_id
            sig_to_group[signature] = group
            next_group_id += 1
            for ch, cnt in zip(current_channels, current_counts):
                f_group.write(f"{group}\t{ch}\t{cnt}\n")
        f_auth.write(f"{current_author}\t{group}\n")

    f_auth.close()
    f_group.close()
    print("✅ Done, wrote author_groups and groups_channelid_numc")


def compute_groups_num_authors(
    author_groups_path,
    out_path,
):
    """Section 2.2.2: groups_num_authors.tsv.gz."""
    groups = pd.read_csv(author_groups_path, compression="infer", sep="\t")

    result = (
        groups.groupby("group")["author"]
        .nunique()
        .reset_index(name="num_authors")
    )

    result.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print("✅ wrote:", out_path)


def compute_groups_num_channels(
    groups_channelid_numc_path,
    out_path,
    total_lines=None,
):
    """Section 2.2.3: groups_num_channels.tsv.gz."""
    with gzip.open(groups_channelid_numc_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(out_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        if total_lines is not None:
            reader = tqdm(reader, total=total_lines)

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["group", "num_channels"])

        current_group = None
        current_count = 0

        for row in reader:
            group = row.get("group")
            channel_id = row.get("channel_id")

            if not group or not channel_id:
                continue

            if current_group is None:
                current_group = group

            if group != current_group:
                flush_group_count(current_group, current_count, writer)
                current_group = group
                current_count = 0

            # each row = one distinct channel
            current_count += 1

        # flush last group
        flush_group_count(current_group, current_count, writer)

    print("✅ wrote:", out_path)


def compute_groups_total_comments(
    groups_channelid_numc_path,
    out_path,
    total_lines=None,
):
    """Section 2.2.4: groups_total_comments.tsv.gz."""
    with gzip.open(groups_channelid_numc_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(out_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        if total_lines is not None:
            reader = tqdm(reader, total=total_lines)

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["group", "total_comments"])

        current_group = None
        current_total = 0

        for row in reader:
            g = row.get("group")
            nc = row.get("num_comments")

            if not g or nc is None:
                continue

            if current_group is None:
                current_group = g

            if g != current_group:
                flush_group_total_comments(current_group, current_total, writer)
                current_group = g
                current_total = 0

            current_total += int(nc)

        flush_group_total_comments(current_group, current_total, writer)

    print("✅ wrote:", out_path)


def build_group_features(
    groups_path,
    channels_path,
    out_path,
    total_rows=None,
):
    """Section 2.4.1: groups_features.tsv.gz."""
    # channel_id -> category_cc
    channels = pd.read_csv(channels_path, sep="\t")
    if "channel" in channels.columns and "channel_id" not in channels.columns:
        channels = channels.rename(columns={"channel": "channel_id"})

    channel_to_cat = dict(zip(channels["channel_id"], channels["category_cc"]))
    del channels

    with gzip.open(groups_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(out_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        if total_rows is not None:
            reader = tqdm(reader, total=total_rows)

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["group", "total_comments", "num_channels",
                         "fidelity", "category_entropy"])

        current_group = None
        channel_counts = {}
        cat_counts = {}
        current_total_comments = 0

        for row in reader:
            g = row.get("group")
            cid = row.get("channel_id")
            nc = row.get("num_comments")

            if not g or not cid or nc is None:
                continue

            nc = int(nc)

            if current_group is None:
                current_group = g

            # boundary: new group
            if g != current_group:
                flush_group_features(
                    current_group,
                    channel_counts,
                    cat_counts,
                    current_total_comments,
                    writer,
                )
                current_group = g
                channel_counts = {}
                cat_counts = {}
                current_total_comments = 0

            current_total_comments += nc
            channel_counts[cid] = channel_counts.get(cid, 0) + nc

            cat = channel_to_cat.get(cid)
            if cat is not None:
                cat_counts[cat] = cat_counts.get(cat, 0) + nc

        # flush last group
        flush_group_features(
            current_group,
            channel_counts,
            cat_counts,
            current_total_comments,
            writer,
        )

    print("✅ wrote:", out_path)


def category_analysis_for_groups(
    groups_path,
    channels_path,
    out_path,
    total_lines=None,
):
    """Section 2.6.1: groups_category_numc.tsv.gz."""
    # 1) channel_id -> category_cc
    channels = pd.read_csv(channels_path, sep="\t")
    if "channel" in channels.columns and "channel_id" not in channels.columns:
        channels = channels.rename(columns={"channel": "channel_id"})
    channel_to_cat = dict(zip(channels["channel_id"], channels["category_cc"]))
    del channels

    # 2) Stream groups_channelid_numc.tsv.gz and aggregate per group/category
    with gzip.open(groups_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(out_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        if total_lines is not None:
            reader = tqdm(reader, total=total_lines)

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["group", "category_cc", "num_comments"])

        current_group = None
        cat_counts = defaultdict(int)

        for row in reader:
            g = row.get("group")
            cid = row.get("channel_id")
            nc = row.get("num_comments")

            if not g or not cid or nc is None:
                continue

            if current_group is None:
                current_group = g

            if g != current_group:
                flush_group_category_counts(current_group, cat_counts, writer)
                current_group = g
                cat_counts = defaultdict(int)

            cat = channel_to_cat.get(cid)
            if cat is None:
                continue

            cat_counts[cat] += int(nc)

        flush_group_category_counts(current_group, cat_counts, writer)

    print("✅ wrote:", out_path)




# 4) Group-level distributions / histograms (Section 2.3)

def export_group_histograms(
    groups_num_authors_path: str,
    groups_total_comments_path: str,
    groups_num_channels_path: str,
    out_dir: str,
) -> None:
    """
    Export basic group-level histograms used in Section 2.3:

    - group_size_hist.json              (num_authors per group)
    - groups_total_comments_hist.json   (total comments per group)
    - groups_num_channels_hist.json     (num_channels per group)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Group size (num_authors)
    n_author = pd.read_csv(groups_num_authors_path, sep="\t", compression="infer")
    export_hist(
        n_author["num_authors"].to_numpy(),
        bins=50,
        out_dir=out_dir,
        out_name="group_size_hist.json",
    )

    # 2) Total comments (total_comments) with log-spaced bins
    total_c = pd.read_csv(groups_total_comments_path, sep="\t", compression="infer")
    x = total_c["total_comments"].to_numpy()
    x = x[x > 0]
    log_bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 60)
    export_hist(
        x,
        bins=log_bins,
        out_dir=out_dir,
        out_name="groups_total_comments_hist.json",
    )

    # 3) Num channels (num_channels)
    n_channels = pd.read_csv(groups_num_channels_path, sep="\t", compression="infer")
    export_hist(
        n_channels["num_channels"].to_numpy(),
        bins=50,
        out_dir=out_dir,
        out_name="groups_num_channels_hist.json",
    )

    print("✅ Exported basic group histograms to", out_dir)






# 5) Feature-level distributions (fidelity / category entropy)


def export_feature_distributions(
    features_path: str,
    out_dir: str,
    min_tc: int = 10,
    max_tc: int = 1000,
) -> None:
    """
    Export histograms and heatmaps for the regime:

        min_tc <= total_comments < max_tc

    Outputs (all into out_dir):

    - fidelity_hist.json
    - category_entropy_hist.json
    - fidelity_vs_category_entropy_heatmap.json
    - log_num_channels_vs_category_entropy_heatmap.json
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = pd.read_csv(features_path, sep="\t", compression="infer")
    for c in ["total_comments", "num_channels", "fidelity", "category_entropy"]:
        feat[c] = pd.to_numeric(cast:=feat[c], errors="coerce")

    df = feat[
        (feat["total_comments"] >= min_tc)
        & (feat["total_comments"] < max_tc)
    ].dropna(subset=["fidelity", "category_entropy", "num_channels"])

    # 1D histograms
    export_hist_1d(
        x=df["fidelity"].to_numpy(),
        bins=np.linspace(0, 1, 80),
        out_dir=out_dir,
        out_name="fidelity_hist.json",
    )

    export_hist_1d(
        x=df["category_entropy"].to_numpy(),
        bins=np.linspace(0, 1, 80),
        out_dir=out_dir,
        out_name="category_entropy_hist.json",
    )

    # 2D heatmaps
    export_heatmap_2d(
        x=df["fidelity"].to_numpy(),
        y=df["category_entropy"].to_numpy(),
        xbins=np.linspace(0, 1, 120),
        ybins=np.linspace(0, 1, 120),
        out_dir=out_dir,
        out_name="fidelity_vs_category_entropy_heatmap.json",
    )

    # num_channels > 0 only for log10
    mask = df["num_channels"] > 0
    x = np.log10(df.loc[mask, "num_channels"].to_numpy())
    y = df.loc[mask, "category_entropy"].to_numpy()
    export_heatmap_2d(
        x=x,
        y=y,
        xbins=np.linspace(np.log10(1), np.log10(1000), 120),
        ybins=np.linspace(0, 1, 120),
        out_dir=out_dir,
        out_name="log_num_channels_vs_category_entropy_heatmap.json",
    )

    print("✅ Exported feature histograms / heatmaps to", out_dir)







# 6) K-means clustering on groups (Section 2.5.1)


def run_kmeans_on_groups(
    features_path: str,
    out_clusters_path: str,
    K: int = 10,
    min_tc: int = 10,
    max_tc: int = 1000,
) -> None:
    """
    Run K-means on group features in the regime:

        min_tc <= total_comments < max_tc

    and save cluster labels:

        group, kmeans_cluster
    """
    df = pd.read_csv(features_path, sep="\t", compression="infer")

    # numeric cleanup
    for c in ["total_comments", "num_channels", "fidelity", "category_entropy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["group", "total_comments", "num_channels", "fidelity", "category_entropy"])

    # filter regime
    df = df[(df["total_comments"] >= min_tc) & (df["total_comments"] < max_tc)].copy()
    print("Rows in regime:", len(df))

    # features for K-means
    X = df[["fidelity", "category_entropy", "num_channels"]].to_numpy()
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    df["kmeans_cluster"] = km.fit_predict(Xz)

    df[["group", "kmeans_cluster"]].to_csv(
        out_clusters_path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    print("✅ wrote K-means clusters to:", out_clusters_path)


def summarize_kmeans_cluster_people_share(
    clusters_path: str,
    authors_path: str,
    out_path: str,
) -> None:
    """
    Build kmeans_cluster_people_share.tsv:

    - n_groups, n_authors per cluster
    - share_groups, share_authors
    """
    cl = pd.read_csv(clusters_path, sep="\t", compression="infer", dtype={"group": "string"})
    au = pd.read_csv(authors_path,  sep="\t", compression="infer", dtype={"group": "string"})

    cl["kmeans_cluster"] = pd.to_numeric(cl["kmeans_cluster"], errors="coerce")
    au["num_authors"]    = pd.to_numeric(au["num_authors"], errors="coerce")

    cl = cl.dropna(subset=["group", "kmeans_cluster"])
    au = au.dropna(subset=["group", "num_authors"])

    cl["kmeans_cluster"] = cl["kmeans_cluster"].astype("int32")
    au["num_authors"]    = au["num_authors"].astype("int64")

    m = cl.merge(au, on="group", how="inner")

    summary = (
        m.groupby("kmeans_cluster", sort=True)
         .agg(
             n_groups=("group", "size"),
             n_authors=("num_authors", "sum"),
         )
         .reset_index()
         .rename(columns={"kmeans_cluster": "cluster"})
    )

    total_groups  = int(summary["n_groups"].sum())
    total_authors = int(summary["n_authors"].sum())

    summary["share_groups"]  = summary["n_groups"]  / total_groups if total_groups else 0.0
    summary["share_authors"] = summary["n_authors"] / total_authors if total_authors else 0.0

    summary.to_csv(out_path, sep="\t", index=False)
    print("✅ wrote:", out_path)
    print("Totals in regime:", total_groups, "groups /", total_authors, "authors")



# 7) Visualization helper (Section 2.5.2)

def plot_kmeans_clusters(
    features_path: str,
    clusters_path: str,
    min_tc: int = 10,
    max_tc: int = 1000,
    max_points: int = 300_000,
) -> None:
    """
    Reproduce the scatter plots of K-means clusters:

    - fidelity vs category_entropy
    - log10(total_comments) vs log10(num_channels)
    """
    feat = pd.read_csv(features_path, sep="\t", compression="infer")
    cl   = pd.read_csv(clusters_path, sep="\t", compression="infer")

    df = feat.merge(cl, on="group", how="inner")

    for c in ["total_comments", "num_channels", "fidelity", "category_entropy", "kmeans_cluster"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["fidelity", "category_entropy", "kmeans_cluster"])
    df = df[(df["total_comments"] >= min_tc) & (df["total_comments"] < max_tc)]

    print("Rows to plot:", len(df))

    plot_df = df.sample(n=min(max_points, len(df)), random_state=0)

    # 1) fidelity vs category_entropy
    plt.figure(figsize=(8, 6))
    plt.scatter(
        plot_df["fidelity"],
        plot_df["category_entropy"],
        s=1,
        alpha=0.4,
        c=plot_df["kmeans_cluster"],
    )
    plt.xlabel("Fidelity")
    plt.ylabel("Category entropy")
    plt.title("K-means clusters (fidelity vs category entropy)")
    plt.tight_layout()
    plt.show()

    # 2) log10(total_comments) vs log10(num_channels)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        np.log10(plot_df["total_comments"]),
        np.log10(plot_df["num_channels"]),
        s=1,
        alpha=0.4,
        c=plot_df["kmeans_cluster"],
    )
    plt.xlabel("log10(total_comments)")
    plt.ylabel("log10(num_channels)")
    plt.title("K-means clusters (activity vs channels)")
    plt.tight_layout()
    plt.show()




# 8) Export assets for the K-means explorer (Section 2.5.3)

def export_kmeans_explorer_assets(
    features_path: str,
    out_dir: str,
    min_tc: int = 10,
    max_tc: int = 1000,
    sample_size: int = 120_000,
    K_max: int = 10,
) -> None:
    """
    Create all JSON assets used by the React K-means explorer:

    - points.json
    - scaler.json
    - profiles_allK.json
    - labels_allK.json
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = pd.read_csv(features_path, sep="\t", compression="infer")
    for c in ["total_comments", "num_channels", "fidelity", "category_entropy"]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    df = feat.dropna(subset=["total_comments", "num_channels", "fidelity", "category_entropy"])
    df = df[(df["total_comments"] >= min_tc) & (df["total_comments"] < max_tc)].copy()
    print("Rows in regime:", len(df))

    N = min(sample_size, len(df))
    s = df.sample(n=N, random_state=0).reset_index(drop=True)

    # 1) points.json
    points = {
        "n": int(N),
        "fidelity": s["fidelity"].astype(float).tolist(),
        "catent": s["category_entropy"].astype(float).tolist(),
        "nch": s["num_channels"].astype(int).tolist(),
        "tc": s["total_comments"].astype(int).tolist(),
    }
    with open(os.path.join(out_dir, "points.json"), "w") as f:
        json.dump(points, f)
    print("✅ wrote points.json")

    # 2) K-means space (3D) + scaling
    X = s[["fidelity", "category_entropy", "num_channels"]].to_numpy()
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

    # 3) allK profiles + labels
    all_profiles = {}
    all_labels = {}

    for K in range(1, K_max + 1):
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        labels = km.fit_predict(Xz).astype(int)

        centers = scaler.inverse_transform(km.cluster_centers_)  # back to original space

        tmp = s.copy()
        tmp["cluster"] = labels

        summ = (
            tmp.groupby("cluster", sort=True)
               .agg(
                   n_points=("cluster", "size"),
                   tc_median=("total_comments", "median"),
                   nch_median=("num_channels", "median"),
                   fidelity_median=("fidelity", "median"),
                   catent_median=("category_entropy", "median"),
                   tc_mean=("total_comments", "mean"),
                   nch_mean=("num_channels", "mean"),
                   fidelity_mean=("fidelity", "mean"),
                   catent_mean=("category_entropy", "mean"),
               )
               .reset_index()
        )

        profiles = []
        for _, row in summ.iterrows():
            cid = int(row["cluster"])
            profiles.append({
                "id": cid,
                "n_points": int(row["n_points"]),
                "share": float(row["n_points"] / N),
                "tc_median": float(row["tc_median"]),
                "nch_median": float(row["nch_median"]),
                "fidelity_median": float(row["fidelity_median"]),
                "catent_median": float(row["catent_median"]),
                "tc_mean": float(row["tc_mean"]),
                "nch_mean": float(row["nch_mean"]),
                "fidelity_mean": float(row["fidelity_mean"]),
                "catent_mean": float(row["catent_mean"]),
            })

        centroids = [
            {
                "id": int(i),
                "fidelity": float(centers[i, 0]),
                "catent": float(centers[i, 1]),
                "nch": float(centers[i, 2]),
                "n_points": int((labels == i).sum()),
            }
            for i in range(K)
        ]

        all_profiles[str(K)] = profiles
        all_labels[str(K)] = {"labels": labels.tolist(), "centroids": centroids}

        print(f"✅ computed K={K}")

    with open(os.path.join(out_dir, "profiles_allK.json"), "w") as f:
        json.dump({"K_max": K_max, "profiles": all_profiles}, f)
    print("✅ wrote profiles_allK.json")

    with open(os.path.join(out_dir, "labels_allK.json"), "w") as f:
        json.dump({"K_max": K_max, "data": all_labels}, f)
    print("✅ wrote labels_allK.json")

def export_kmeans_people_share_json(
    people_share_tsv: str,
    json_out: str,
    K: int,
) -> None:
    """
    Convert kmeans_cluster_people_share.tsv into a JSON file
    consumed by the front-end.
    """
    df = pd.read_table(people_share_tsv, sep="\t")
    df["cluster"] = df["cluster"].astype(int)
    df["share_authors"] = df["share_authors"].astype(float)

    payload = {
        "K": int(K),
        "people_share": {
            str(int(row.cluster)): float(row.share_authors)
            for row in df.itertuples(index=False)
        },
    }

    with open(json_out, "w") as f:
        json.dump(payload, f)

    print("✅ wrote", json_out)




# 9) Heavy streaming / grouping.

# Small helper: show_output_summary
def show_output_summary(output_path: Path, head_n: int = 10) -> None:
    """
    Print a small summary for a gzipped TSV:
      - path
      - number of lines (excluding header)
      - head of the file
    """
    print("Path:", output_path)
    try:
        n_lines = subprocess.check_output(
            f"gzcat '{output_path}' | wc -l",
            shell=True,
            text=True,
        ).strip()
        n_lines = int(n_lines) - 1  # exclude header
        print("Number of lines (excluding header):", n_lines)
    except Exception as e:  # pragma: no cover
        print("Could not count lines:", e)
    print(f"\nHead (first {head_n} rows):")

    with gzip.open(output_path, "rt", encoding="utf-8", errors="replace") as f:
        df_head = pd.read_csv(
            f,
            sep="\t",
            nrows=head_n,
        )
    display(df_head)  # type: ignore[name-defined]


def build_author_namecc_comment_counts(
    comments_path: Path,
    meta_path: Path,
    channels_path: Path,
    output_path: Path,
    total_lines_comments: int,
) -> None:
    """
    Build author_namecc_comment_counts.tsv.gz by streaming comments
    and aggregating counts per (author, name_cc).

    """
    # 1) Build video_id -> name_cc lookup
    meta = pd.read_feather(meta_path)

    # Normalize column names if needed
    if "display_id" not in meta.columns and "video_id" in meta.columns:
        meta = meta.rename(columns={"video_id": "display_id"})
    if "channel_id" not in meta.columns and "channel" in meta.columns:
        meta = meta.rename(columns={"channel": "channel_id"})

    meta = meta[["display_id", "channel_id"]].dropna()
    meta = meta.astype({"display_id": "string", "channel_id": "string"})

    channels = pd.read_csv(
        channels_path,
        sep="\t",
        usecols=["channel", "name_cc"],
        dtype={"channel": "string", "name_cc": "string"},
    ).rename(columns={"channel": "channel_id"}).dropna(subset=["channel_id", "name_cc"])

    meta_chn = meta.merge(channels, on="channel_id", how="inner")[["display_id", "name_cc"]]
    meta_chn = meta_chn.rename(columns={"display_id": "video_id"})

    video_to_namecc = dict(zip(meta_chn["video_id"], meta_chn["name_cc"]))

    # free memory
    del meta, channels, meta_chn

    def flush_author(author: str | None, counts_dict: dict[str, int], writer: csv.writer) -> None:
        if author is None:
            return
        for name_cc, cnt in counts_dict.items():
            writer.writerow([author, name_cc, cnt])

    # 2) Stream comments and aggregate per author
    with gzip.open(comments_path, "rt", encoding="utf-8", newline="") as f_in, \
         gzip.open(output_path, "wt", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(["author", "name_cc", "num_comments"])

        current_author: str | None = None
        current_counts: dict[str, int] = defaultdict(int)

        for row in tqdm(reader, total=total_lines_comments, unit="rows"):
            author = row.get("author")
            video_id = row.get("video_id")
            if not author or not video_id:
                continue

            name_cc = video_to_namecc.get(video_id)
            if name_cc is None:
                continue

            if current_author is None:
                current_author = author

            if author != current_author:
                flush_author(current_author, current_counts, writer)
                current_author = author
                current_counts = defaultdict(int)
            current_counts[name_cc] += 1

        # flush last author
        flush_author(current_author, current_counts, writer)

    print("✅ Done, wrote:", output_path)


def ensure_author_namecc_comment_counts(
    comments_path: Path,
    meta_path: Path,
    channels_path: Path,
    output_path: Path,
    total_lines_comments: int,
) -> None:
    """
    Restart-safe wrapper: if output already exists, just print summary.
    Otherwise build it, then show summary.

    """
    if output_path.exists():
        print("File already exists, skipping this step.")
    else:
        build_author_namecc_comment_counts(
            comments_path=comments_path,
            meta_path=meta_path,
            channels_path=channels_path,
            output_path=output_path,
            total_lines_comments=total_lines_comments,
        )

    show_output_summary(output_path, head_n=10)


def build_author_groups_and_patterns(
    input_path: Path,
    out_author_groups: Path,
    out_groups_pattern: Path,
    total_data_rows: int,
) -> None:
    """
    Build:
      - author_groups.tsv.gz  (author -> group)
      - groups_channel_numc.tsv.gz  (group, channel, num_comments)

    """
    # signature (hashable canonical pattern) -> group id
    sig_to_gid: dict[bytes, int] = {}
    next_gid = 0

    def finalize_author(
        author: bytes | None,
        pairs: List[tuple[bytes, bytes]],
        f_auth,
        f_group,
    ) -> None:
        nonlocal next_gid
        if author is None or not pairs:
            return

        # canonicalize order inside author
        pairs.sort(key=lambda x: x[0])  # sort by channel bytes

        # signature is a sequence of (channel, count) bytes
        # since we want to map identical patterns to the same group id
        sig = b"|".join(ch + b":" + cnt for ch, cnt in pairs)

        gid = sig_to_gid.get(sig)
        if gid is None:
            gid = next_gid
            sig_to_gid[sig] = gid
            next_gid += 1

        # write pairs and author->group mapping
        for ch, cnt in pairs:
            f_group.write(f"{gid}\t{ch.decode('utf-8')}\t{int(cnt)}\n".encode("utf-8"))
        f_auth.write(f"{author.decode('utf-8')}\t{gid}\n".encode("utf-8"))

    with gzip.open(input_path, "rb") as fin, \
         gzip.open(out_author_groups, "wb") as f_auth, \
         gzip.open(out_groups_pattern, "wb") as f_group:

        header = fin.readline()
        cols = header.rstrip(b"\n").split(b"\t")

        col_author = cols.index(b"author")
        col_channel = cols.index(b"name_cc")
        col_count = cols.index(b"num_comments")

        # output headers
        f_auth.write(b"author\tgroup\n")
        f_group.write(b"group\tchannel\tnum_comments\n")

        current_author: bytes | None = None
        pairs: list[tuple[bytes, bytes]] = []

        for line in tqdm(fin, total=total_data_rows, desc="Grouping authors", unit="rows"):
            line = line.rstrip(b"\n")
            if not line:
                continue

            parts = line.split(b"\t")
            author = parts[col_author]
            channel = parts[col_channel]
            cnt = parts[col_count]

            if current_author is None:
                current_author = author

            if author != current_author:
                finalize_author(current_author, pairs, f_auth, f_group)
                pairs.clear()
                current_author = author

            pairs.append((channel, cnt))

        # flush last author
        finalize_author(current_author, pairs, f_auth, f_group)

    print(f"✅ Done. Unique groups: {next_gid:,}")
    print("Wrote:", out_author_groups)
    print("Wrote:", out_groups_pattern)


def ensure_author_groups_and_patterns(
    input_path: Path,
    out_author_groups: Path,
    out_groups_pattern: Path,
    total_data_rows: int,
) -> None:
    """
    Restart-safe wrapper around build_author_groups_and_patterns.
    """
    if out_author_groups.exists() and out_groups_pattern.exists():
        print("Group files already exist, skipping grouping step.")
        show_output_summary(out_author_groups, head_n=8)
        show_output_summary(out_groups_pattern, head_n=8)
        return

    print("Building author groups…")
    build_author_groups_and_patterns(
        input_path=input_path,
        out_author_groups=out_author_groups,
        out_groups_pattern=out_groups_pattern,
        total_data_rows=total_data_rows,
    )
    show_output_summary(out_author_groups, head_n=8)
    show_output_summary(out_groups_pattern, head_n=8)


# 10) Exploratory metrics: author-level & group-level (streaming)
def stream_author_activity_stats(
    path: Path,
    sample_cap: int = 2_000_000,
) -> Tuple[int, List[int], List[int]]:
    """
    Streaming stats assuming file is sorted by author.

    Computes (sampled):
      - total comments per author
      - number of channels per author

    Uses reservoir sampling to keep memory bounded.
    Returns:
      n_authors_seen, totals_sample, nchannels_sample
    """
    totals_sample: List[int] = []
    nchannels_sample: List[int] = []

    current_author: str | None = None
    total_comments = 0
    n_channels = 0
    n_authors_seen = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ia = header.index("author")
        iw = header.index("num_comments")

        for line in tqdm(f, desc="Author activity scan", unit="rows"):
            if not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")
            author = parts[ia]
            w = int(parts[iw])

            if current_author is None:
                current_author = author

            if author != current_author:
                n_authors_seen += 1

                if len(totals_sample) < sample_cap:
                    totals_sample.append(total_comments)
                    nchannels_sample.append(n_channels)
                else:
                    j = random.randint(0, n_authors_seen)
                    if j < sample_cap:
                        totals_sample[j] = total_comments
                        nchannels_sample[j] = n_channels

                current_author = author
                total_comments = 0
                n_channels = 0

            total_comments += w
            n_channels += 1

        # last author
        if current_author is not None:
            totals_sample.append(total_comments)
            nchannels_sample.append(n_channels)
            n_authors_seen += 1

    return n_authors_seen, totals_sample, nchannels_sample


def stream_group_concentration_stats(
    path: Path,
    sample_cap: int = 2_000_000,
) -> Tuple[int, List[float], List[int]]:
    """
    Streaming group stats (sampled):
      - number of channels per group
      - top-1 channel share within group

    Returns:
      n_groups_seen, top1_share_sample, nchannels_sample
    """

    current_group: int | None = None
    total = 0
    top1 = 0
    n_channels = 0
    n_groups_seen = 0

    top1_share_sample: List[float] = []
    nchannels_sample: List[int] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ig = header.index("group")
        iw = header.index("num_comments")

        for line in tqdm(f, desc="Group concentration scan", unit="rows"):
            if not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")
            g = int(parts[ig])
            w = int(parts[iw])

            if current_group is None:
                current_group = g

            if g != current_group:
                n_groups_seen += 1
                if total > 0:
                    share = top1 / total
                    if len(top1_share_sample) < sample_cap:
                        top1_share_sample.append(share)
                        nchannels_sample.append(n_channels)
                    else:
                        j = random.randint(0, n_groups_seen)
                        if j < sample_cap:
                            top1_share_sample[j] = share
                            nchannels_sample[j] = n_channels

                current_group = g
                total = 0
                top1 = 0
                n_channels = 0

            total += w
            top1 = max(top1, w)
            n_channels += 1

        # last group
        if current_group is not None and total > 0:
            top1_share_sample.append(top1 / total)
            nchannels_sample.append(n_channels)
            n_groups_seen += 1

    return n_groups_seen, top1_share_sample, nchannels_sample

def compute_group_sizes(
    author_groups_gz: Path,
    out_path: Path,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """
    Reads author_groups.tsv.gz and produces per-group author counts using chunked pandas.
    Output columns: group, num_authors
    """
    counter = defaultdict(int)
    total_rows = 0
    chunk_i = 0

    reader = pd.read_csv(
        author_groups_gz,
        sep="\t",
        usecols=["group"],
        chunksize=chunksize,
        dtype={"group": "int64"},
    )

    for chunk in tqdm(reader, desc="A1: counting groups", unit="chunk"):
        chunk_i += 1
        vc = chunk["group"].value_counts()
        for g, c in vc.items():
            counter[int(g)] += int(c)
        total_rows += len(chunk)
        if chunk_i % 50 == 0:
            print(f"[A1] progress: {total_rows:,} rows")

    df = pd.DataFrame(
        {
            "group": list(counter.keys()),
            "num_authors": list(counter.values()),
        }
    )
    df = df.sort_values("num_authors", ascending=False)

    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=3) as f:
        df.to_csv(f, sep="\t", index=False)

    print(f"[A1] Done. Processed rows: {total_rows:,}")
    print(f"[A1] Wrote: {out_path}")
    return df


def load_or_compute_group_sizes(
    author_groups_gz: Path,
    out_path: Path,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """
      - If groups_num_authors.tsv.gz exists: load it with pandas
      - Else: compute it from author_groups.tsv.gz

    Returns the resulting DataFrame in both cases.
    """
    if out_path.exists():
        print("[A1] Found cached:", out_path)
        with gzip.open(out_path, "rt", encoding="utf-8", errors="replace") as f:
            df = pd.read_csv(f, sep="\t")
        return df

    if not author_groups_gz.exists():
        raise FileNotFoundError(f"Missing {author_groups_gz}")

    return compute_group_sizes(author_groups_gz, out_path, chunksize=chunksize)


# 11) Plotting helpers and high-level plotting routines

def downsample(x: Sequence[float], max_n: int = 200_000):
    if len(x) <= max_n:
        return list(x)
    idx = random.sample(range(len(x)), max_n)
    return [x[i] for i in idx]


def add_quantile_vlines(values, label_prefix: str = "", show_legend: bool = True):
    """
    Adds vertical lines at median/p90/p99 for the given 1D values list.
    Works with both linear and log x-axes.
    """
    arr = np.asarray(values, dtype=float)
    q50 = np.percentile(arr, 50)
    q90 = np.percentile(arr, 90)
    q99 = np.percentile(arr, 99)

    plt.axvline(q50, color="gray", linestyle="--", alpha=0.7, label=f"{label_prefix}median={q50:.0f}")
    plt.axvline(q90, color="orange", linestyle="--", alpha=0.7, label=f"{label_prefix}p90={q90:.0f}")
    plt.axvline(q99, color="red", linestyle="--", alpha=0.7, label=f"{label_prefix}p99={q99:.0f}")

    if show_legend:
        plt.legend()


def plot_preliminary_scale_concentration(
    author_totals_sample,
    author_nchannels_sample,
    group_top1_share_sample,
    max_n: int = 200_000,
):
    """
    Reproduces the 'Preliminary plots: intuition on scale & concentration' cell.

    It:
      - downsamples arrays
      - plots CCDF (log-log + linear) for total comments per author
      - (re-)plots with same x/ccdf but different labels for channels per author
      - plots histogram of top-1 channel share within groups
    """

    # Downsample for plotting clarity
    author_totals_plot = downsample(author_totals_sample, max_n=max_n)
    author_nchannels_plot = downsample(author_nchannels_sample, max_n=max_n)
    group_top1_share_plot = downsample(group_top1_share_sample, max_n=max_n)

    # 1) Total comments per author (CCDF, log-log)
    x = sorted(author_totals_plot)
    ccdf = [1 - i / len(x) for i in range(len(x))]

    plt.figure(figsize=(6, 4))
    plt.loglog(x, ccdf)
    add_quantile_vlines(x)
    plt.xlabel("Total comments per author")
    plt.ylabel("CCDF (fraction of authors with ≥ x comments)")
    plt.title("Author activity is extremely heavy-tailed (log-log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(x, ccdf)
    add_quantile_vlines(x)
    plt.xlabel("Total comments per author")
    plt.ylabel("CCDF (fraction of authors with ≥ x comments)")
    plt.title("Author activity concentration (linear scale)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 2) Channels per author (CCDF, log-log)
    plt.figure(figsize=(6, 4))
    plt.loglog(x, ccdf)
    add_quantile_vlines(x)
    plt.xlabel("Channels per author")
    plt.ylabel("CCDF (fraction of authors with ≥ x channels)")
    plt.title("Multi-channel engagement exists at scale (log-log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(x, ccdf)
    add_quantile_vlines(x)
    plt.xlabel("Channels per author")
    plt.ylabel("CCDF (fraction of authors with ≥ x channels)")
    plt.title("Multi-channel engagement exists at scale (linear scale)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3) Top-1 channel share within groups (histogram)
    plt.figure(figsize=(6, 4))
    plt.hist(group_top1_share_plot, bins=50)
    add_quantile_vlines(group_top1_share_plot, label_prefix="", show_legend=True)
    plt.xlabel("Top-1 channel share within group")
    plt.ylabel("Count of groups")
    plt.title("Groups are rarely dominated by a single channel")
    plt.grid(True, alpha=0.3)
    plt.show()
# group size summary + plots + tables



def do_something(): 
    print("hello")


# Helpers 
def gini_coefficient(x):
    vals = [float(v) for v in x if v >= 0]
    if not vals:
        return float("nan")
    vals.sort()
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1) / n


def lorenz_points(x):
    vals = [float(v) for v in x if v >= 0]
    vals.sort()
    n = len(vals)
    if n == 0:
        return [0.0, 1.0], [0.0, 1.0]
    total = sum(vals)
    if total == 0:
        return [0.0, 1.0], [0.0, 1.0]
    cum = 0.0
    xs = [0.0]
    ys = [0.0]
    for i, v in enumerate(vals, start=1):
        cum += v
        xs.append(i / n)
        ys.append(cum / total)
    return xs, ys


def run_a1_summary_and_plots(
    groups_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
):
    """
      - computes summary stats + Gini
      - writes A1_summary.tsv
      - creates histograms, CCDF, rank-size, Lorenz
      - writes A1_concentration_curve.tsv.gz
      - writes A1_top_group_shares.tsv
      - writes A1_top_groups_by_size.tsv.gz
      - shows the same plots
    """

    # ---------- Summary table ----------
    sizes = groups_df["num_authors"].to_numpy()
    n_groups = len(sizes)
    n_authors = int(sizes.sum())
    maxv = int(sizes.max()) if n_groups else 0

    summary = pd.DataFrame(
        [
            {
                "n_groups": n_groups,
                "n_authors": n_authors,
                "authors_per_group_mean": (n_authors / n_groups) if n_groups else float("nan"),
                "median_group_size": float(pd.Series(sizes).median()) if n_groups else float("nan"),
                "p90_group_size": float(pd.Series(sizes).quantile(0.90)) if n_groups else float("nan"),
                "p99_group_size": float(pd.Series(sizes).quantile(0.99)) if n_groups else float("nan"),
                "max_group_size": maxv,
                "fraction_groups_singleton": float((sizes == 1).mean()) if n_groups else float("nan"),
                "fraction_authors_in_singletons": float(
                    sizes[sizes == 1].sum() / n_authors
                ) if n_authors else float("nan"),
                "gini_group_size": gini_coefficient(sizes),
            }
        ]
    )

    summary_path = tables_dir / "A1_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print("Wrote:", summary_path)

    # ---------- Plots ----------

    # Histogram (linear)
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=80)
    plt.xlabel("Group size (#authors)")
    plt.ylabel("Count of groups")
    plt.title("A1: Group size histogram (linear)")
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_group_size_hist_linear.png", dpi=200)
    plt.show()

    # Histogram (log-binned x, log y)
    if maxv >= 2:
        log_bins = [1]
        b = 1
        while b < maxv:
            b = int(max(b + 1, math.floor(b * 1.25)))
            log_bins.append(b)
        log_bins = sorted(set(log_bins))

        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=log_bins)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Group size (#authors, log)")
        plt.ylabel("Count of groups (log)")
        plt.title("A1: Group size histogram (log-binned x, log y)")
        plt.tight_layout()
        plt.savefig(plots_dir / "A1_group_size_hist_logbinned.png", dpi=200)
        plt.show()

    # CCDF (log-log)
    size_counts = Counter(sizes)
    xs = sorted(size_counts.keys())
    total = sum(size_counts.values())
    remaining = total
    ccdf_y = []
    for x in xs:
        ccdf_y.append(remaining / total)
        remaining -= size_counts[x]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ccdf_y)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Group size k")
    plt.ylabel("P(size ≥ k)")
    plt.title("A1: Group size CCDF (log-log)")
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_group_size_ccdf_loglog.png", dpi=200)
    plt.show()

    # Rank-size
    groups_sorted = groups_df.sort_values("num_authors", ascending=False).reset_index(drop=True)
    ranks = (groups_sorted.index.to_numpy() + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, groups_sorted["num_authors"].to_numpy())
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Group rank (1 = largest)")
    plt.ylabel("Group size (#authors)")
    plt.title("A1: Rank-size plot (log-log)")
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_rank_size_loglog.png", dpi=200)
    plt.show()

    # Lorenz + Gini
    gini = gini_coefficient(sizes)
    lx, ly = lorenz_points(sizes)

    plt.figure(figsize=(7, 7))
    plt.plot(lx, ly)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Cumulative share of groups")
    plt.ylabel("Cumulative share of authors")
    plt.title(f"A1: Lorenz curve (Gini={gini:.4f})")
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_lorenz_curve.png", dpi=200)
    plt.show()

    # ---------- Concentration curve table + plot ----------
    cum_auth = groups_sorted["num_authors"].cumsum()
    total_auth = float(cum_auth.iloc[-1]) if len(cum_auth) else 0.0

    fracs, shares = [], []
    for i in range(1, len(groups_sorted) + 1):
        if i in {1, 10, 100, 1000} or (i % 50000 == 0) or (i == len(groups_sorted)):
            fracs.append(i / len(groups_sorted))
            shares.append(float(cum_auth.iloc[i - 1] / total_auth) if total_auth else float("nan"))

    conc_df = pd.DataFrame({"top_fraction_groups": fracs, "share_of_authors": shares})
    conc_path = tables_dir / "A1_concentration_curve.tsv.gz"
    with gzip.open(conc_path, "wt", encoding="utf-8", compresslevel=3) as f:
        conc_df.to_csv(f, sep="\t", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(conc_df["top_fraction_groups"], conc_df["share_of_authors"])
    plt.xscale("log")
    plt.xlabel("Top fraction of groups (log)")
    plt.ylabel("Share of authors contained")
    plt.title("A1: Author concentration vs top group fraction")
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_concentration_curve.png", dpi=200)
    plt.show()

    # Top share cutoffs
    top_shares = []
    for pct in [0.0001, 0.001, 0.01, 0.05, 0.10]:
        top_n = max(1, int(pct * len(groups_sorted)))
        share = float(cum_auth.iloc[top_n - 1] / total_auth) if total_auth else float("nan")
        top_shares.append(
            {
                "top_fraction_groups": pct,
                "top_n_groups": top_n,
                "share_of_authors": share,
            }
        )

    top_shares_df = pd.DataFrame(top_shares)
    top_shares_path = tables_dir / "A1_top_group_shares.tsv"
    top_shares_df.to_csv(top_shares_path, sep="\t", index=False)
    print("Wrote:", top_shares_path)

    # Save top groups
    topN_path = tables_dir / "A1_top_groups_by_size.tsv.gz"
    with gzip.open(topN_path, "wt", encoding="utf-8", compresslevel=3) as f:
        groups_sorted.head(5000).to_csv(f, sep="\t", index=False)
    print("Wrote:", topN_path)

    # Return key DataFrames
    return {
        "summary": summary,
        "conc_df": conc_df,
        "top_shares_df": top_shares_df,
        "groups_sorted": groups_sorted,
    }


# 11) extras 

def load_totals_from_summary_or_stream(
    groups_num_authors_path: Path,
    summary_path: Path,
) -> Tuple[int, int]:
    """
    Returns (TOTAL_GROUPS, TOTAL_AUTHORS).

    Uses A1_summary.tsv if it exists, otherwise streams groups_num_authors.tsv.gz.
    """
    if summary_path.exists():
        A1_summary = pd.read_csv(summary_path, sep="\t")
        total_groups = int(A1_summary.loc[0, "n_groups"])
        total_authors = int(A1_summary.loc[0, "n_authors"])
        return total_groups, total_authors

    total_authors = 0
    total_groups = 0
    with gzip.open(groups_num_authors_path, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in tqdm(f, desc="A1 totals fallback (stream)", unit="rows"):
            if not line.strip():
                continue
            _, a = line.rstrip("\n").split("\t")
            total_authors += int(a)
            total_groups += 1

    return total_groups, total_authors


def ensure_groups_df(
    groups_df: Optional[pd.DataFrame],
    groups_num_authors_path: Path,
) -> pd.DataFrame:
    """
    If groups_df is provided and non-empty, returns it.
    Otherwise loads groups_num_authors_path and sorts by num_authors desc.
    """
    if groups_df is not None and len(groups_df) > 0:
        return groups_df

    df = pd.read_csv(groups_num_authors_path, sep="\t")
    df = df.sort_values("num_authors", ascending=False).reset_index(drop=True)
    return df


def compute_topK_coverage(
    groups_df: pd.DataFrame,
    total_authors: int,
    topk_list: List[int],
    out_path: Path,
) -> pd.DataFrame:
    """
    Computes coverage of authors by top-K largest groups and writes A1_topK_group_coverage.tsv.
    """
    cov_rows = []
    cum = 0

    for i, a in enumerate(groups_df["num_authors"].astype(int).tolist(), start=1):
        cum += a
        if i in topk_list:
            cov_rows.append(
                {
                    "top_k_groups": i,
                    "authors_covered": cum,
                    "share_of_all_authors": cum / total_authors if total_authors else float("nan"),
                }
            )
        if i >= max(topk_list):
            break

    df_cov = pd.DataFrame(cov_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cov.to_csv(out_path, sep="\t", index=False)
    print("Saved:", out_path)
    return df_cov


def enrich_top_groups(
    groups_df: pd.DataFrame,
    groups_channel_numc_path: Path,
    top_groups_path: Path,
    out_path: Path,
    topn_groups_enrich: int = 5000,
) -> pd.DataFrame:
    """
    Enriches top-N groups with channel-pattern metrics by scanning groups_channel_numc.tsv.gz.

    Writes A1_top_groups_enriched.tsv.gz and returns the enriched DataFrame.
    """
    # Decide which top groups to use
    if top_groups_path.exists():
        topG = pd.read_csv(top_groups_path, sep="\t")
        topG = topG.sort_values("num_authors", ascending=False).head(topn_groups_enrich).copy()
    else:
        topG = groups_df[["group", "num_authors"]].head(topn_groups_enrich).copy()

    topG["group"] = topG["group"].astype(int)
    topG_set = set(topG["group"].tolist())
    print("Top groups to enrich:", len(topG_set))

    top_stats: Dict[int, dict] = {}

    def finalize_group_acc(g, counts):
        if g is None or not counts:
            return
        total = int(sum(counts))
        if total <= 0:
            return
        counts_sorted = sorted(counts, reverse=True)
        top1 = int(counts_sorted[0])
        top10 = int(sum(counts_sorted[:10]))
        top_stats[g] = {
            "pattern_n_channels": int(len(counts)),
            "per_author_total_comments": int(total),
            "top1_share": float(top1 / total),
            "top10_share": float(top10 / total),
        }

    cur_g = None
    counts: List[int] = []

    with gzip.open(groups_channel_numc_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ig = header.index("group")
        iw = header.index("num_comments")

        for line in tqdm(f, desc="Enrich top groups (scan groups_channel_numc)", unit="rows"):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            g = int(parts[ig])
            w = int(parts[iw])

            if cur_g is None:
                cur_g = g

            if g != cur_g:
                if cur_g in topG_set:
                    finalize_group_acc(cur_g, counts)
                cur_g = g
                counts = []

            if g in topG_set:
                counts.append(w)

    # flush last group
    if cur_g in topG_set:
        finalize_group_acc(cur_g, counts)

    df_topstats = pd.DataFrame([{"group": g, **d} for g, d in top_stats.items()])
    df_top_enriched = topG.merge(df_topstats, on="group", how="left")

    df_top_enriched["group_total_comments"] = (
        df_top_enriched["num_authors"].astype("int64")
        * df_top_enriched["per_author_total_comments"].astype("float64")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        df_top_enriched.to_csv(f, sep="\t", index=False)
    print("Saved:", out_path)

    return df_top_enriched


def build_denoised_overlays(
    groups_df: pd.DataFrame,
    groups_channel_numc_path: Path,
    total_authors: int,
    K_THRESHOLDS: List[int],
    out_path_keep: Path,
) -> Tuple[Dict[int, List[int]], Dict[int, int], pd.DataFrame]:
    """
    Builds pass-sets by per-author total comments > K, then computes kept group sizes / authors.

    Writes A1_keep_by_user_comment_threshold.tsv and returns:
        (sizes_by_K, authors_kept_by_K, df_keep)
    """
    pass_groups = {K: set() for K in K_THRESHOLDS}

    cur_g = None
    total = 0

    with gzip.open(groups_channel_numc_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ig = header.index("group")
        iw = header.index("num_comments")

        for line in tqdm(f, desc="Build pass-sets by per-author total", unit="rows"):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            g = int(parts[ig])
            w = int(parts[iw])

            if cur_g is None:
                cur_g = g

            if g != cur_g:
                for K in K_THRESHOLDS:
                    if total > K:
                        pass_groups[K].add(cur_g)
                cur_g = g
                total = 0

            total += w

    # flush last group
    if cur_g is not None:
        for K in K_THRESHOLDS:
            if total > K:
                pass_groups[K].add(cur_g)

    for K in K_THRESHOLDS:
        print(f"K>{K}: groups passing = {len(pass_groups[K]):,}")

    # Collect group sizes for the passing groups using groups_df
    sizes_by_K: Dict[int, List[int]] = {K: [] for K in K_THRESHOLDS}
    authors_kept_by_K: Dict[int, int] = {K: 0 for K in K_THRESHOLDS}

    for g, a in zip(
        groups_df["group"].astype(int).tolist(),
        groups_df["num_authors"].astype(int).tolist(),
    ):
        for K in K_THRESHOLDS:
            if g in pass_groups[K]:
                sizes_by_K[K].append(a)
                authors_kept_by_K[K] += a

    df_keep = pd.DataFrame(
        [
            {
                "K_threshold": K,
                "n_groups_kept": len(sizes_by_K[K]),
                "n_authors_kept": int(authors_kept_by_K[K]),
                "share_authors_kept": float(authors_kept_by_K[K] / total_authors)
                if total_authors
                else float("nan"),
                "max_group_size_kept": int(max(sizes_by_K[K])) if sizes_by_K[K] else 0,
            }
            for K in K_THRESHOLDS
        ]
    )

    out_path_keep.parent.mkdir(parents=True, exist_ok=True)
    df_keep.to_csv(out_path_keep, sep="\t", index=False)
    print("Saved:", out_path_keep)

    return sizes_by_K, authors_kept_by_K, df_keep


def run_a1_extras(
    groups_num_authors_path: Path,
    groups_channel_numc_path: Path,
    summary_path: Path,
    top_groups_path: Path,
    out_a1x_dir: Path,
    groups_df: Optional[pd.DataFrame] = None,
):
    """
    Orchestrates:
      (1) coverage table
      (2) top-group enrichment
      (4) denoised overlays

    Returns a dict with:
      - coverage      (df_cov)
      - top_enriched  (df_top_enriched)
      - keep          (df_keep)
      - sizes_by_K
      - authors_kept_by_K
      - groups_df     (possibly reloaded)
      - TOTAL_GROUPS, TOTAL_AUTHORS
    """
    out_a1x_dir.mkdir(parents=True, exist_ok=True)

    TOTAL_GROUPS, TOTAL_AUTHORS = load_totals_from_summary_or_stream(
        groups_num_authors_path=groups_num_authors_path,
        summary_path=summary_path,
    )
    print("TOTAL_GROUPS =", f"{TOTAL_GROUPS:,}")
    print("TOTAL_AUTHORS=", f"{TOTAL_AUTHORS:,}")

    groups_df = ensure_groups_df(groups_df, groups_num_authors_path)

    # (1) Coverage
    TOPK_LIST = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    df_cov_path = out_a1x_dir / "A1_topK_group_coverage.tsv"
    df_cov = compute_topK_coverage(
        groups_df=groups_df,
        total_authors=TOTAL_AUTHORS,
        topk_list=TOPK_LIST,
        out_path=df_cov_path,
    )

    # (2) Top-N enrichment
    out_top_enriched = out_a1x_dir / "A1_top_groups_enriched.tsv.gz"
    df_top_enriched = enrich_top_groups(
        groups_df=groups_df,
        groups_channel_numc_path=groups_channel_numc_path,
        top_groups_path=top_groups_path,
        out_path=out_top_enriched,
        topn_groups_enrich=5000,
    )

    # (4) Denoised overlays
    K_THRESHOLDS = [10, 20, 50, 100, 200, 500, 1000]
    keep_path = out_a1x_dir / "A1_keep_by_user_comment_threshold.tsv"
    sizes_by_K, authors_kept_by_K, df_keep = build_denoised_overlays(
        groups_df=groups_df,
        groups_channel_numc_path=groups_channel_numc_path,
        total_authors=TOTAL_AUTHORS,
        K_THRESHOLDS=K_THRESHOLDS,
        out_path_keep=keep_path,
    )

    return {
        "coverage": df_cov,
        "top_enriched": df_top_enriched,
        "keep": df_keep,
        "sizes_by_K": sizes_by_K,
        "authors_kept_by_K": authors_kept_by_K,
        "groups_df": groups_df,
        "TOTAL_GROUPS": TOTAL_GROUPS,
        "TOTAL_AUTHORS": TOTAL_AUTHORS,
    }


# 12) extra plots


def _scatter(x, y, xlabel, ylabel, title, fname, out_dir: Path, logx=False, logy=False):
    plt.figure(figsize=(6.5, 4))
    plt.scatter(x, y, s=8)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=160)
    plt.show()


def _ccdf(arr):
    x = sorted(arr)
    n = len(x)
    if n == 0:
        return [], []
    y = [1 - (i / n) for i in range(n)]
    return x, y


def run_a1_extra_plots(
    out_a1x_dir: Path,
    cov_table: pd.DataFrame,
    top_enriched: pd.DataFrame,
    sizes_by_K: dict,
    authors_kept_by_K: dict,
):
    """
    Reproduces the 'A1 extra plots' cell:

      - creates PLOTS_A1X under out_a1x_dir
      - (1) coverage curve
      - (2) top-group scatter diagnostics
      - (4) denoised overlays CCDF + hist

    All PNGs are saved in out_a1x_dir / 'plots'.
    """
    plots_dir = out_a1x_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # (1) Coverage curve: top-K groups vs share of authors covered
    # -------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(cov_table["top_k_groups"], cov_table["share_of_all_authors"])
    plt.xscale("log")
    plt.xlabel("Top K largest groups (log)")
    plt.ylabel("Share of all authors covered")
    plt.title("A1: How concentrated are authors into the largest groups?")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_topK_group_coverage_curve.png", dpi=160)
    plt.show()

    # -------------------------
    # (2) Top-group diagnostics: size vs pattern metrics
    # -------------------------
    df = top_enriched.dropna().copy()

    _scatter(
        df["num_authors"],
        df["pattern_n_channels"],
        "Group size (#authors)",
        "Pattern width (#channels)",
        "A1: Do large groups correspond to simpler patterns?",
        "A1_top_groups_size_vs_pattern_channels.png",
        plots_dir,
        logx=True,
        logy=True,
    )

    _scatter(
        df["num_authors"],
        df["top1_share"],
        "Group size (#authors)",
        "Top-1 channel share (per-author pattern)",
        "A1: Are large groups dominated by 1 channel?",
        "A1_top_groups_size_vs_top1_share.png",
        plots_dir,
        logx=True,
        logy=False,
    )

    _scatter(
        df["num_authors"],
        df["per_author_total_comments"],
        "Group size (#authors)",
        "Per-author total comments (pattern sum)",
        "A1: Are large groups 'high activity' or just common patterns?",
        "A1_top_groups_size_vs_per_author_total_comments.png",
        plots_dir,
        logx=True,
        logy=True,
    )

    # -------------------------
    # (4) Denoised overlays: CCDF of group sizes for each K
    # -------------------------
    plt.figure(figsize=(7, 5))
    for K in sorted(sizes_by_K.keys()):
        sizes = sizes_by_K[K]
        x, y = _ccdf(sizes)
        if len(x) == 0:
            continue
        plt.loglog(x, y, label=f"K>{K} (authors kept={authors_kept_by_K[K]:,})")

    plt.xlabel("Group size (#authors)")
    plt.ylabel("CCDF  P(size ≥ x)")
    plt.title("A1: Group-size CCDF after filtering groups by per-author total comments")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_ccdf_group_size_by_user_comment_threshold.png", dpi=170)
    plt.show()

    # Optional: hist overlay (log-binned) for readability
    plt.figure(figsize=(7, 5))
    for K in sorted(sizes_by_K.keys()):
        sizes = sizes_by_K[K]
        if len(sizes) == 0:
            continue
        mx = max(sizes)
        if mx < 2:
            continue
        bins = [1]
        b = 1
        while b < mx:
            b = int(max(b + 1, math.floor(b * 1.35)))
            bins.append(b)
        bins = sorted(set(bins))
        plt.hist(sizes, bins=bins, histtype="step", linewidth=1.3, label=f"K>{K}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Group size (#authors, log)")
    plt.ylabel("Count of groups (log)")
    plt.title("A1: Group-size histogram after filtering groups by per-author total comments")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "A1_hist_group_size_by_user_comment_threshold.png", dpi=170)
    plt.show()

    print("Saved plots to:", plots_dir)




@dataclass
class GroupStats:
    num_channels: int = 0
    total_comments: int = 0
    max_channel_comments: int = 0
    top10_heap: list = field(default_factory=list)


def heap_push_topk(heap, x, k):
    if len(heap) < k:
        heapq.heappush(heap, x)
    else:
        if x > heap[0]:
            heapq.heapreplace(heap, x)

def logspace_bins(max_val: int, n_bins: int = 70) -> np.ndarray:
    max_val = max(2, int(max_val))
    edges = np.unique(np.floor(np.logspace(0, math.log10(max_val + 1), n_bins)).astype(int))
    edges[0] = 1
    if edges[-1] < max_val + 1:
        edges = np.append(edges, max_val + 1)
    edges = np.unique(edges)
    if edges[-1] != max_val + 1:
        edges = np.append(edges, max_val + 1)
    return edges


def stream_group_sizes_histograms(groups_num_authors_gz: Path, thresholds,
                                  max_size_for_bins=2_000_000, n_bins=70,
                                  out_dir: Path = None):
    bins = logspace_bins(max_size_for_bins, n_bins=n_bins)
    nb = len(bins) - 1
    hist_counts = {t: np.zeros(nb, dtype=np.int64) for t in thresholds}
    total_groups = {t: 0 for t in thresholds}
    total_authors = {t: 0 for t in thresholds}

    with gzip.open(groups_num_authors_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        ai = header.index("num_authors")

        for line in tqdm(f, desc="A1(noise): stream sizes", unit="groups"):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            a = int(parts[ai])

            bi = int(np.searchsorted(bins, a, side="right") - 1)
            bi = max(0, min(nb - 1, bi))

            for t in thresholds:
                if a >= t:
                    hist_counts[t][bi] += 1
                    total_groups[t] += 1
                    total_authors[t] += a

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

        for t in thresholds:
            out = out_dir / f"A1_noise_removed_hist_ge{t}.tsv"
            with open(out, "w", encoding="utf-8") as w:
                w.write("bin_left\tbin_right\tcount_groups\n")
                for i in range(nb):
                    w.write(f"{bins[i]}\t{bins[i+1]}\t{int(hist_counts[t][i])}\n")

        totals_out = out_dir / "A1_noise_removed_totals.tsv"
        with open(totals_out, "w", encoding="utf-8") as w:
            w.write("threshold_ge\tn_groups\tn_authors\tmean_group_size\n")
            for t in thresholds:
                ng = total_groups[t]
                na = total_authors[t]
                mean = (na / ng) if ng else float("nan")
                w.write(f"{t}\t{ng}\t{na}\t{mean}\n")

    return bins, hist_counts


def plot_noise_removed_distributions(bins, hist_counts, out_dir: Path):
    thresholds = sorted(hist_counts.keys())
    xs = bins[:-1].astype(float)

    out_dir.mkdir(parents=True, exist_ok=True)

    # CCDF overlay
    plt.figure(figsize=(10, 6))
    for t in thresholds:
        counts = hist_counts[t]
        total = counts.sum()
        if total == 0:
            continue
        ccdf = np.cumsum(counts[::-1])[::-1] / total
        plt.plot(xs, ccdf, label=f"size ≥ {t}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Group size k"); plt.ylabel("P(size ≥ k)")
    plt.title("A1 (noise removed): CCDF overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "A1_noise_removed_ccdf_overlay.png", dpi=200)
    plt.show()

    # log-binned histogram overlay
    plt.figure(figsize=(10, 6))
    for t in thresholds:
        counts = hist_counts[t].astype(float)
        mids = np.sqrt(bins[:-1] * bins[1:]).astype(float)
        m = counts > 0
        if m.sum() == 0:
            continue
        plt.plot(mids[m], counts[m], label=f"size ≥ {t}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Group size bin (log)")
    plt.ylabel("Count of groups in bin (log)")
    plt.title("A1 (noise removed): log-binned histogram overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "A1_noise_removed_hist_overlay.png", dpi=200)
    plt.show()


def reservoir_sample_groups(groups_num_authors_gz: Path, n_singleton=50_000,
                            n_nonsingleton=50_000, seed=42):
    rng = random.Random(seed)
    single, nons = [], []
    seen_s = 0
    seen_n = 0

    with gzip.open(groups_num_authors_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        gi = header.index("group")
        ai = header.index("num_authors")

        for line in tqdm(f, desc="Sample groups (reservoir)", unit="groups"):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            g = int(parts[gi]); a = int(parts[ai])

            if a == 1:
                seen_s += 1
                if len(single) < n_singleton:
                    single.append(g)
                else:
                    j = rng.randrange(seen_s)
                    if j < n_singleton:
                        single[j] = g
            else:
                seen_n += 1
                if len(nons) < n_nonsingleton:
                    nons.append(g)
                else:
                    j = rng.randrange(seen_n)
                    if j < n_nonsingleton:
                        nons[j] = g

    return single, nons


def scan_groups_channel_for_sample(groups_channel_numc_gz: Path, sample_groups: Dict[int, GroupStats], topk=10):
    with gzip.open(groups_channel_numc_gz, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        gi = header.index("group")
        ni = header.index("num_comments")

        for line in tqdm(f, desc="Scan groups_channel for sampled groups", unit="rows"):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            g = int(parts[gi])
            st = sample_groups.get(g)
            if st is None:
                continue

            nc = int(parts[ni])
            st.num_channels += 1
            st.total_comments += nc
            st.max_channel_comments = max(st.max_channel_comments, nc)
            heap_push_topk(st.top10_heap, nc, topk)


def summarize_and_plot_singleton_vs_non(single_ids, non_ids, stats: Dict[int, GroupStats], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def extract(ids):
        num_channels, total_comments, top1_share, top10_share = [], [], [], []
        for g in ids:
            st = stats.get(g)
            if st is None:
                continue
            num_channels.append(st.num_channels)
            total_comments.append(st.total_comments)
            if st.total_comments > 0:
                top1_share.append(st.max_channel_comments / st.total_comments)
                top10_share.append(sum(st.top10_heap) / st.total_comments)
            else:
                top1_share.append(0.0)
                top10_share.append(0.0)
        return (
            np.asarray(num_channels),
            np.asarray(total_comments),
            np.asarray(top1_share),
            np.asarray(top10_share),
        )

    s_nc, s_tc, s_t1, s_t10 = extract(single_ids)
    n_nc, n_tc, n_t1, n_t10 = extract(non_ids)

    out_tsv = out_dir / "singleton_vs_non_summary.tsv"

    def row(name, nc, tc, t1, t10):
        if len(nc) == 0:
            return {"cohort": name, "N": 0}
        return {
            "cohort": name,
            "N": int(len(nc)),
            "mean_num_channels": float(nc.mean()),
            "median_num_channels": float(np.median(nc)),
            "mean_total_comments": float(tc.mean()),
            "median_total_comments": float(np.median(tc)),
            "mean_top1_share": float(t1.mean()),
            "median_top1_share": float(np.median(t1)),
            "mean_top10_share": float(t10.mean()),
            "median_top10_share": float(np.median(t10)),
        }

    df_sum = pd.DataFrame([
        row("singleton", s_nc, s_tc, s_t1, s_t10),
        row("non_singleton", n_nc, n_tc, n_t1, n_t10),
    ])
    df_sum.to_csv(out_tsv, sep="\t", index=False)
    print("Wrote:", out_tsv)
    display(df_sum)

    def overlay_hist(x1, x2, xlabel, out_png, logx=False, logy=False, bins=60):
        plt.figure(figsize=(10, 6))
        if logx:
            x1p = x1[x1 > 0]; x2p = x2[x2 > 0]
            lo = min(x1p.min() if len(x1p) else 1, x2p.min() if len(x2p) else 1)
            hi = max(x1p.max() if len(x1p) else 10, x2p.max() if len(x2p) else 10)
            edges = np.logspace(math.log10(lo), math.log10(hi + 1), bins)
            plt.hist(x1p, bins=edges, alpha=0.6, label="singleton")
            plt.hist(x2p, bins=edges, alpha=0.6, label="non-singleton")
            plt.xscale("log")
        else:
            plt.hist(x1, bins=bins, alpha=0.6, label="singleton")
            plt.hist(x2, bins=bins, alpha=0.6, label="non-singleton")
        if logy:
            plt.yscale("log")
        plt.xlabel(xlabel); plt.ylabel("count")
        plt.title(f"singleton_vs_non: {xlabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.show()

    overlay_hist(s_nc, n_nc, "num_channels per group", out_dir/"singleton_vs_non_num_channels_hist.png", logx=True, logy=True)
    overlay_hist(s_tc, n_tc, "total_comments per group", out_dir/"singleton_vs_non_total_comments_hist.png", logx=True, logy=True)
    overlay_hist(s_t1, n_t1, "top1_share (dominance)", out_dir/"singleton_vs_non_top1_share_hist.png")
    overlay_hist(s_t10, n_t10, "top10_share", out_dir/"singleton_vs_non_top10_share_hist.png")

def run_a1_noise_and_singleton_diagnostics(
    groups_num_authors_path: Path,
    groups_channel_numc_path: Path,
    part1_dir: Path,
    part2_dir: Path,
):
    thresholds = [1, 2, 5, 10, 20, 50, 100, 1000]

    bins, hist_counts = stream_group_sizes_histograms(
        groups_num_authors_path,
        thresholds,
        out_dir=part1_dir,
    )
    plot_noise_removed_distributions(bins, hist_counts, out_dir=part1_dir)

    single_ids, non_ids = reservoir_sample_groups(
        groups_num_authors_path,
        n_singleton=50_000,
        n_nonsingleton=50_000,
        seed=42,
    )

    stats = {g: GroupStats() for g in single_ids}
    stats.update({g: GroupStats() for g in non_ids})

    scan_groups_channel_for_sample(groups_channel_numc_path, stats, topk=10)

    summarize_and_plot_singleton_vs_non(single_ids, non_ids, stats, out_dir=part2_dir)


