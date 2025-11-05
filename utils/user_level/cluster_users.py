def user_features(g):
    total_comments = g['num_comments'].sum()
    n_channels = g['name_cc'].nunique()
    weights = g['num_comments'] / total_comments

    f = {}
    f['total_comments'] = total_comments
    f['n_channels'] = n_channels
    f['avg_comments_per_channel'] = total_comments / n_channels if n_channels else 0

    cat_counts = g.groupby('category_cc')['num_comments'].sum()
    p = cat_counts / cat_counts.sum()
    f['category_entropy'] = -(p * np.log(p)).sum()
    f['num_unique_categories'] = len(p)

    for c in top_categories:
        f[f'cat_{c}'] = p.get(c, 0)

    f['mean_log_subs'] = np.average(g['log_subs'], weights=weights)
    f['std_log_subs'] = np.sqrt(np.cov(g['log_subs'], aweights=weights)) if len(g) > 1 else 0

    if g['join_year'].notna().any():
        yrs = g['join_year'].dropna()
        w = weights.loc[yrs.index]
        f['mean_join_year'] = np.average(yrs, weights=w)
        f['std_join_year'] = np.sqrt(np.cov(yrs, aweights=w)) if len(yrs) > 1 else 0
    else:
        f['mean_join_year'] = f['std_join_year'] = np.nan

    for col in ['delta_videos', 'activity']:
        if col in g:
            f[f'mean_{col}'] = np.average(g[col].fillna(0), weights=weights)
        else:
            f[f'mean_{col}'] = np.nan

    for col in ['delta_subs', 'delta_views']:
        if col in g:
            f[f'mean_{col}'] = np.average(g[col].fillna(0), weights=weights)
        else:
            f[f'mean_{col}'] = np.nan

    return pd.Series(f)




