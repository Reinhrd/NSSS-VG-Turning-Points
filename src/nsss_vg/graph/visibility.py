"""Visibility graph construction.

The pipeline defaults to Horizontal Visibility Graph (HVG), because HVG is
more robust to noise and simpler to audit for no-look-ahead behavior.

For each date t, the graph is built from a trailing window ending at t only:
[t-window+1, ..., t].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def horizontal_visibility_graph(values) -> nx.Graph:
    """Build a Horizontal Visibility Graph from a numeric sequence."""
    y = np.asarray(values, dtype=float)
    n = len(y)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1:
                graph.add_edge(i, j)
                continue

            if np.all(y[i + 1 : j] < min(y[i], y[j])):
                graph.add_edge(i, j)

    return graph


def natural_visibility_graph(values) -> nx.Graph:
    """Build a Natural Visibility Graph from a numeric sequence."""
    y = np.asarray(values, dtype=float)
    n = len(y)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1:
                graph.add_edge(i, j)
                continue

            visible = True
            for k in range(i + 1, j):
                interpolated = y[i] + (y[j] - y[i]) * (k - i) / (j - i)
                if y[k] >= interpolated:
                    visible = False
                    break

            if visible:
                graph.add_edge(i, j)

    return graph


def build_visibility_graph(values, mode: str = "horizontal") -> nx.Graph:
    """Build either HVG or NVG."""
    if mode == "horizontal":
        return horizontal_visibility_graph(values)
    if mode == "natural":
        return natural_visibility_graph(values)
    raise ValueError("mode must be either 'horizontal' or 'natural'")


def graph_feature_dict(values, mode: str = "horizontal") -> dict:
    """Extract graph-level and latest-node features from a trailing window."""
    y = np.asarray(values, dtype=float)
    graph = build_visibility_graph(y, mode=mode)

    n = len(y)
    last = n - 1
    degree = dict(graph.degree())

    if not degree:
        return {}

    prefix = mode
    max_degree = max(degree.values())
    avg_degree = float(np.mean(list(degree.values())))

    try:
        betweenness_last = nx.betweenness_centrality(graph).get(last, np.nan)
    except Exception:
        betweenness_last = np.nan

    try:
        closeness_last = nx.closeness_centrality(graph).get(last, np.nan)
    except Exception:
        closeness_last = np.nan

    try:
        clustering_last = nx.clustering(graph, last)
        avg_clustering = nx.average_clustering(graph)
    except Exception:
        clustering_last = np.nan
        avg_clustering = np.nan

    return {
        f"{prefix}_last_degree": degree.get(last, 0),
        f"{prefix}_last_degree_norm": degree.get(last, 0) / (n - 1) if n > 1 else np.nan,
        f"{prefix}_max_degree": max_degree,
        f"{prefix}_avg_degree": avg_degree,
        f"{prefix}_density": nx.density(graph) if n > 1 else np.nan,
        f"{prefix}_last_betweenness": betweenness_last,
        f"{prefix}_last_closeness": closeness_last,
        f"{prefix}_last_clustering": clustering_last,
        f"{prefix}_avg_clustering": avg_clustering,
        f"{prefix}_edges": graph.number_of_edges(),
    }


def add_visibility_features(
    df: pd.DataFrame,
    window_size: int = 60,
    price_col: str = "Close",
    mode: str = "horizontal",
) -> pd.DataFrame:
    """Add trailing-window visibility graph features."""
    records = []

    for i in range(len(df)):
        if i < window_size - 1:
            records.append({})
            continue

        window = df[price_col].iloc[i - window_size + 1 : i + 1].values
        records.append(graph_feature_dict(window, mode=mode))

    features = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
