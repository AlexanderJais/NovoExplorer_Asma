"""Interactive PPI network visualization using NetworkX + Plotly."""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def build_ppi_network(
    ppi_df: pd.DataFrame,
    src_col: str = "source_name",
    tgt_col: str = "target_name",
    score_col: str = "score",
    fc_map: Optional[Dict[str, float]] = None,
    layout: str = "spring",
    up_color: str = "#D55E00",
    down_color: str = "#0072B2",
    ns_color: str = "#BBBBBB",
) -> go.Figure:
    """Build an interactive Plotly network graph from PPI interaction data.

    Parameters
    ----------
    ppi_df : DataFrame with at least src_col and tgt_col columns.
    fc_map : Optional mapping gene_name (upper) -> log2FC for colouring nodes.
    layout : NetworkX layout algorithm ('spring', 'kamada_kawai', 'circular').
    """
    G = nx.Graph()

    for _, row in ppi_df.iterrows():
        src = str(row[src_col])
        tgt = str(row[tgt_col])
        score = float(row[score_col]) if score_col in row.index else 0.5
        G.add_edge(src, tgt, score=score)

    if len(G) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No interactions to display",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Compute layout
    if layout == "kamada_kawai" and len(G) <= 500:
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=1.5 / (len(G) ** 0.5), iterations=80, seed=42)

    # --- Edges ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#CCCCCC"),
        hoverinfo="none",
    )

    # --- Nodes ---
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    degrees = dict(G.degree())
    node_names = list(G.nodes())
    node_degrees = [degrees[n] for n in node_names]

    # Size: scale by degree
    max_deg = max(node_degrees) if node_degrees else 1
    node_sizes = [max(6, 6 + 20 * (d / max_deg)) for d in node_degrees]

    # Color by log2FC if available, otherwise by degree
    if fc_map:
        node_colors = []
        for n in node_names:
            fc = fc_map.get(n.upper())
            if fc is not None and fc > 0:
                node_colors.append(up_color)
            elif fc is not None and fc < 0:
                node_colors.append(down_color)
            else:
                node_colors.append(ns_color)
        hover_texts = []
        for n in node_names:
            fc = fc_map.get(n.upper())
            fc_str = f"{fc:.3f}" if fc is not None else "N/A"
            hover_texts.append(
                f"<b>{n}</b><br>Connections: {degrees[n]}<br>log2FC: {fc_str}"
            )
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=0.8, color="#333333")),
            text=[n if degrees[n] >= max(3, max_deg * 0.3) else "" for n in node_names],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_texts,
            hoverinfo="text",
        )
    else:
        hover_texts = [
            f"<b>{n}</b><br>Connections: {degrees[n]}" for n in node_names
        ]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_degrees,
                colorscale="YlOrRd",
                colorbar=dict(title="Connections", thickness=15),
                line=dict(width=0.8, color="#333333"),
            ),
            text=[n if degrees[n] >= max(3, max_deg * 0.3) else "" for n in node_names],
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hover_texts,
            hoverinfo="text",
        )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=f"PPI Network — {len(G.nodes())} genes, {len(G.edges())} interactions",
            font=dict(size=16),
        ),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        height=700,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )

    return fig
