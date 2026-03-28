"""Interactive PPI network visualization using NetworkX + Plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    import networkx as nx


def _nx():
    """Lazy-import networkx so the app doesn't crash if it's not installed."""
    try:
        import networkx as _nx_mod
        return _nx_mod
    except ImportError:
        raise ImportError(
            "networkx is required for PPI network visualisation. "
            "Install it with:  pip install networkx"
        )


def _build_graph(
    ppi_df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    score_col: str,
) -> "nx.Graph":
    """Build a NetworkX graph from a PPI DataFrame."""
    nx = _nx()
    G = nx.Graph()
    has_score = score_col in ppi_df.columns
    for _, row in ppi_df.iterrows():
        src = str(row[src_col])
        tgt = str(row[tgt_col])
        score = float(row[score_col]) if has_score else 0.5
        G.add_edge(src, tgt, score=score)
    return G


def _compute_layout(G: "nx.Graph", layout: str) -> dict:
    """Compute node positions using the requested layout algorithm."""
    nx = _nx()
    if layout == "kamada_kawai" and len(G) <= 500:
        return nx.kamada_kawai_layout(G)
    if layout == "circular":
        return nx.circular_layout(G)
    return nx.spring_layout(G, k=1.5 / (len(G) ** 0.5), iterations=80, seed=42)


def _empty_figure(message: str, height: int = 400) -> go.Figure:
    """Return a blank figure with a message title."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=height,
    )
    return fig


def _label_threshold(degrees: dict, node_names: list) -> int:
    """Determine degree threshold above which labels are shown.

    For small or sparse networks, show all labels. For larger networks,
    show only hubs.
    """
    if len(node_names) <= 30:
        return 0  # show all labels
    max_deg = max(degrees[n] for n in node_names) if node_names else 1
    return max(3, int(max_deg * 0.3))


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
    G = _build_graph(ppi_df, src_col, tgt_col, score_col)

    if len(G) == 0:
        return _empty_figure("No interactions to display")

    pos = _compute_layout(G, layout)

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

    label_thresh = _label_threshold(degrees, node_names)

    # Color by log2FC if available, otherwise by degree
    if fc_map:
        node_colors = []
        hover_texts = []
        for n in node_names:
            fc = fc_map.get(n.upper())
            if fc is not None and fc > 0:
                node_colors.append(up_color)
            elif fc is not None and fc < 0:
                node_colors.append(down_color)
            else:
                node_colors.append(ns_color)
            fc_str = f"{fc:.3f}" if fc is not None else "N/A"
            hover_texts.append(
                f"<b>{n}</b><br>Connections: {degrees[n]}<br>log2FC: {fc_str}"
            )
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=0.8, color="#333333")),
            text=[n if degrees[n] >= label_thresh else "" for n in node_names],
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
            text=[n if degrees[n] >= label_thresh else "" for n in node_names],
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


def build_ego_network(
    ppi_df: pd.DataFrame,
    gene: str,
    src_col: str = "source_name",
    tgt_col: str = "target_name",
    score_col: str = "score",
    radius: int = 1,
    fc_map: Optional[Dict[str, float]] = None,
    layout: str = "spring",
    up_color: str = "#D55E00",
    down_color: str = "#0072B2",
    ns_color: str = "#BBBBBB",
    center_color: str = "#FFD700",
) -> go.Figure:
    """Build a neighborhood (ego) network centered on a specific gene.

    Parameters
    ----------
    gene : The gene to center the network on.
    radius : How many hops from the center gene to include (1 = direct neighbors,
             2 = neighbors of neighbors).
    center_color : Highlight colour for the query gene node.
    """
    G_full = _build_graph(ppi_df, src_col, tgt_col, score_col)

    # Find the gene node (case-insensitive)
    gene_upper = gene.upper()
    node_match = None
    for n in G_full.nodes():
        if str(n).upper() == gene_upper:
            node_match = n
            break

    if node_match is None:
        return _empty_figure(f"Gene '{gene}' not found in the network")

    # Extract ego graph (subgraph within `radius` hops)
    nx = _nx()
    ego = nx.ego_graph(G_full, node_match, radius=radius)

    if len(ego) == 0:
        return _empty_figure(f"No interactions found for '{gene}'")

    # Layout — use kamada_kawai for small ego networks for cleaner look
    if len(ego) <= 200:
        pos = nx.kamada_kawai_layout(ego)
    elif layout == "circular":
        pos = nx.circular_layout(ego)
    else:
        pos = nx.spring_layout(ego, k=2.0 / (len(ego) ** 0.5), iterations=100, seed=42)

    # --- Edges ---
    edge_x, edge_y = [], []
    for u, v in ego.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.0, color="#CCCCCC"),
        hoverinfo="none",
    )

    # --- Nodes ---
    node_names = list(ego.nodes())
    node_x = [pos[n][0] for n in node_names]
    node_y = [pos[n][1] for n in node_names]
    degrees = dict(ego.degree())
    node_degrees = [degrees[n] for n in node_names]
    max_deg = max(node_degrees) if node_degrees else 1
    node_sizes = [max(8, 8 + 22 * (d / max_deg)) for d in node_degrees]

    # Color: center gene gets center_color, others by log2FC or degree
    node_colors = []
    hover_texts = []
    for i, n in enumerate(node_names):
        is_center = str(n).upper() == gene_upper
        if is_center:
            node_colors.append(center_color)
            node_sizes[i] = max(node_sizes[i], 28)
        elif fc_map:
            fc = fc_map.get(str(n).upper())
            if fc is not None and fc > 0:
                node_colors.append(up_color)
            elif fc is not None and fc < 0:
                node_colors.append(down_color)
            else:
                node_colors.append(ns_color)
        else:
            node_colors.append(ns_color)

        fc_str = ""
        if fc_map:
            fc = fc_map.get(str(n).upper())
            fc_str = f"<br>log2FC: {fc:.3f}" if fc is not None else "<br>log2FC: N/A"
        role = " (query)" if is_center else ""
        hover_texts.append(
            f"<b>{n}</b>{role}<br>Connections: {degrees[n]}{fc_str}"
        )

    # Show labels on all nodes for ego networks (they're typically small)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.0, color="#333333"),
        ),
        text=node_names,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=hover_texts,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=(
                f"Neighborhood of <b>{node_match}</b> — "
                f"{len(ego.nodes())} genes, {len(ego.edges())} interactions "
                f"(radius {radius})"
            ),
            font=dict(size=16),
        ),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
    )

    return fig
