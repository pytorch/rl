# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Callable

from tensordict import NestedKey


def _plot_plotly_tree(
    tree: Tree, make_labels: Callable[[Tree], str] | None = None  # noqa: F821
):
    import plotly.graph_objects as go
    from igraph import Graph

    if make_labels is None:

        def make_labels(tree, path, *args, **kwargs):
            return str((tree.node_id, tree.hash))

    nr_vertices = tree.num_vertices()
    vertices = tree.vertices(key_type="path")

    v_label = [make_labels(subtree, path) for path, subtree in vertices.items()]
    G = Graph(nr_vertices, tree.edges())

    layout = G.layout_sugiyama(range(nr_vertices))

    position = {k: layout[k] for k in range(nr_vertices)}
    # Y = [layout[k][1] for k in range(nr_vertices)]
    # M = max(Y)

    # es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    # Yn = [2 * M - position[k][1] for k in range(L)]
    Yn = [position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        # Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
        Ye += [position[edge[0]][1], position[edge[1]][1], None]

    labels = v_label
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line={"color": "rgb(210,210,210)", "width": 5},
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers+text",
            name="bla",
            marker={
                "symbol": "circle-dot",
                "size": 40,
                "color": "#6175c1",  # '#DB4551',
                "line": {"color": "rgb(50,50,50)", "width": 1},
            },
            text=labels,
            hoverinfo="text",
            textposition="middle right",
            opacity=0.8,
        )
    )
    fig.show()


def _plot_plotly_box(tree: Tree, info: list[NestedKey] = None):  # noqa: F821
    import plotly.graph_objects as go

    if info is None:
        info = ["hash", ("next", "reward")]

    parents = [""]
    labels = [tree._label(info, tree, root=True)]

    _tree = tree

    def extend(tree: Tree, parent):  # noqa: F821
        children = tree.subtree
        if children is None:
            return
        for child in children:
            labels.append(tree._label(info, child))
            parents.append(parent)
            extend(child, labels[-1])

    extend(_tree, labels[-1])
    fig = go.Figure(go.Treemap(labels=labels, parents=parents))
    fig.show()
