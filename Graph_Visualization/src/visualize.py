"""
Visualization functions for the Delhi Metro graph.
Requires matplotlib.
"""

import matplotlib.pyplot as plt

from src.graph_algorithms import dijkstra, astar


def _draw_base_network(ax, graph, edge_color="lightgray", node_color="lightgray"):
    drawn = set()
    for u in graph.adj:
        for v, w in graph.adj[u]:
            key = tuple(sorted((u, v)))
            if key not in drawn and u in graph.coords and v in graph.coords:
                drawn.add(key)
                lat1, lon1 = graph.coords[u]
                lat2, lon2 = graph.coords[v]
                ax.plot([lon1, lon2], [lat1, lat2], color=edge_color, linewidth=0.6, zorder=1)

    lats = [graph.coords[s][0] for s in graph.coords]
    lons = [graph.coords[s][1] for s in graph.coords]
    ax.scatter(lons, lats, s=6, color=node_color, zorder=2)


def plot_network(graph, title="Delhi Metro Network"):
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_base_network(ax, graph, edge_color="lightgray", node_color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.axis("equal")
    plt.show()


def plot_path(graph, path, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_base_network(ax, graph)

    if path:
        path_lats = [graph.coords[s][0] for s in path]
        path_lons = [graph.coords[s][1] for s in path]
        ax.plot(path_lons, path_lats, color="crimson", linewidth=2.5, zorder=3)
        ax.scatter(path_lons, path_lats, s=40, color="crimson", zorder=4)
        ax.scatter([path_lons[0]], [path_lats[0]], s=120, color="green", zorder=5, label="Start")
        ax.scatter([path_lons[-1]], [path_lats[-1]], s=120, color="black", zorder=5, label="End")

    ax.set_title(title)
    ax.legend()
    plt.axis("equal")
    plt.show()


def plot_algorithm_comparison(graph, start, end):
    """
    Side-by-side Dijkstra vs A* comparison, showing every node each
    algorithm explored during search (not just the final path) -- this is
    what visually demonstrates why A* is faster.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    results = [
        ("Dijkstra", dijkstra(graph, start, end, return_visited=True)),
        ("A*", astar(graph, start, end, return_visited=True)),
    ]

    for ax, (name, (path, dist, visited)) in zip(axes, results):
        _draw_base_network(ax, graph, edge_color="whitesmoke", node_color="whitesmoke")

        exp_lats = [graph.coords[s][0] for s in visited if s in graph.coords]
        exp_lons = [graph.coords[s][1] for s in visited if s in graph.coords]
        ax.scatter(exp_lons, exp_lats, s=25, color="orange", alpha=0.6, zorder=2, label="Explored")

        path_lats = [graph.coords[s][0] for s in path]
        path_lons = [graph.coords[s][1] for s in path]
        ax.plot(path_lons, path_lats, color="crimson", linewidth=2.5, zorder=3)
        ax.scatter([path_lons[0]], [path_lats[0]], s=120, color="green", zorder=5, label="Start")
        ax.scatter([path_lons[-1]], [path_lats[-1]], s=120, color="black", zorder=5, label="End")

        ax.set_title(f"{name}: {dist:.2f} km, {len(visited)} nodes explored")
        ax.legend(loc="lower right")
        ax.axis("equal")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.graph_algorithms import Graph

    g = Graph().build_from_csv()
    plot_network(g)
    plot_algorithm_comparison(g, "Rithala", "Rajouri Garden")
