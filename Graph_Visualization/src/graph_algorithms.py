"""
Graph class and graph algorithms for the Delhi Metro network.
All algorithms implemented from scratch (no networkx shortest-path calls).
"""

import heapq
import math
from collections import defaultdict, deque

from data.data_loader import load_records, group_by_line, station_coordinates


class Graph:
    def __init__(self):
        self.adj = defaultdict(list)   # station -> list of (neighbor, weight_km)
        self.stations = set()
        self.coords = {}               # station -> (lat, lon), for A*

    def add_edge(self, u, v, weight):
        if u == v:
            return
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
        self.stations.add(u)
        self.stations.add(v)

    def build_from_csv(self):
        records = load_records()
        self.coords = station_coordinates(records)
        lines = group_by_line(records)

        for line_name, stations in lines.items():
            for i in range(len(stations) - 1):
                a, b = stations[i], stations[i + 1]
                u, v = a["canonical_name"], b["canonical_name"]
                weight = round(abs(b["distance_km"] - a["distance_km"]), 3)
                if weight == 0:
                    weight = 0.1
                self.add_edge(u, v, weight)
        return self

    def neighbors(self, station):
        return self.adj[station]

    def all_edges(self):
        """Unique edges as (u, v, weight) -- for Kruskal's MST."""
        seen = set()
        edges = []
        for u in self.adj:
            for v, w in self.adj[u]:
                key = tuple(sorted((u, v)))
                if key not in seen:
                    seen.add(key)
                    edges.append((u, v, w))
        return edges

    def connected_components(self):
        """Returns list of station-lists, one per connected component."""
        visited = set()
        components = []
        for node in self.adj:
            if node not in visited:
                comp = []
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        comp.append(n)
                        for neighbor, _ in self.adj[n]:
                            stack.append(neighbor)
                components.append(comp)
        return components

    def __len__(self):
        return len(self.stations)


def _reconstruct(parent, end):
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def bfs(graph, start, end, return_visited=False):
    """Unweighted shortest path (fewest stations, not shortest distance)."""
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    explored = 0

    while queue:
        node = queue.popleft()
        explored += 1
        if node == end:
            break
        for neighbor, _ in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    if end not in parent:
        return (None, explored) if not return_visited else (None, explored, visited)
    path = _reconstruct(parent, end)
    return (path, explored) if not return_visited else (path, explored, visited)


def dijkstra(graph, start, end, return_visited=False):
    """Shortest path by total real distance (km)."""
    dist = {start: 0}
    parent = {start: None}
    visited = set()
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for neighbor, weight in graph.neighbors(node):
            new_dist = d + weight
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    if end not in dist:
        return (None, None, len(visited)) if not return_visited else (None, None, visited)
    path = _reconstruct(parent, end)
    if return_visited:
        return path, dist[end], visited
    return path, dist[end], len(visited)


def haversine_km(coord1, coord2):
    """Great-circle distance between two (lat, lon) points, in km."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def astar(graph, start, end, return_visited=False):
    """
    Shortest path by real distance, guided by a straight-line (haversine)
    heuristic. Admissible because straight-line distance is always <= real
    track distance -- so A* still finds the optimal path, exploring fewer
    nodes than Dijkstra.
    """
    end_coord = graph.coords[end]
    g_score = {start: 0}
    parent = {start: None}
    visited = set()
    pq = [(haversine_km(graph.coords[start], end_coord), 0, start)]

    while pq:
        f, g, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            break
        for neighbor, weight in graph.neighbors(node):
            tentative_g = g + weight
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = node
                h = haversine_km(graph.coords.get(neighbor, end_coord), end_coord)
                heapq.heappush(pq, (tentative_g + h, tentative_g, neighbor))

    if end not in g_score:
        return (None, None, len(visited)) if not return_visited else (None, None, visited)
    path = _reconstruct(parent, end)
    if return_visited:
        return path, g_score[end], visited
    return path, g_score[end], len(visited)


class UnionFind:
    """Disjoint-set structure used by Kruskal's MST."""

    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def kruskal_mst(graph):
    """Minimum spanning tree -- min total track length connecting all stations."""
    edges = sorted(graph.all_edges(), key=lambda e: e[2])
    uf = UnionFind(graph.stations)
    mst_edges = []
    total_weight = 0
    for u, v, w in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_weight += w
    return mst_edges, round(total_weight, 2)


if __name__ == "__main__":
    g = Graph().build_from_csv()
    start, end = "Rithala", "Rajouri Garden"

    path, explored = bfs(g, start, end)
    print(f"BFS: {len(path)-1} stops, {explored} explored")

    path, dist, explored = dijkstra(g, start, end)
    print(f"Dijkstra: {dist:.2f} km, {explored} explored")

    path, dist, explored = astar(g, start, end)
    print(f"A*: {dist:.2f} km, {explored} explored")

    mst_edges, total = kruskal_mst(g)
    print(f"MST total length: {total} km ({len(mst_edges)} edges)")

    components = g.connected_components()
    print(f"Connected components: {len(components)}")
