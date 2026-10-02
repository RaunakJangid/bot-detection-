"""Network topologies: Internet Topology Zoo graphs and synthetic IoT (random geometric) graphs.

Every topology is a connected nx.Graph on nodes 0..n-1 with a per-edge `delay` in milliseconds.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Original site first, then the Internet Archive's byte-for-byte copy ("id_" = raw file) of the same URL,
# because topology-zoo.org is often unreachable. (The authors' GitHub "sources" are yEd drawings with
# pixel x/y, not the published Latitude/Longitude files, so they are not used.)
ZOO_URLS = (
    "http://www.topology-zoo.org/files/{name}.graphml",
    "https://web.archive.org/web/2024id_/http://www.topology-zoo.org/files/{name}.graphml",
)
KM_PER_MS = 200.0  # light in fibre: ~2e8 m/s


@dataclass
class Topology:
    name: str
    G: nx.Graph
    coords: np.ndarray  # (n, 2): (lat, lon) for Zoo, (x, y) for synthetic

    @property
    def n(self) -> int:
        return self.G.number_of_nodes()


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _largest_component_relabelled(G: nx.Graph) -> nx.Graph:
    comp = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(comp).copy(), ordering="sorted")


def _valid_graphml(path: Path) -> bool:
    """Parses, and has geographic coordinates (needed for link propagation delay)."""
    try:
        G = nx.read_graphml(path)
        return any("Latitude" in d and "Longitude" in d for _, d in G.nodes(data=True))
    except Exception:
        return False


def download_zoo(name: str, cache_dir: Path, timeout: float = 20.0, retries: int = 2) -> Path:
    """Fetch <name>.graphml once. Writes to a temp file and renames only after it parses, so an
    interrupted download never leaves a truncated file that later runs would trust."""
    path = cache_dir / f"{name}.graphml"
    if path.exists() and _valid_graphml(path):
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".graphml.part")
    errors = []
    for url in (u.format(name=name) for u in ZOO_URLS):
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    tmp.write_bytes(resp.read())
                if _valid_graphml(tmp):
                    tmp.replace(path)
                    return path
                errors.append(f"{url}: not a valid graphml")
                break
            except OSError as exc:
                errors.append(f"{url} (attempt {attempt}): {exc}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download Topology Zoo graph {name!r}:\n  " + "\n  ".join(errors))


def load_zoo(name: str, cache_dir: Path) -> Topology:
    raw = nx.Graph(nx.read_graphml(download_zoo(name, cache_dir)))  # collapse parallel links
    located = [v for v, d in raw.nodes(data=True) if "Latitude" in d and "Longitude" in d]
    G = raw.subgraph(located).copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    G = _largest_component_relabelled(G)
    coords = np.array([[float(G.nodes[v]["Latitude"]), float(G.nodes[v]["Longitude"])] for v in G.nodes])
    for u, v in G.edges:
        km = haversine_km(*coords[u], *coords[v])
        G.edges[u, v]["delay"] = max(km / KM_PER_MS, 0.01)
    return Topology(name, G, coords)


def synthetic_iot(n: int, seed: int | None = None, avg_degree: float = 5.0) -> Topology:
    """Random geometric graph in the unit square, made connected by bridging nearest component pairs.
    Edge delay scales with length into [1, 10] ms."""
    seed = n if seed is None else seed
    r = math.sqrt(avg_degree / (math.pi * n))
    G = nx.random_geometric_graph(n, r, seed=seed)
    pos = np.array([G.nodes[v]["pos"] for v in range(n)])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    main = set(comps[0])
    for comp in comps[1:]:
        a_nodes, b_nodes = np.array(sorted(main)), np.array(sorted(comp))
        d = np.linalg.norm(pos[a_nodes][:, None] - pos[b_nodes][None], axis=2)
        i, j = np.unravel_index(d.argmin(), d.shape)
        G.add_edge(int(a_nodes[i]), int(b_nodes[j]))
        main |= comp
    lengths = {e: float(np.linalg.norm(pos[e[0]] - pos[e[1]])) for e in G.edges}
    longest = max(lengths.values())
    for e, length in lengths.items():
        G.edges[e]["delay"] = 1.0 + 9.0 * length / longest
    for v in G.nodes:
        G.nodes[v].pop("pos", None)
    return Topology(f"syn{n}", G, pos)


def get_topology(name: str, cache_dir: Path) -> Topology:
    if name.startswith("syn"):
        return synthetic_iot(int(name[3:]))
    return load_zoo(name, cache_dir)


def latency_matrix(G: nx.Graph, removed_nodes: set[int] | None = None) -> np.ndarray:
    """All-pairs shortest-path latency (ms); inf where disconnected."""
    n = G.number_of_nodes()
    rows, cols, w = [], [], []
    for u, v, d in G.edges(data="delay"):
        if removed_nodes and (u in removed_nodes or v in removed_nodes):
            continue
        rows += [u, v]; cols += [v, u]; w += [d, d]
    A = csr_matrix((w, (rows, cols)), shape=(n, n))
    return dijkstra(A, directed=False)
