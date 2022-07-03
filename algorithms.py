import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import networkx as nx
from networkx.algorithms import tree
from celluloid import Camera

l = 50


def get_lin(u1, u2, v1, v2):
    flag1 = False
    flag2 = False
    if u1 > u2:
        u1, u2 = u2, u1
        flag1 = True
    if v1 > v2:
        v1, v2 = v2, v1
        flag2 = True
    if 2 * np.pi - u2 + u1 < u2 - u1:
        u1, u2 = u2, u1 + 2 * np.pi
        flag1 = not flag1
    if 2 * np.pi - v2 + v1 < v2 - v1:
        v1, v2 = v2, v1 + 2 * np.pi
        flag2 = not flag2
    u = np.linspace(u1, u2, l)
    v = np.linspace(v1, v2, l)
    if flag1 and not flag2:
        u = np.flip(u)
    if not flag1 and flag2:
        v = np.flip(v)
    return u, v


def rotate_x(x, y, alpha):
    return x * np.cos(alpha) - y * np.sin(alpha)


def rotate_y(x, y, alpha):
    return x * np.sin(alpha) + y * np.cos(alpha)


class TSP:
    values: np.ndarray
    data: Dict[str, Dict[str, Any]]
    n: int
    mul: int
    graph: Any
    three_dim: bool
    circular: bool
    phi: np.ndarray
    psi: np.ndarray

    def __init__(self, num: int, circular=False, three_dim=False):
        self.n = num
        self.mul = 100
        self.values = np.random.rand(2, self.n) * self.mul
        self.three_dim = three_dim
        self.circular = circular
        if circular and not three_dim:
            # nums = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / self.n)
            nums = np.random.rand(self.n) * 2 * np.pi
            self.values = np.array([np.cos(nums), np.sin(nums)]) * self.mul
        if three_dim and not circular:
            self.values = np.random.rand(3, self.n) * self.mul
            self.values[0] *= np.sign(np.sin(np.random.rand(self.n) * self.mul))
            self.values[1] *= np.sign(np.sin(np.random.rand(self.n) * self.mul))
        if three_dim and circular:
            phi = np.random.rand(self.n) * 2 * np.pi
            psi = np.random.rand(self.n) * 2 * np.pi
            self.phi = phi
            self.psi = psi
            self.values = np.array([np.sin(phi) * np.cos(psi),
                                    np.sin(phi) * np.sin(psi),
                                    np.cos(phi)]) * self.mul
        self.data = {
            'Random': {},
            'Greedy': {},
            '2-Approximation': {},
            'Minimal Spanning Tree': {}
        }
        self.graph = None
        self.update_permutation('Random', np.random.permutation(self.n))

    def use_graph(self):
        if self.graph is not None:
            return
        self.graph = nx.Graph()
        for i in range(self.n):
            for j in range(i, self.n):
                self.graph.add_edge(i, j, weight=self.dist(i, j))
                self.graph.add_edge(j, i, weight=self.dist(j, i))

    def update_permutation(self, algorithm: str, new_permutation: np.ndarray):
        self.data[algorithm]['Edges'] = [(new_permutation[i], new_permutation[i+1])
                                         for i in range(self.n-1)] + [(new_permutation[-1], new_permutation[0])]
        self.data[algorithm]['Result'] = self.calculate_total_distance(self.data[algorithm]['Edges'])

    def make_edge_list(self, permutation: np.ndarray):
        edges = []
        for i in range(self.n-1):
            edges.append((permutation[i], permutation[i+1]))
        edges.append((permutation[-1], permutation[0]))
        return edges

    def calculate_total_distance(self, edges: List[tuple[int, int]]):
        res = 0
        for edge in edges:
            res += self.dist(edge[0], edge[1])
        return res

    def draw(self, algorithm: str):
        edges = self.data[algorithm]['Edges']
        values = self.values
        res = self.data[algorithm]['Result']
        units = 'total weight' if algorithm == 'Minimal Spanning Tree' else 'result'
        priority = ['Random', 'Greedy', '2-Approximation', 'Minimal Spanning Tree'].index(algorithm)
        # 2-dimensional both for circular and non-circular
        if not self.three_dim:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
            for edge in edges:
                i = edge[0]
                j = edge[1]
                ax.plot([values[0][i], values[0][j]],
                        [values[1][i], values[1][j]], alpha=0.4, color='#0954b1', zorder=1)
                ax.scatter(values[0][i], values[1][i], zorder=2, color='#0954b1')
                ax.scatter(values[0][j], values[1][j], zorder=2, color='#0954b1')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{algorithm} {units}: {int(res)}")
            plt.savefig(f'static/graphs/{priority}_{algorithm}.png')
        # 3-dimensional sphere
        elif self.circular and self.three_dim:
            fig = plt.figure(figsize=(5, 5), dpi=200)
            ax = plt.axes(projection='3d')

            n = self.n
            r = self.mul
            phi = self.phi
            psi = self.psi
            values = self.values

            edges = self.data[algorithm]['Edges']

            def get_x(u, v):
                return np.sin(u) * np.cos(v) * r

            def get_y(u, v):
                return np.sin(u) * np.sin(v) * r

            def get_z(u, v):
                return np.cos(u) * r

            fig = plt.figure(figsize=(5, 5), dpi=200)
            ax = plt.axes(projection='3d')
            camera = Camera(fig)

            angle = 0

            frames = 80
            for _ in range(frames):
                ax.set_box_aspect([1, 1, 1])
                angle += 2 * np.pi / frames

                for edge in edges:
                    i = edge[0]
                    j = edge[1]
                    u1, u2 = phi[i], phi[j]
                    v1, v2 = psi[i], psi[j]
                    u, v = get_lin(u1, u2, v1, v2)
                    ax.plot(rotate_x(get_x(u, v), get_y(u, v), angle), rotate_y(get_x(u, v), get_y(u, v), angle),
                            get_z(u, v), alpha=0.6, color='#0954b1', zorder=1)
                    ax.scatter(rotate_x(values[0][i], values[1][i], angle),
                               rotate_y(values[0][i], values[1][i], angle), values[2][i], zorder=2,
                               color='#0954b1')
                    ax.scatter(rotate_x(values[0][j], values[1][j], angle),
                               rotate_y(values[0][j], values[1][j], angle), values[2][j],
                               zorder=2, color='#0954b1')

                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = r * np.outer(np.cos(u), np.sin(v))
                y = r * np.outer(np.sin(u), np.sin(v))
                z = r * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, rstride=4, cstride=4, color='#0954b1', linewidth=0, alpha=0.05)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_title(f'{algorithm} {units}: {int(res)}')

                camera.snap()

            animation = camera.animate(interval=100, repeat=True)
            animation.save(f'static/graphs/{priority}_{algorithm}.gif')

        # just 3-dimensional
        else:
            fig = plt.figure(figsize=(5, 5), dpi=200)
            ax = plt.axes(projection='3d')
            camera = Camera(fig)

            angle = 0

            frames = 80

            for _ in range(frames):
                ax.set_box_aspect([1, 1, 1])
                angle += 2 * np.pi / frames
                for edge in edges:
                    i = edge[0]
                    j = edge[1]
                    x1 = rotate_x(values[0][i], values[1][i], angle)
                    y1 = rotate_y(values[0][i], values[1][i], angle)
                    x2 = rotate_x(values[0][j], values[1][j], angle)
                    y2 = rotate_y(values[0][j], values[1][j], angle)
                    ax.plot([x1, x2], [y1, y2], [values[2][i], values[2][j]], alpha=0.4, color='#0954b1', zorder=1)
                    ax.scatter(x1, y1, values[2][i], zorder=2, color='#0954b1')
                    ax.scatter(x2, y2, values[2][j], zorder=2, color='#0954b1')
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_title(f"{algorithm} result: {int(res)}")

                camera.snap()

            animation = camera.animate(interval=100, repeat=True)
            animation.save(f'static/graphs/{priority}_{algorithm}.gif')

    def dist(self, i: int, j: int):
        x1, y1, x2, y2 = self.values[0][i], self.values[1][i], self.values[0][j], self.values[1][j]
        z1, z2 = (self.values[2][i], self.values[2][j]) if self.three_dim else (0, 0)
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        if self.circular and self.three_dim:
            return 2 * self.mul * np.arcsin(dist / (2 * self.mul))
        return dist

        # Algorithms

    def use_greedy(self):
        final_res = 2 * self.mul * self.n
        final_permutation = np.array([])
        for start in range(self.n):
            path = [start]
            location = start
            taboo_list = [True] * self.n
            taboo_list[start] = False
            while sum(taboo_list) != 0:
                dists = [self.dist(location, i) if taboo_list[i] else self.mul * self.n for i in range(self.n)]
                location = dists.index(min(dists))
                path.append(location)
                taboo_list[location] = False
            res = self.calculate_total_distance(self.make_edge_list(np.array(path)))
            if final_res > res:
                final_res = res
                final_permutation = np.array(path)
        self.update_permutation('Greedy', np.array(final_permutation))

    def use_2_approximation(self):
        self.use_graph()
        mst = tree.minimum_spanning_edges(self.graph, algorithm="kruskal", data=False)
        edges = list(mst)
        euler = nx.Graph(mst)
        k = len(edges)
        for i in range(k):
            euler.add_edge(edges[i][1], edges[i][0])
        euler = nx.eulerize(euler)
        path = nx.eulerian_circuit(euler)
        taboo_list = [True] * self.n
        res = []
        for edge in path:
            if taboo_list == [True] * self.n:
                taboo_list[edge[0]] = False
                res.append(edge[0])
            if taboo_list[edge[1]]:
                taboo_list[edge[1]] = False
                res.append(edge[1])
        self.update_permutation('2-Approximation', np.array(res))

    def use_mst(self):
        self.use_graph()
        mst = tree.minimum_spanning_edges(self.graph, algorithm="kruskal", data=False)
        edges = list(mst)
        res = np.sum([self.dist(edge[0], edge[1]) for edge in edges])
        name = 'Minimal Spanning Tree'
        self.data[name]['Edges'] = edges
        self.data[name]['Result'] = res
