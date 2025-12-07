import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from abc import ABC, abstractmethod

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices, dtype=float)
        self.faces = np.array(faces, dtype=int)

    @classmethod
    def from_obj(cls, filename):
        verts, faces = [], []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    verts.append([float(p) for p in parts[1:4]])
                elif parts[0] == 'f':
                    faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])
        return cls(verts, faces)

class TopologyBuilder:
    @staticmethod
    def build_adjacency(mesh):
        faces = mesh.faces
        n_faces = len(faces)

        edges_raw = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])

        edges_sorted = np.sort(edges_raw, axis=1)

        unique_edges, inverse_indices = np.unique(edges_sorted, axis=0, return_inverse=True)

        edge_to_faces = {i: [] for i in range(len(unique_edges))}

        for i, edge_idx in enumerate(inverse_indices):
            face_idx = i % n_faces
            edge_to_faces[edge_idx].append(face_idx)

        vert_neighbors = {i: set() for i in range(len(mesh.vertices))}
        for u, v in unique_edges:
            vert_neighbors[u].add(v)
            vert_neighbors[v].add(u)

        return unique_edges, edge_to_faces, vert_neighbors

class SubdivisionStrategy(ABC):
    @abstractmethod
    def apply(self, mesh: Mesh) -> Mesh:
        pass

    def _split_edges(self, mesh, unique_edges, new_edge_points):
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)

        edge_idx_map = {tuple(edge): (n_verts + i) for i, edge in enumerate(unique_edges)}

        new_faces = []
        for face in mesh.faces:
            v0, v1, v2 = face

            e0 = edge_idx_map[tuple(sorted((v0, v1)))]
            e1 = edge_idx_map[tuple(sorted((v1, v2)))]
            e2 = edge_idx_map[tuple(sorted((v2, v0)))]

            new_faces.append([v0, e0, e2])
            new_faces.append([v1, e1, e0])
            new_faces.append([v2, e2, e1])
            new_faces.append([e0, e1, e2])

        new_vertices = np.vstack([mesh.vertices, new_edge_points])

        return Mesh(new_vertices, new_faces)

class LoopStrategy(SubdivisionStrategy):
    def apply(self, mesh: Mesh) -> Mesh:
        unique_edges, edge_to_faces, vert_neighbors = TopologyBuilder.build_adjacency(mesh)
        verts = mesh.vertices
        faces = mesh.faces

        new_edge_points = []
        for i, edge in enumerate(unique_edges):
            u, v = edge
            face_indices = edge_to_faces[i]

            if len(face_indices) == 2:
                f0 = faces[face_indices[0]]
                f1 = faces[face_indices[1]]

                opp0 = [x for x in f0 if x not in edge][0]
                opp1 = [x for x in f1 if x not in edge][0]

                pt = (3/8)*(verts[u] + verts[v]) + (1/8)*(verts[opp0] + verts[opp1])
            else:
                pt = 0.5 * (verts[u] + verts[v])
            new_edge_points.append(pt)

        updated_old_verts = np.zeros_like(verts)
        for i in range(len(verts)):
            neighbors = list(vert_neighbors[i])
            k = len(neighbors)
            if k == 0:
                updated_old_verts[i] = verts[i]
                continue

            beta = (3/16) if k == 3 else (3 / (8*k))

            sum_neighbors = np.sum(verts[neighbors], axis=0)
            updated_old_verts[i] = (1.0 - k*beta) * verts[i] + beta * sum_neighbors

        mesh_temp = Mesh(updated_old_verts, mesh.faces)

        return self._split_edges(mesh_temp, unique_edges, np.array(new_edge_points))

class ButterflyStrategy(SubdivisionStrategy):
    def apply(self, mesh: Mesh) -> Mesh:
        unique_edges, edge_to_faces, _ = TopologyBuilder.build_adjacency(mesh)
        verts = mesh.vertices
        faces = mesh.faces

        pair_to_opp = {}
        for f in faces:
            pair_to_opp[(f[0], f[1])] = f[2]
            pair_to_opp[(f[1], f[2])] = f[0]
            pair_to_opp[(f[2], f[0])] = f[1]

        new_edge_points = []
        for i, edge in enumerate(unique_edges):
            u, v = edge
            face_indices = edge_to_faces[i]

            pt = 0.5 * (verts[u] + verts[v])

            if len(face_indices) == 2:
                try:
                    opp0 = pair_to_opp.get((u, v))
                    opp1 = pair_to_opp.get((v, u))

                    if opp0 is not None and opp1 is not None:
                        t0 = pair_to_opp.get((u, opp0))
                        t1 = pair_to_opp.get((opp0, v))
                        t2 = pair_to_opp.get((opp1, u))
                        t3 = pair_to_opp.get((v, opp1))

                        if None not in [t0, t1, t2, t3]:
                            pt = 0.5 * (verts[u] + verts[v]) + \
                                 (2 * 0.0625) * (verts[opp0] + verts[opp1]) - \
                                 (0.0625) * (verts[t0] + verts[t1] + verts[t2] + verts[t3])
                except:
                    pass

            new_edge_points.append(pt)

        return self._split_edges(mesh, unique_edges, np.array(new_edge_points))

class SubdivisionApp:
    def __init__(self, obj_file):
        self.base_mesh = Mesh.from_obj(obj_file)
        self.current_mesh = self.base_mesh

        self.level = 0
        self.strategies = {
            'Loop (Approx)': LoopStrategy(),
            'Butterfly (Interp)': ButterflyStrategy()
        }
        self.current_strategy_name = 'Loop (Approx)'

        self.fig = plt.figure(figsize=(14, 8))

        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.05)

        self.ax = self.fig.add_subplot(111, projection='3d')

        self._init_widgets()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.update_plot()

    def _init_widgets(self):
        ax_color = 'lightgoldenrodyellow'

        ax_lvl = plt.axes([0.02, 0.6, 0.15, 0.25], facecolor=ax_color)
        self.radio_lvl = RadioButtons(ax_lvl, ('0', '1', '2', '3', '4'))
        self.radio_lvl.on_clicked(self.on_level_change)
        ax_lvl.text(0.5, 1.05, "Iterációk (Szint)", ha='center', transform=ax_lvl.transAxes, fontsize=10, weight='bold')

        ax_alg = plt.axes([0.02, 0.3, 0.15, 0.15], facecolor=ax_color)
        self.radio_alg = RadioButtons(ax_alg, list(self.strategies.keys()))
        self.radio_alg.on_clicked(self.on_method_change)
        ax_alg.text(0.5, 1.05, "Algoritmus", ha='center', transform=ax_alg.transAxes, fontsize=10, weight='bold')

        self.info_text = self.fig.text(0.02, 0.05,
                                       "Billentyűk:\n'0'-'4': Szint\n'L': Loop\n'B': Butterfly",
                                       fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    def on_key_press(self, event):
        if event.key in ['0', '1', '2', '3', '4']:
            self.radio_lvl.set_active(int(event.key))
        elif event.key.lower() == 'l':
            self.radio_alg.set_active(0)
        elif event.key.lower() == 'b':
            self.radio_alg.set_active(1)

    def process_mesh(self):
        mesh = self.base_mesh
        strategy = self.strategies[self.current_strategy_name]

        print(f"Számítás: {self.current_strategy_name}, Szint: {self.level}...")

        for i in range(self.level):
            mesh = strategy.apply(mesh)

        self.current_mesh = mesh

    def on_level_change(self, label):
        self.level = int(label)
        self.process_mesh()
        self.update_plot()

    def on_method_change(self, label):
        self.current_strategy_name = label
        self.process_mesh()
        self.update_plot()

    def update_plot(self):
        v = self.current_mesh.vertices
        f = self.current_mesh.faces

        self.ax.clear()

        title = f"{self.current_strategy_name}\nLevel: {self.level} | Verts: {len(v)} | Faces: {len(f)}"
        self.ax.set_title(title, fontsize=12)

        self._center_camera(v)

        self.ax.set_axis_off()
        self.ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=f,
                             cmap='plasma', edgecolor='k', linewidth=0.2, alpha=0.9, antialiased=True)

        self.fig.canvas.draw_idle()

    def _center_camera(self, v):
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        centers = (mins + maxs) / 2
        max_range = (maxs - mins).max() / 2

        self.ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
        self.ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
        self.ax.set_zlim(centers[2] - max_range, centers[2] + max_range)
        self.ax.set_box_aspect([1,1,1])

    def show(self):
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager.window, 'state'):
                manager.window.state('zoomed')
        except: pass
        plt.show()

if __name__ == "__main__":
    try:
        app = SubdivisionApp("modell_3.obj")
        app.show()
    except FileNotFoundError:
        print("valami nem lesz jó")