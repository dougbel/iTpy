import numpy as np
import sys

import trimesh


class Trainer:
    normal_env = np.asarray([])

    pv_points = np.asarray([])
    pv_vectors = np.asarray([])
    pv_norms = np.asarray([])

    pv_max_norm = sys.float_info.min
    pv_min_norm = sys.float_info.max
    pv_mapped_norms = np.asarray([])

    def __init__(self, tri_mesh_ibs, tri_mesh_env, sampler):
        self._get_env_normal(tri_mesh_env)
        self._get_provenance_vectors(tri_mesh_ibs, tri_mesh_env, sampler)
        self._set_pv_min_max_mapped_norms()
        self._get_mapped_norms()
        self.order_by_mapped_norms()

    def order_by_mapped_norms(self):
        idx_order = np.argsort(self.pv_mapped_norms)[::-1]
        self.pv_points = self.pv_points[idx_order]
        self.pv_vectors = self.pv_vectors[idx_order]
        self.pv_norms = self.pv_norms[idx_order]
        self.pv_mapped_norms = self.pv_mapped_norms[idx_order]


    def _get_provenance_vectors(self, tri_mesh_ibs, tri_mesh_env, sampler):
        sampler.execute(tri_mesh_ibs, tri_mesh_env)
        self.pv_points = sampler.pv_points
        self.pv_vectors = sampler.pv_vectors
        self.pv_norms = sampler.pv_norms

    def _get_env_normal(self, tri_mesh_env):
        (_, __, triangle_id) = tri_mesh_env.nearest.on_surface(np.array([0, 0, 0]).reshape(-1, 3))
        self.normal_env = tri_mesh_env.face_normals[triangle_id]

    def _map_norm(self, norm, max, min):
        return (norm - min) * (0 - 1) / (max - min) + 1

    def _set_pv_min_max_mapped_norms(self):
        self.pv_max_norm = self.pv_norms.max()
        self.pv_min_norm = self.pv_norms.min()

    def _get_mapped_norms(self):
        self.pv_mapped_norms = np.asarray(
            [self._map_norm(norm, self.pv_max_norm, self.pv_min_norm) for norm in self.pv_norms])
