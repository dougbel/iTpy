import math
from enum import Enum
from collections import Counter
import numpy as np

import trimesh
from transforms3d.derivations.eulerangles import z_rotation

import it.util as util


class Trainer:

    def __init__(self, tri_mesh_ibs, tri_mesh_env, sampler):
        self._get_env_normal(tri_mesh_env)
        self._getIBSCloud(tri_mesh_ibs, tri_mesh_env, sampler)
        self._set_pv_min_max_mapped_norms()

    def _getIBSCloud(self, tri_mesh_ibs, tri_mesh_env, sampler):
        sampler.execute(tri_mesh_ibs, tri_mesh_env)

        self.pv_points = sampler.pv_points
        self.pv_vectors = sampler.pv_vectors
        self.pv_norms = sampler.pv_norms

    def _get_env_normal(self, tri_mesh_env):
        (_, __, triangle_id) = tri_mesh_env.nearest.on_surface(np.array([0, 0, 0]).reshape(-1, 3))
        self.env_normal = tri_mesh_env.face_normals[triangle_id]

    def _map_norm(self, norm, max, min):
        return (norm - min) * (0 - 1) / (max - min) + 1
        # return ( value_in - min_original_range) * (max_mapped- min_mapped) / (max_original_range - min_original_range) + min_mapped;

    def _set_pv_min_max_mapped_norms(self):
        self.pv_max_norm = self.pv_norms.max()
        self.pv_min_norm = self.pv_norms.min()

        self.pv_mapped_norms = np.asarray([self._map_norm(norm, self.pv_max_norm, self.pv_min_norm) for norm in self.pv_norms])
