import math
import json
import os
import numpy as np
import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation


class Agglomerator:
    ORIENTATIONS = 8
    it_trainer = None
    def __init__(self, it_trainer):

        self.it_trainer = it_trainer

        orientations = [x * (2 * math.pi / self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]

        agglomerated_pv_points = []
        agglomerated_pv_vectors = []
        agglomerated_pv_vdata = []
        agglomerated_normals = []

        self.trainer = it_trainer
        self.sample_size = it_trainer.pv_points.shape[0]

        pv_vdata = np.zeros((self.sample_size, 3), np.float64)
        pv_vdata[:, 0:2] = np.hstack((it_trainer.pv_norms.reshape(-1, 1), it_trainer.pv_mapped_norms.reshape(-1, 1)))

        for angle in orientations:
            rotation = z_rotation(angle)
            agglomerated_pv_points.append(np.dot(it_trainer.pv_points, rotation))
            agglomerated_pv_vectors.append(np.dot(it_trainer.pv_vectors, rotation))
            agglomerated_pv_vdata.append(pv_vdata)
            agglomerated_normals.append(np.dot(it_trainer.normal_env, rotation))

        self.agglomerated_pv_points = np.asarray(agglomerated_pv_points).reshape(-1, 3)
        self.agglomerated_pv_vectors = np.asarray(agglomerated_pv_vectors).reshape(-1, 3)
        self.agglomerated_pv_vdata = np.asarray(agglomerated_pv_vdata).reshape(-1, 3)
        self.agglomerated_normals = np.asarray(agglomerated_normals).reshape(-1, 3)
