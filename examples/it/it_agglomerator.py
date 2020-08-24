import os
import numpy as np
import pandas as pd

import open3d as o3d
import trimesh
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose

from it.training.sampler import *
from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'hang'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    directory = interaction.iloc[0]['directory']

    tri_mesh_env = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_obj']))
    tri_mesh_ibs = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_ibs']))
    o3d_cloud_src_ibs = o3d.io.read_point_cloud(os.path.join(directory, interaction.iloc[0]['o3d_cloud_sources_ibs']))
    np_cloud_env = np.asarray(o3d_cloud_src_ibs.points)

    influence_radio_ratio = 1.2
    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]

    rate_generated_random_numbers = 500

    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)

    agg = Agglomerator(trainer)

    angle = (2 * math.pi / agg.ORIENTATIONS)


    for ori in range(agg.ORIENTATIONS):
        idx_from = ori * agg.sample_size
        idx_to = idx_from + agg.sample_size

        pv_origin = agg.agglomerated_pv_points[idx_from:idx_to]
        pv_final = agg.agglomerated_pv_points[idx_from:idx_to] + agg.agglomerated_pv_vectors[idx_from:idx_to]

        reference_ori = np.array([[0, 0, 0]])
        reference_fin = np.array([[0.5, 0, 0]])

        R = z_rotation(angle * ori)  # accumulated rotation in each iteration
        Z = np.ones(3)  # zooms
        T = [0, 0, 0]  # translation
        A = compose(T, R, Z)

        tri_mesh_obj.apply_transform(A)
        tri_mesh_env.apply_transform(A)
        reference_ori = np.dot(reference_ori, R.T) # np.mat(reference_ori)*np.mat(R)
        reference_fin = np.dot(reference_fin, R.T) # np.mat(reference_ori)*np.mat(R)

        # VISUALIZATION
        tri_pv = trimesh.load_path(np.hstack((pv_origin, pv_final)).reshape(-1, 2, 3))
        tri_pv_origin = trimesh.points.PointCloud(pv_origin, color=[0, 0, 255, 250])

        reference = trimesh.load_path(np.hstack((reference_ori, reference_fin)).reshape(-1, 2, 3))

        scene = trimesh.Scene([
            tri_mesh_env,
            tri_pv,
            tri_pv_origin,
            tri_mesh_obj,
            reference
        ])
        scene.show()

        tri_mesh_obj.apply_transform(linalg.inv(A))
        tri_mesh_env.apply_transform(linalg.inv(A))
