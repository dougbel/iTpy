import os
import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
from sklearn import preprocessing

from it import util
from it.training.maxdistancescalculator import MaxDistancesCalculator
from it.training.sampler import OnGivenPointCloudWeightedSampler
from it.training.trainer import Trainer

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'ride'
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

    rate_generated_random_numbers = 500

    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer_weighted = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)
    max_d = MaxDistancesCalculator(pv_points=trainer_weighted.pv_points, pv_vectors=trainer_weighted.pv_vectors,
                                   tri_mesh_obj=tri_mesh_obj, consider_collision_with_object=True,
                                   radio_ratio=influence_radio_ratio)

    print(max_d.sum_max_distances)

    pv_origin = trimesh.points.PointCloud(trainer_weighted.pv_points, color=[255, 0, 0, 255])

    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    max_d.sphere_of_influence.visual.face_colors = [0, 0, 255, 25]

    pv_3d_path = np.hstack((trainer_weighted.pv_points,
                            trainer_weighted.pv_points + trainer_weighted.pv_vectors)).reshape(-1, 2, 3)

    # pv_normalized_vectors = preprocessing.normalize(trainer_weighted.pv_vectors)
    pv_max_vectors = preprocessing.normalize(trainer_weighted.pv_vectors) * max_d.max_distances.reshape(-1, 1)
    calculated_max_intersections = trainer_weighted.pv_points + pv_max_vectors

    pv_max_path = np.hstack((trainer_weighted.pv_points,
                             calculated_max_intersections)).reshape(-1, 2, 3)

    pv_intersections = trimesh.points.PointCloud(calculated_max_intersections, color=[0, 0, 255, 250])

    provenance_vectors = trimesh.load_path(pv_3d_path)
    provenance_vectors_max_path = trimesh.load_path(pv_max_path)

    scene = trimesh.Scene([
        provenance_vectors,
        pv_origin,
        tri_mesh_ibs_segmented,
        tri_mesh_obj,
        tri_mesh_env
    ])

    scene.show(flags={'cull': False, 'wireframe': False, 'axis': False})

    scene = trimesh.Scene([
        provenance_vectors_max_path,
        pv_origin,
        pv_intersections,
        max_d.sphere_of_influence,
        tri_mesh_obj
    ])
    scene.show(flags={'cull': False})
