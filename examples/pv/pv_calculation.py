import os
import numpy as np
import pandas as pd
import open3d as o3d
import trimesh

from it.training.trainer import Trainer
from it.training.sampler import *
import it.util as util


def get_camera(scene):
    np.set_printoptions(suppress=True)
    print(scene.camera_transform)


def visualize(trainer, sampler, tri_mesh_env, tri_mesh_obj, caption):
    tri_cloud_ibs = trimesh.points.PointCloud(sampler.np_cloud_ibs, color=[255, 0, 0, 100])
    pv_origin = trimesh.points.PointCloud(trainer.pv_points, color=[0, 0, 255, 250])

    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]

    pv_3d_path = np.hstack((trainer.pv_points, trainer.pv_points + trainer.pv_vectors)).reshape(-1, 2, 3)

    provenance_vectors = trimesh.load_path(pv_3d_path)

    scene = trimesh.Scene([
        provenance_vectors,
        pv_origin,
        tri_cloud_ibs,
        tri_mesh_env,
        tri_mesh_obj
    ])
    scene.show(callback=get_camera, caption=caption)


if __name__ == '__main__':
    '''
    Show the three different weighted sampling strategies developed and show a visualization of them
    '''
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'hang'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    directory = interaction.iloc[0]['directory']

    tri_mesh_env = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_obj']))
    tri_mesh_ibs = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_ibs']))
    o3d_cloud_src_ibs = o3d.io.read_point_cloud(os.path.join(directory, interaction.iloc[0]['o3d_cloud_sources_ibs']))
    np_cloud_env = np.asarray(o3d_cloud_src_ibs.points)

    influence_radio_ratio = 2
    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    rate_ibs_samples = 5
    rate_generated_random_numbers = 500

    # sampler_poissondisc_random = PoissonDiscRandomSampler( rate_ibs_samples )
    sampler_poissondisc_weighted = PoissonDiscWeightedSampler(rate_ibs_samples, rate_generated_random_numbers)
    # sampler_meshvertices_random =  OnVerticesRandomSampler()
    sampler_ibs_vertices_weighted = OnVerticesWeightedSampler(rate_generated_random_numbers)
    # sampler_ibs_srcs_randomly =  OnGivenPointCloudRandomSampler( np_input_cloud = np_cloud_env )
    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer_poisson = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_poissondisc_weighted)
    trainer_vertices = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_vertices_weighted)
    trainer_weighted = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)

    # VISUALIZATION
    visualize(trainer_poisson, sampler_poissondisc_weighted, tri_mesh_env, tri_mesh_obj, "poisson disc weighted")
    visualize(trainer_vertices, sampler_ibs_vertices_weighted, tri_mesh_env, tri_mesh_obj, "ibs_vertices weighted")
    visualize(trainer_weighted, sampler_ibs_srcs_weighted, tri_mesh_env, tri_mesh_obj, "ibs sources weighted")
