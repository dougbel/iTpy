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


def visualize(trainer, sampler, tri_mesh_env, tri_mesh_obj):
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
    scene.show(callback=get_camera)


if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")
    interaction_to_test = 'hang'
    interaction = interactions_data[ interactions_data['interaction'] == interaction_to_test]

    tri_mesh_env = trimesh.load_mesh( interaction[ 'tri_mesh_env'][0] )
    tri_mesh_obj = trimesh.load_mesh( interaction[ 'tri_mesh_obj'][0] )
    tri_mesh_ibs_segmented = trimesh.load_mesh( interaction[ 'tri_mesh_ibs_segmented'][0] )
    np_cloud_env = np.asarray(o3d.io.read_point_cloud(interaction[ 'o3d_cloud_sources_ibs'][0] ).points)

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
    visualize(trainer_poisson, sampler_poissondisc_weighted,  tri_mesh_env, tri_mesh_obj)

    visualize(trainer_vertices, sampler_ibs_vertices_weighted,  tri_mesh_env, tri_mesh_obj)

    visualize(trainer_weighted, sampler_ibs_srcs_weighted, tri_mesh_env, tri_mesh_obj)
