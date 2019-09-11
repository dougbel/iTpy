import numpy as np

import open3d as o3d
import trimesh

from  it.training.trainer import Trainer
from it.training.sampler import *
import it.util as util


def get_camera(scene):
  np.set_printoptions(suppress=True)
  print(scene.camera_transform)


def visualize ( trainer, tri_mesh_env, tri_mesh_obj ):
    tri_cloud_ibs = trimesh.points.PointCloud( trainer.np_cloud_ibs, color=[255,0,0,100] )
    pv_origin     = trimesh.points.PointCloud( trainer.pv_points, color=[0,0,255,250] )

    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]

    pv_3d_path = np.hstack(( trainer.pv_points,trainer.pv_points + trainer.pv_vectors)).reshape(-1, 2, 3)

    provenance_vectors = trimesh.load_path( pv_3d_path )
    
    scene = trimesh.Scene( [ 
                            provenance_vectors,
                            pv_origin,
                            tri_cloud_ibs,
                            tri_mesh_env,
                            tri_mesh_obj
                            ])
    scene.show(callback= get_camera)


if __name__ == '__main__':

    #PLACE BOWL TABLE
    #tri_mesh_env = trimesh.load_mesh( './data/interactions/table_bowl/table.ply' )
    #tri_mesh_obj = trimesh.load_mesh( './data/interactions/table_bowl/bowl.ply' )
    #tri_mesh_ibs_segmented = trimesh.load_mesh( './data/interactions/table_bowl/ibs_table_bowl_sampled_600_resamplings_2.ply' )
    #np_cloud_env = np.asarray( o3d.io.read_point_cloud( "./data/interactions/table_bowl/env_samplings_ibs_table_bowl_sample_600_resamplings_2.pcd" ) )
    #RIDE A MOTORCYCLE
    #tri_mesh_env = trimesh.load_mesh("./data/interactions/motorbike_rider/motorbike.ply")
    #tri_mesh_obj = trimesh.load_mesh("./data/interactions/motorbike_rider/biker.ply")
    #tri_mesh_ibs_segmented = trimesh.load_mesh("./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_600_resamplings_2.ply")
    #np_cloud_env = np.asarray( o3d.io.read_point_cloud( "./data/interactions/motorbike_rider/env_samplings_ibs_motorbike_biker_sample_600_resamplings_2.pcd" ) )
    #HANK AN UMBRELLA
    tri_mesh_env = trimesh.load_mesh("./data/interactions/hanging-rack_umbrella/hanging-rack.ply")
    tri_mesh_obj = trimesh.load_mesh("./data/interactions/hanging-rack_umbrella/umbrella.ply")
    tri_mesh_ibs_segmented = trimesh.load_mesh("./data/interactions/hanging-rack_umbrella/ibs_hanging-rack_umbrella_sampled_600_resamplings_2.ply")
    np_cloud_env = np.asarray( o3d.io.read_point_cloud( "./data/interactions/hanging-rack_umbrella/env_samplings_ibs_hanging-rack_umbrella_sample_600_resamplings_2.pcd" ) )


    rate_ibs_samples = 5
    rate_generated_random_numbers = 500
    
    #sampler_poissondisc_random = PoissonDiscRandomSampler( rate_ibs_samples ) 
    sampler_poissondisc_weighted = PoissonDiscWeightedSampler( rate_ibs_samples=rate_ibs_samples, rate_generated_random_numbers=rate_generated_random_numbers)
    #sampler_meshvertices_random =  OnVerticesRandomSampler()
    sampler_ibs_vertices_weighted =  OnVerticesWeightedSampler( rate_generated_random_numbers=rate_generated_random_numbers )
    #sampler_ibs_srcs_randomly =  OnGivenPointCloudRandomSampler( np_input_cloud = np_cloud_env )
    sampler_ibs_srcs_weighted =  OnGivenPointCloudWeightedSampler( np_input_cloud = np_cloud_env, rate_generated_random_numbers=rate_generated_random_numbers)

    trainer_poisson = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, sampler_poissondisc_weighted )    
    trainer_vertices = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_vertices_weighted )
    trainer_weighted = Trainer( tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted )



    #VISUALIZATION    

    visualize ( trainer_poisson, tri_mesh_env, tri_mesh_obj )

    visualize ( trainer_vertices, tri_mesh_env, tri_mesh_obj )

    visualize ( trainer_weighted, tri_mesh_env, tri_mesh_obj )