'''
This implementation inttend to check teh availibility of used the point that were used to
generate the IBS surfaces instead of sampling directly in the IBS (poisson disk, vertices, uniform)
'''
import time
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

import open3d as o3d
import trimesh

from   it.training.sampler import *
import it.util as util

def get_camera(scene):
  np.set_printoptions(suppress=True)
  print(scene.camera_transform)


def visualize(sampler):
    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_cloud_ibs = trimesh.points.PointCloud( sampler.np_cloud_ibs, color=[0,0,255,150] )
    pv_origin     = trimesh.points.PointCloud( sampler.pv_points, color=[0,0,255,250] )
    pv_3d_path = np.hstack(( sampler.pv_points,sampler.pv_points + sampler.pv_vectors)).reshape(-1, 2, 3)
    provenance_vectors = trimesh.load_path( pv_3d_path )
    scene = trimesh.Scene( [ provenance_vectors, pv_origin, tri_cloud_ibs, tri_mesh_env, tri_mesh_obj ])
    
    scene.show(callback=get_camera)

if __name__ == '__main__':
    

    data_frame = pd.DataFrame(columns=['obj', 'env', 'rate_samples_in_ibs', 'rate_random_samples',
                                       'ibs_sampled_points', 'random_num_generated', 'exec_time'])

    to_test = pd.DataFrame([
            {
            'env': "hanging-rack", 
            'obj': "umbrella", 
            'tri_mesh_env': "./data/interactions/hanging-rack_umbrella/hanging-rack.ply", 
            'tri_mesh_obj': "./data/interactions/hanging-rack_umbrella/umbrella.ply", 
            'tri_mesh_ibs_segmented': "./data/interactions/hanging-rack_umbrella/ibs_hanging-rack_umbrella_sampled_3000_resamplings_2.ply"
            }
            #,
            #{'env': "table", 
            #'obj': "bowl", 
            #'tri_mesh_env': './data/interactions/table_bowl/table.ply', 
            #'tri_mesh_obj': './data/interactions/table_bowl/bowl.ply', 
            #'tri_mesh_ibs_segmented': './data/pv/ibs_mesh_segmented.ply'
            #},
            #{
            #'env': "motorbike", 
            #'obj': "rider", 
            #'tri_mesh_env': "./data/interactions/motorbike_rider/motorbike.ply", 
            #'tri_mesh_obj': "./data/interactions/motorbike_rider/biker.ply", 
            #'tri_mesh_ibs_segmented': "./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_3000_resamplings_2.ply"
            #}
            ])
    


for index_label, row_series in to_test.iterrows():
    env = to_test.at[index_label,'env']
    obj = to_test.at[index_label,'obj']
    tri_mesh_env = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_env'] )
    tri_mesh_obj = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_obj'] )
    tri_mesh_ibs_segmented = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_ibs_segmented'] )

    rate_ibs_samples = 5
    rate_generated_random_numbers = 7

    #start = time.time()  # timing execution
    sampler_poissondisc_random = PoissonDiscRandomSampler(tri_mesh_ibs_segmented, tri_mesh_env) 

    visualize( sampler_poissondisc_random )


    sampler_poissondisc_weighted = PoissonDiscWeightedSampler(tri_mesh_ibs_segmented, 
                                                                tri_mesh_env,  
                                                                rate_ibs_samples=rate_ibs_samples, 
                                                                rate_generated_random_numbers=rate_generated_random_numbers)

    visualize( sampler_poissondisc_weighted )

    sampler_meshvertices_random =  OnVerticesRandomSampler(tri_mesh_ibs_segmented, tri_mesh_env)
    
    visualize( sampler_meshvertices_random )


    #end = time.time()  # timing execution
    #execution_time = end - start

    #ibs_sampled_points = rate_ibs_samples * trainer_weighted.SAMPLE_SIZE
    #random_num_generated = rate_generated_random_numbers * ibs_sampled_points

    #data_frame.loc[len(data_frame)] = [obj, env, rate_ibs_samples, rate_generated_random_numbers,
    #                                    ibs_sampled_points, random_num_generated, execution_time]

    #VISUALIZING information
   