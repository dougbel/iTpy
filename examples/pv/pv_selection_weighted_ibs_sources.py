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
    scene = generate_scene(sampler)
    scene.show()

def generate_scene(sampler):
    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_cloud_ibs = trimesh.points.PointCloud( sampler.np_cloud_ibs, color=[0,255,255,100] )
    pv_origin     = trimesh.points.PointCloud( sampler.pv_points, color=[0,0,255,250] )
    pv_3d_path = np.hstack(( sampler.pv_points,sampler.pv_points + sampler.pv_vectors)).reshape(-1, 2, 3)
    provenance_vectors = trimesh.load_path( pv_3d_path )
    scene = trimesh.Scene( [ provenance_vectors, pv_origin, tri_cloud_ibs, tri_mesh_env, tri_mesh_obj ])
    return scene

def saveOutput(sampler, scene, output_dir, file_name):
    camera_transformations =[ [[ 1., 0., 0., -0.000145  ], [ 0., 1., 0., -0.000213  ,], [ 0., 0., 1., 0.54402789], [ 0., 0., 0., 1. ] ],
        [[ 1., 0., 0., 0.04396537], [ 0., 1., 0., 0.23108601], [-0., 0., 1., 3.35716683], [ 0., 0., 0., 1. ] ],
        [[ 0.99948765, 0.02547537, -0.01937649, -0.00579407], [-0.02175125, 0.09651196, -0.99509413, -2.77415963], [-0.02348033, 0.99500575, 0.09701663, 0.41565603], [ 0., 0., 0., 1. ] ],
        [[-0.16499592, -0.05685685, 0.98465407, 3.39108636], [ 0.9861865,  0.00524637, 0.16555564, 0.71746308], [-0.01457883, 0.99836856, 0.05520583, 0.11499131], [ 0., 0., 0., 1. ] ],
        [[-0.53874073, -0.34343771, 0.76929121, 2.63196857], [ 0.8423785, -0.20601971, 0.49795014, 1.92104601], [-0.0125257,  0.9163004,  0.40029575, 1.34028108], [ 0., 0., 0., 1. ] ],
        [[-0.34202215, -0.63235448, 0.69508896, 0.64184869], [ 0.92533648, -0.09784244, 0.36630486, 0.42891027], [-0.16362532, 0.76847555, 0.61860495, 0.41507145], [ 0., 0., 0., 1. ]] ]
    for cam in range(len(camera_transformations)):
        scene.camera_transform = camera_transformations[cam]

        png = scene.save_image()
        with open(output_dir+file_name+"_"+str(cam)+".png", 'wb') as f:
            f.write(png)
            f.close()



if __name__ == '__main__':
    
    output_dir = './output/pv_selection_weighted_ibs_sources/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_frame = pd.DataFrame(columns=['strategy', 'obj', 'env','sampling_ibs_strategy','probability_density_function', 'rate_samples_in_ibs', 'rate_random_samples',
                                       'ibs_sampled_points', 'random_num_generated', 'exec_time'])

    to_test = pd.DataFrame([
            {
            'env': "hanging-rack", 
            'obj': "umbrella", 
            'tri_mesh_env': "./data/interactions/hanging-rack_umbrella/hanging-rack.ply", 
            'tri_mesh_obj': "./data/interactions/hanging-rack_umbrella/umbrella.ply", 
            'tri_mesh_ibs_segmented': "./data/interactions/hanging-rack_umbrella/ibs_hanging-rack_umbrella_sampled_600_resamplings_2.ply",
            'o3d_cloud_sources_ibs': "./data/interactions/hanging-rack_umbrella/env_samplings_ibs_hanging-rack_umbrella_sample_600_resamplings_2.pcd"
            }
            ,
            {'env': "table", 
            'obj': "bowl", 
            'tri_mesh_env': './data/interactions/table_bowl/table.ply', 
            'tri_mesh_obj': './data/interactions/table_bowl/bowl.ply', 
            'tri_mesh_ibs_segmented': './data/interactions/table_bowl/ibs_table_bowl_sampled_600_resamplings_2.ply',
            'o3d_cloud_sources_ibs': "./data/interactions/table_bowl/env_samplings_ibs_table_bowl_sample_600_resamplings_2.pcd"
            }
            ,
            {
            'env': "motorbike", 
            'obj': "rider", 
            'tri_mesh_env': "./data/interactions/motorbike_rider/motorbike.ply", 
            'tri_mesh_obj': "./data/interactions/motorbike_rider/biker.ply", 
            'tri_mesh_ibs_segmented': "./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_600_resamplings_2.ply",
            'o3d_cloud_sources_ibs': "./data/interactions/motorbike_rider/env_samplings_ibs_motorbike_biker_sample_600_resamplings_2.pcd"
            }
            ])
    


    for index_label, row_series in to_test.iterrows():
        env = to_test.at[index_label,'env']
        obj = to_test.at[index_label,'obj']
        tri_mesh_env = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_env'] )
        tri_mesh_obj = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_obj'] )
        tri_mesh_ibs_segmented = trimesh.load_mesh( to_test.at[index_label,'tri_mesh_ibs_segmented'] )
        o3d_cloud_sources_ibs = o3d.io.read_point_cloud( to_test.at[index_label,'o3d_cloud_sources_ibs'] )
        np_cloud_env = np.asarray(o3d_cloud_sources_ibs.points)

        rate_ibs_samples = 5
        rate_generated_random_numbers = 500
        
        #####  POISSON DISC SAMPLING ON IBS SURFACE

        start = time.time()
        sampler_poissondisc_random = PoissonDiscRandomSampler(tri_mesh_ibs_segmented, tri_mesh_env, rate_ibs_samples) 
        end = time.time()
        execution_time = end - start

        ibs_sampled_points = rate_ibs_samples * sampler_poissondisc_random.SAMPLE_SIZE
        random_num_generated = sampler_poissondisc_random.SAMPLE_SIZE
        sampling_ibs_strategy = 'poisson disc'
        probability_density_function = 'uniform'

        data_frame.loc[len(data_frame)] = ["ibs_poisson_sampling-chosen_randomlyuniform", obj, env, sampling_ibs_strategy, probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]
        print("POISSON DISC SAMPLING ON IBS SURFACE - CHOSEN RANDOMLY")
        #visualize( sampler_poissondisc_random )
        scene = generate_scene(sampler_poissondisc_random)
        saveOutput(sampler_poissondisc_random, scene, output_dir, env+"_"+obj+"_ibs_poisson_sampling-chosen_randomlyuniform" )



        start = time.time()
        sampler_poissondisc_weighted = PoissonDiscWeightedSampler(tri_mesh_ibs_segmented, 
                                                                    tri_mesh_env,  
                                                                    rate_ibs_samples=rate_ibs_samples, 
                                                                    rate_generated_random_numbers=rate_generated_random_numbers)
        end = time.time()
        execution_time = end - start
        ibs_sampled_points = rate_ibs_samples * sampler_poissondisc_weighted.SAMPLE_SIZE
        random_num_generated = sampler_poissondisc_weighted.np_cloud_ibs.shape[0] * sampler_poissondisc_weighted.rate_generated_random_numbers
        sampling_ibs_strategy = 'poisson disc'
        probability_density_function = 'weighted'
        data_frame.loc[len(data_frame)] = ["ibs_poisson_sampling-chosen_weights", obj, env, sampling_ibs_strategy , probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]
        print("POISSON DISC SAMPLING ON IBS SURFACE - CHOSEN BY WEIGTHS")
        #visualize( sampler_poissondisc_weighted )
        scene = generate_scene(sampler_poissondisc_weighted)
        saveOutput(sampler_poissondisc_weighted, scene, output_dir, env+"_"+obj+"_ibs_poisson_sampling-chosen_weights" )



        #####  SAMPLING ON IBS VERTICES

        start = time.time()
        sampler_meshvertices_random =  OnVerticesRandomSampler(tri_mesh_ibs_segmented, tri_mesh_env)
        end = time.time()
        execution_time = end - start
        ibs_sampled_points = sampler_meshvertices_random.np_cloud_ibs.shape[0]
        random_num_generated = sampler_meshvertices_random.SAMPLE_SIZE
        sampling_ibs_strategy = 'ibs vertices'
        probability_density_function = 'uniform'
        data_frame.loc[len(data_frame)] = ["ibs_vertices_sampling-chosen_randomlyuniform", obj, env, sampling_ibs_strategy , probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]
        print("SAMPLING ON IBS SURFACE VERTICES - CHOSEN RANDOMLY")
        #visualize( sampler_meshvertices_random )
        scene = generate_scene( sampler_meshvertices_random )
        saveOutput( sampler_meshvertices_random, scene, output_dir, env+"_"+obj+"_ibs_vertices_sampling-chosen_randomlyuniform" )





        start = time.time()
        sampler_ibs_vertices_weighted =  OnVerticesWeightedSampler(tri_mesh_ibs_segmented, tri_mesh_env,
                                                                      rate_generated_random_numbers=rate_generated_random_numbers)
        end = time.time()
        execution_time = end - start
        ibs_sampled_points = sampler_ibs_vertices_weighted.np_cloud_ibs.shape[0]
        random_num_generated = sampler_ibs_vertices_weighted.np_cloud_ibs.shape[0] * sampler_ibs_vertices_weighted.rate_generated_random_numbers
        sampling_ibs_strategy = 'ibs vertices'
        probability_density_function = 'weighted'
        data_frame.loc[len(data_frame)] = ["ibs_vertices_sampling-chosen_weights", obj, env, sampling_ibs_strategy , probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]
        print("SAMPLING ON IBS SURFACE VERTICES - CHOSEN BY WEIGTHS")  
        #visualize( sampler_ibs_vertices_weighted )
        scene = generate_scene( sampler_ibs_vertices_weighted )
        saveOutput( sampler_ibs_vertices_weighted, scene, output_dir, env+"_"+obj+"_ibs_vertices_sampling-chosen_weights" )
        
        



        #####  SAMPLING ON POINT THAT GENERATE THE IBS SURFACE
        start = time.time()
        sampler_ibs_srcs_randomly =  OnGivenPointCloudRandomSampler(tri_mesh_ibs_segmented, tri_mesh_env, 
                                                                      np_input_cloud = np_cloud_env)
        end = time.time()
        execution_time = end - start
        ibs_sampled_points = sampler_ibs_srcs_randomly.np_cloud_env.shape[0]
        random_num_generated = sampler_ibs_srcs_randomly.SAMPLE_SIZE
        sampling_ibs_strategy = 'ibs sources'
        probability_density_function = 'uniform'
        data_frame.loc[len(data_frame)] = ["ibs_srcs_sampling-chosen_randomlyuniform", obj, env, sampling_ibs_strategy , probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]

        print("SAMPLING ON POINTS THAT GENERATE THE IBS - CHOSEN BY RANDOMLY")  
        #visualize( sampler_ibs_srcs_randomly )
        scene = generate_scene( sampler_ibs_srcs_randomly )
        saveOutput( sampler_ibs_srcs_randomly, scene, output_dir, env+"_"+obj+"_ibs_srcs_sampling-chosen_randomlyuniform" )




        start = time.time()
        sampler_ibs_srcs_weighted =  OnGivenPointCloudWeightedSampler(tri_mesh_ibs_segmented, tri_mesh_env, 
                                                                      np_input_cloud = np_cloud_env,
                                                                      rate_generated_random_numbers=rate_generated_random_numbers)
        end = time.time()
        execution_time = end - start
        ibs_sampled_points = sampler_ibs_srcs_randomly.np_cloud_env.shape[0]
        random_num_generated = sampler_ibs_srcs_weighted.np_cloud_ibs.shape[0] * sampler_ibs_srcs_weighted.rate_generated_random_numbers
        sampling_ibs_strategy = 'ibs sources'
        probability_density_function = 'weighted'
        data_frame.loc[len(data_frame)] = ["ibs_srcs_sampling-chosen_weights", obj, env, sampling_ibs_strategy , probability_density_function, rate_ibs_samples, rate_generated_random_numbers,
                                            ibs_sampled_points, random_num_generated, execution_time]

        print("SAMPLING ON POINTS THAT GENERATE THE IBS - CHOSEN BY WEIGTHS")  
        #visualize( sampler_ibs_srcs_weighted )
        scene = generate_scene( sampler_ibs_srcs_weighted )
        saveOutput( sampler_ibs_srcs_weighted, scene, output_dir, env+"_"+obj+"_ibs_srcs_sampling-chosen_weights" )



   
        
    filename = "%soutput_info.csv" % (output_dir)
    data_frame.to_csv(filename)
