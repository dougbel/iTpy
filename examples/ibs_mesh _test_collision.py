import time
import numpy as np
from os import remove

import trimesh

import it.util as util
from it.training.ibs import IBSMesh


if __name__ == '__main__':

    tri_mesh_obj = trimesh.load_mesh("./data/motorbike_rider/biker.ply")

    obj_min_bound = np.asarray(tri_mesh_obj.vertices).min(axis=0)
    obj_max_bound = np.asarray(tri_mesh_obj.vertices).max(axis=0)

    tri_mesh_env = trimesh.load_mesh('./data/motorbike_rider/motorbike.ply')

    extension = np.linalg.norm(obj_max_bound-obj_min_bound)
    middle_point = (obj_max_bound+obj_min_bound)/2

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(
        tri_mesh_env, middle_point, extension)

    # calculating cropping sphere parameters
    radio = np.linalg.norm(obj_max_bound - obj_min_bound)
    np_pivot = np.asarray(obj_max_bound + obj_min_bound) / 2

    in_collision = True
    resamplings = 2
    sampled_points = 600

    while in_collision :
        ibs_calculator = IBSMesh(tri_mesh_env_segmented,  tri_mesh_obj, sampled_points, resamplings)
        tri_mesh_ibs = ibs_calculator.get_trimesh()
        tri_mesh_ibs_segmented = util.slide_mesh_by_sphere( tri_mesh_ibs, np_pivot, radio)

        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object('environment',tri_mesh_env)
        collision_tester.add_object('object',tri_mesh_obj)

        in_collision = collision_tester.in_collision_single(tri_mesh_ibs)

        if in_collision :
            sampled_points = sampled_points + 200
        
        #getting sampled point in environment and object used to generate the IBS surface
        np_env_sampled_points = ibs_calculator.points[ : ibs_calculator.size_cloud_env ]
        np_obj_sampled_points = ibs_calculator.points[ ibs_calculator.size_cloud_env : ]

        size_env_sampled_points =  np_env_sampled_points.shape[0]
        size_obj_sampled_points =  np_obj_sampled_points.shape[0]

        print (" original_points: ", sampled_points, " final_env_points: " , size_env_sampled_points, " final_obj_points: ",  size_obj_sampled_points )  
        print("In collission: " + str(in_collision)+"\n")
        print("----------------------------------")
        
        # VISUALIZATION
        tri_mesh_obj.visual.face_colors = [0, 255, 0, 70]
        tri_mesh_env_segmented.visual.face_colors = [255,0,0,100]
        tri_mesh_ibs_segmented.visual.face_colors = [0,0,255,40]

        visualizer2 = trimesh.Scene( [ 
                                    trimesh.points.PointCloud(np_obj_sampled_points, colors=[0,255,0,255] ), 
                                    trimesh.points.PointCloud(np_env_sampled_points, colors=[255,0,0,255] ), 
                                    tri_mesh_obj,
                                    tri_mesh_env_segmented,
                                    tri_mesh_ibs_segmented#,
                                    
                                    ] )
        # display the environment with callback
        visualizer2.show()
