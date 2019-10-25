import time
import pandas as pd
import numpy as np
import os

import trimesh
import open3d as o3d

import it.util as util
from it.training.ibs import IBSMesh

if __name__ == '__main__':
    '''
    Test, generate and store IBS generated with different parameters. 
    The parameter influence_radio_ratio is established to 2 in order to crop surfaces and keep most information of IBS 
    to observe. 
    '''

    interactions_data = pd.read_csv("./data/interactions/interaction.csv")
    to_test = 'ride'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    tri_mesh_env = trimesh.load_mesh(
        os.path.join(interaction.iloc[0]['directory'], interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(
        os.path.join(interaction.iloc[0]['directory'], interaction.iloc[0]['tri_mesh_obj']))
    obj_name = interaction.iloc[0]['obj']
    env_name = interaction.iloc[0]['env']

    influence_radio_ratio = 2

    extension, middle_point = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    # calculating cropping sphere parameters
    radio, np_pivot = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    # execution parameters
    improve_by_collission = True
    in_collision = True
    resamplings = 2
    original_sample_size = 600
    sampled_points = 600

    data_frame = pd.DataFrame(columns=['obj_sample', 'env_sample', 'resamplings', 'improved_by_collision',
                                       'obj_resampling', 'env_resampling', 'exec_time', 'in_collision',
                                       'collision_points'])

    output_dir = './output/ibs_generate_info_from_interaction/' + env_name + '_' + obj_name + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while in_collision:
        start = time.time()  # timing execution
        ibs_calculator = IBSMesh(sampled_points, resamplings, improve_by_collission)
        ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)
        end = time.time()  # timing execution
        execution_time = end - start

        print("time: ", execution_time)

        tri_mesh_ibs = ibs_calculator.get_trimesh()
        tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, np_pivot, radio)

        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object(env_name, tri_mesh_env)
        collision_tester.add_object(obj_name, tri_mesh_obj)

        in_collision, data = collision_tester.in_collision_single(tri_mesh_ibs, return_data=True)

        # getting sampled point in environment and object used to generate the IBS surface
        np_env_sampled_points = ibs_calculator.points[: ibs_calculator.size_cloud_env]
        np_obj_sampled_points = ibs_calculator.points[ibs_calculator.size_cloud_env:]

        size_env_sampled_points = np_env_sampled_points.shape[0]
        size_obj_sampled_points = np_obj_sampled_points.shape[0]

        print("original_points: ", sampled_points, " final_env_points: ",
              size_env_sampled_points, " final_obj_points: ", size_obj_sampled_points)
        print("In collission: ", in_collision)
        print("----------------------------------")

        data_frame.loc[len(data_frame)] = [sampled_points, sampled_points, resamplings, improve_by_collission,
                                           size_obj_sampled_points, size_env_sampled_points, execution_time,
                                           in_collision, len(data)]

        filename = "%sibs_%s_%s_sampled_%d_resamplings_%d.ply" % (
            output_dir, env_name, obj_name, sampled_points, resamplings)
        tri_mesh_ibs.export(filename, "ply")
        filename = "%sibs_segmented_%s_%s_sampled_%d_resamplings_%d.ply" % (
            output_dir, env_name, obj_name, sampled_points, resamplings)
        tri_mesh_ibs.export(filename, "ply")

        filename = "%sibs_%s_%s_output_info.csv" % (
            output_dir, env_name, obj_name)
        data_frame.to_csv(filename)

        filename = "%senv_samplings_ibs_%s_%s_sample_%d_resamplings_%d.pcd" % (
            output_dir, env_name, obj_name, sampled_points, resamplings)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_env_sampled_points)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

        filename = "%sobj_samplings_ibs_%s_%s_sample_%d_resamplings_%d.pcd" % (
            output_dir, env_name, obj_name, sampled_points, resamplings)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_obj_sampled_points)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

        if in_collision:
            sampled_points = sampled_points + 600

    # VISUALIZATION
    tri_mesh_obj_sampled_points = trimesh.points.PointCloud(np_obj_sampled_points)
    tri_mesh_env_sampled_points = trimesh.points.PointCloud(np_env_sampled_points)

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 70]
    tri_mesh_obj_sampled_points.colors = [0, 255, 0, 255]

    tri_mesh_env_segmented.visual.face_colors = [255, 0, 0, 100]
    tri_mesh_env_sampled_points.colors = [255, 0, 0, 255]

    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 40]

    visualizer2 = trimesh.Scene([
        tri_mesh_obj_sampled_points,
        tri_mesh_env_sampled_points,
        tri_mesh_obj,
        tri_mesh_env_segmented,
        tri_mesh_ibs_segmented
    ])
    # display the environment with callback
    visualizer2.show(flags={'cull': False})

    # VISUALIZATION REAL SOURCED POINTS
    used_points = np.unique(np.asarray(ibs_calculator.ridge_points))
    tri_mesh_ibs_source_points = trimesh.points.PointCloud(ibs_calculator.points[used_points])

    tri_mesh_ibs_source_points.colors = [0, 0, 255, 255]

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 70]
    tri_mesh_env_segmented.visual.face_colors = [255, 0, 0, 100]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 40]

    visualizer2 = trimesh.Scene([
        tri_mesh_ibs_source_points,
        tri_mesh_obj,
        tri_mesh_env_segmented,
        tri_mesh_ibs_segmented
    ])
    # display the environment with callback
    visualizer2.show(flags={'cull': False})
