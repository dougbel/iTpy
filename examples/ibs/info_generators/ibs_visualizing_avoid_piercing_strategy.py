import time
import numpy as np
import os

import trimesh

import it.util as util
from it.training.ibs import IBSMesh


def get_camera(scene):
    np.set_printoptions(suppress=True)
    print(scene.camera_transform)


if __name__ == '__main__':

    env = "table"
    obj = "bowl"

    tri_mesh_obj = trimesh.load_mesh("./data/interactions/table_bowl/bowl.ply")
    tri_mesh_env = trimesh.load_mesh('./data/interactions/table_bowl/table.ply')

    extension, middle_point = util.influence_sphere(tri_mesh_obj)

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    directory = './output/ibs_visualizing_avoid_piercing_strategy/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    points = [150, 300, 600, 1200, 5000, 10000]

    for sampled_points in points:
        start = time.time()  # timing execution
        ibs_calculator = IBSMesh(sampled_points, 4)
        ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)
        end = time.time()  # timing execution

        # getting sampled point in environment and object used to generate the IBS surface
        np_env_sampled_points = ibs_calculator.points[: ibs_calculator.size_cloud_env]
        np_obj_sampled_points = ibs_calculator.points[ibs_calculator.size_cloud_env:]

        size_env_sampled_points = np_env_sampled_points.shape[0]
        size_obj_sampled_points = np_obj_sampled_points.shape[0]

        print("time:", end - start, " original_points: ", sampled_points, " final_env_points: ",
              size_env_sampled_points, " final_obj_points: ", size_obj_sampled_points)

        # extracting point no farther than the principal sphere
        radio, np_pivot = util.influence_sphere(tri_mesh_obj)

        [idx_extracted, np_ibs_vertices_extracted] = util.extract_cloud_by_sphere(ibs_calculator.vertices, np_pivot,
                                                                                  radio)

        # extracting edges in the polygon mesh
        edges_from, edges_to = util.get_edges(ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted)

        # segmenting ibs mesh
        tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(ibs_calculator.get_trimesh(), middle_point, extension)

        # VISUALIZATION
        tri_mesh_obj.visual.face_colors = [0, 255, 0, 70]
        tri_mesh_env_segmented.visual.face_colors = [255, 0, 0, 100]
        tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 40]

        visualizer2 = trimesh.Scene([
            trimesh.points.PointCloud(np_obj_sampled_points, colors=[0, 255, 0, 255]),
            trimesh.points.PointCloud(np_env_sampled_points, colors=[255, 0, 0, 255]),
            # trimesh.points.PointCloud(np_ibs_vertices_extracted, colors=[0,0,255,255] ),
            tri_mesh_obj,
            tri_mesh_env_segmented,
            tri_mesh_ibs_segmented  # ,
            # trimesh.load_path( np.hstack(( edges_from, edges_to)).reshape(-1, 2, 3) )
        ])

        visualizer2.camera_transform = [[0.99865791, -0.0082199, 0.05113514, 0.0030576],
                                        [0.05145371, 0.0448673, -0.997667, -0.11272119],
                                        [0.00590642, 0.99895914, 0.04523003, 0.00018095],
                                        [0., 0., 0., 1., ]]

        png = visualizer2.save_image()
        with open(directory + "horizontal_orig_sampling_" + str(sampled_points) + "_" + env + "_" + str(
                size_env_sampled_points) + "_" + obj + "_" + str(size_obj_sampled_points) + ".png", 'wb') as f:
            f.write(png)
            f.close()

        visualizer2.camera_transform = [[0.99893183, -0.00619928, 0.04579028, 0.00597269],
                                        [0.04550399, 0.30429899, -0.9514891, -0.30608829],
                                        [-0.0080354, 0.9525564, 0.30425604, 0.09243122],
                                        [0., 0., 0., 1., ]]

        png = visualizer2.save_image()
        with open(directory + "diagonal_orig_sampling_" + str(sampled_points) + "_" + env + "_" + str(
                size_env_sampled_points) + "_" + obj + "_" + str(size_obj_sampled_points) + ".png", 'wb') as f:
            f.write(png)
            f.close()

    # display the environment with callback
    visualizer2.show(callback=get_camera)
