import time
import numpy as np
import pandas as pd
import os

import trimesh

import it.util as util
from it.training.ibs import IBSMesh

if __name__ == '__main__':
    '''
    Shows the IBS calculated using meshes. Developed strategies for sampling on the object and environment surfaces 
    allow that IBS avoids pierce them.     
    '''

    interactions_data = pd.read_csv("./data/interactions/interaction.csv")
    to_test = 'place'
    interaction = interactions_data[interactions_data['interaction'] == to_test]
    tri_mesh_env = trimesh.load_mesh(os.path.join(interaction.iloc[0]['directory'], interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(os.path.join(interaction.iloc[0]['directory'], interaction.iloc[0]['tri_mesh_obj']))
    obj_name = interaction.iloc[0]['obj']
    env_name = interaction.iloc[0]['env']

    influence_radio_ratio = 1.5

    extension, middle_point = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    start = time.time()  # timing execution
    ibs_calculator = IBSMesh(600, 2)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)
    end = time.time()  # timing execution
    print(end - start, " seconds on IBS calculation (600 original points)")  # timing execution

    ####################################################################################################################
    # 1. IBS VISUALIZATION
    edges_from, edges_to = util.get_edges(ibs_calculator.vertices, ibs_calculator.ridge_vertices)

    visualizer = trimesh.Scene([
        trimesh.load_path(np.hstack((edges_from, edges_to)).reshape(-1, 2, 3)),
        # trimesh.points.PointCloud( np_cloud_obj_poisson , colors=[0,0,255,255] ),
        tri_mesh_obj,
    ])

    visualizer.show()

    ####################################################################################################################
    # 2. CROPPED VISUALIZATION MESH AND POINT CLOUD (IBS)

    # extracting point no farther than the principal sphere

    radio, np_pivot = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    [idx_extracted, np_ibs_vertices_extracted] = util.extract_cloud_by_sphere(ibs_calculator.vertices, np_pivot, radio)

    # cutting edges in the polygon mesh
    edges_from, edges_to = util.get_edges(ibs_calculator.vertices, ibs_calculator.ridge_vertices, idx_extracted)

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_mesh_env_segmented.visual.face_colors = [100, 100, 100, 100]

    visualizer2 = trimesh.Scene([  # trimesh.points.PointCloud( ibs_calculator.cloud_ibs, colors=[0,0,255,255] ),
        # trimesh.points.PointCloud( np_cloud_obj_poisson , colors=[0,0,255,255] ),
        tri_mesh_obj,
        tri_mesh_env_segmented,
        trimesh.points.PointCloud(ibs_calculator.points, colors=[0, 0, 0, 100]),
        trimesh.points.PointCloud(np_ibs_vertices_extracted, colors=[0, 0, 255, 255]),
        trimesh.load_path(np.hstack((edges_from, edges_to)).reshape(-1, 2, 3))
    ])

    # display the environment with callback
    visualizer2.show()
