import time
import numpy as np
import pandas as pd
import os

import trimesh

import it.util as util
from it.training.ibs import IBSMesh

if __name__ == '__main__':
    '''
    Shows the execution of the IBS calculator, using meshes and the developed sampling strategies.  This execution also 
    shows some characteristics of the surface (winding consistency, convexity and watertighty)
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

    ################################
    # GENERATING AND SEGMENTING IBS MESH
    ################################
    start = time.time()  # timing execution

    tri_mesh_ibs = ibs_calculator.get_trimesh()
    # tri_mesh_ibs = tri_mesh_ibs.subdivide()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)

    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro, 16)

    end = time.time()  # timing execution
    print(end - start, " seconds on IBS MESH GENERATION AND SEGMENTATION")  # timing execution

    # tri_mesh_ibs_segmented.export("ibs_mesh_segmented.ply","ply")

    print("is convex: " + str(tri_mesh_ibs.is_convex))
    print("is empty: " + str(tri_mesh_ibs.is_empty))
    print("is watertight: " + str(tri_mesh_ibs.is_watertight))
    print("is winding consistent: " + str(tri_mesh_ibs.is_winding_consistent))

    ################################
    # VISUALIZATION
    ################################

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 150, 100]
    tri_mesh_ibs.visual.face_colors = [0, 0, 150, 100]

    visualizer = trimesh.Scene([
        tri_mesh_env,
        tri_mesh_obj,
        tri_mesh_ibs_segmented
    ])

    # display the environment with callback
    visualizer.show(flags={'cull': False, 'wireframe': False, 'axis': False})
