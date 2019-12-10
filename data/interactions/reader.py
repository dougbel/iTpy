import trimesh
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # interactions_data = pd.read_csv("./data/interactions/motorbike_rider/")

    to_read = 'ride'
    # interaction = interactions_data[interactions_data['interaction'] == to_read]

    tri_mesh_env = trimesh.load_mesh("./data/interactions/motorbike_rider/motorbike.ply")
    tri_mesh_obj = trimesh.load_mesh("./data/interactions/motorbike_rider/rider.ply")
    tri_mesh_ibs_segmented = trimesh.load_mesh("./data/interactions/motorbike_rider/ibs_motorbike_biker_sampled_600_resamplings_2.ply")

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 150, 100]

    visualizer = trimesh.Scene([
        tri_mesh_obj,
        tri_mesh_env,
        tri_mesh_ibs_segmented,
    ])

    visualizer.show()
