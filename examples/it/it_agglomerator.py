import numpy as np
import pandas as pd

import open3d as o3d
import trimesh

from it.training.sampler import *
from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'hang'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    tri_mesh_env = trimesh.load_mesh(interaction['tri_mesh_env'][0])
    tri_mesh_obj = trimesh.load_mesh(interaction['tri_mesh_obj'][0])
    tri_mesh_ibs_segmented = trimesh.load_mesh(interaction['tri_mesh_ibs_segmented'][0])
    np_cloud_env = np.asarray(o3d.io.read_point_cloud(interaction['o3d_cloud_sources_ibs'][0]).points)
    rate_generated_random_numbers = 500

    sampler_ibs_srcs_weighted = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    trainer = Trainer(tri_mesh_ibs_segmented, tri_mesh_env, sampler_ibs_srcs_weighted)

    agg = Agglomerator(trainer)

    agg.save_agglomerated_iT("Hang", "hanging-rack", "umbrella", tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented)

    print("Finished")