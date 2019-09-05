import numpy as np

import open3d as o3d
import trimesh

from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator


if __name__ == '__main__':

    tri_mesh_ibs_segmented = trimesh.load_mesh('./data/pv/ibs_mesh_segmented.ply')

    tri_mesh_ibs = trimesh.load_mesh('./data/pv/ibs_mesh.ply')

    tri_mesh_env = trimesh.load_mesh('./data/interactions/table_bowl/table.ply')

    tri_mesh_obj = trimesh.load_mesh('./data/interactions/table_bowl/bowl.ply')
    
    trainer = Trainer( tri_mesh_ibs_segmented, tri_mesh_env )

    agg = Agglomerator(trainer)

    agg.save_agglomerated_iT( "Place", "table", "bowl", tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented, tri_mesh_ibs )