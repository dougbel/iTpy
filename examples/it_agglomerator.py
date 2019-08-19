import numpy as np

import open3d as o3d
import trimesh

from it.training.trainer import Trainer
from it.training.agglomerator import Agglomerator


if __name__ == '__main__':

    od3_cloud_ibs = o3d.io.read_point_cloud('./data/pv/cloud_ibs.pcd')
    
    np_cloud_ibs = np.asarray( od3_cloud_ibs.points  )


    tri_mesh_env = trimesh.load_mesh('./data/table.ply')
    
    trainer = Trainer( np_cloud_ibs, tri_mesh_env )

    agg = Agglomerator(trainer)

    agg.save_agglomerated_iT( "Place", "table", "bowl" )