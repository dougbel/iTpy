import math
import json
import os
import numpy as np
from open3d import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation


class Agglomerator:
    OUTPUT_DIR = './output/descriptors_repository/'
    ORIENTATIONS = 8

    def __init__(self, it_trainer):

        orientations = [x * (2 * math.pi / self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]

        agglomerated_pv_points = []
        agglomerated_pv_vectors = []
        agglomerated_pv_vdata = []
        agglomerated_normals = []

        self.trainer = it_trainer
        self.sample_size = it_trainer.pv_points.shape[0]

        pv_vdata = np.zeros((self.sample_size, 3), np.float64)
        pv_vdata[:, 0:2] = np.hstack((it_trainer.pv_norms.reshape(-1, 1), it_trainer.pv_mapped_norms.reshape(-1, 1)))

        for angle in orientations:
            rotation = z_rotation(angle)
            agglomerated_pv_points.append(np.dot(it_trainer.pv_points, rotation))
            agglomerated_pv_vectors.append(np.dot(it_trainer.pv_vectors, rotation))
            agglomerated_pv_vdata.append(pv_vdata)
            agglomerated_normals.append(np.dot(it_trainer.normal_env, rotation))

        self.agglomerated_pv_points = np.asarray(agglomerated_pv_points).reshape(-1, 3)
        self.agglomerated_pv_vectors = np.asarray(agglomerated_pv_vectors).reshape(-1, 3)
        self.agglomerated_pv_vdata = np.asarray(agglomerated_pv_vdata).reshape(-1, 3)
        self.agglomerated_normals = np.asarray(agglomerated_normals).reshape(-1, 3)

    def save_agglomerated_iT(self, affordance_name, scene_name, object_name, tri_mesh_env, tri_mesh_obj,
                             tri_mesh_ibs_segmented, tri_mesh_ibs=None):

        directory = self.OUTPUT_DIR + affordance_name + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        self._save_info(directory, affordance_name, scene_name, object_name)

        file_name_pattern = directory + "UNew_" + affordance_name + "_" + object_name + "_descriptor_" + str(
            self.ORIENTATIONS)

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_vectors.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_vdata)
        o3d.io.write_point_cloud(file_name_pattern + "_vdata.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_normals)
        o3d.io.write_point_cloud(file_name_pattern + "_normals_env.pcd", pcd, write_ascii=True)

        file_name_pattern = directory + affordance_name + "_" + object_name
        tri_mesh_ibs_segmented.export(file_name_pattern + "_ibs_mesh_segmented.ply", "ply")
        tri_mesh_env.export(file_name_pattern + "_environment.ply", "ply")
        tri_mesh_obj.export(file_name_pattern + "_object.ply", "ply")
        if tri_mesh_ibs is not None:
            tri_mesh_ibs.export(file_name_pattern + "_ibs_mesh.ply", "ply")

    def _save_info(self, directory, affordance_name, scene_name, object_name):

        data = {}
        data['it_descriptor_version'] = 2.0
        data['scene_name'] = scene_name
        data['object_name'] = object_name
        data['sample_size'] = self.sample_size
        data['orientations'] = self.ORIENTATIONS
        data['trainer'] = self.trainer.get_info()
        # data['reference'] = {}
        # data['reference']['idxRefIBS'] = 8
        # data['reference']['refPointIBS'] = '8,8,8'
        # data['scene_point'] = {}
        # data['scene_point']['idxScenePoint'] = 9
        # data['scene_point']['refPointScene'] = '9,9,9'
        # data['ibs_point_vector'] = {}
        # data['ibs_point_vector']['idx_ref_ibs'] = 10
        # data['ibs_point_vector']['vect_scene_to_ibs'] = '10,10,10'
        # data['obj_point_vector'] = {}
        # data['obj_point_vector']['idx_ref_object'] = 11
        # data['obj_point_vector']['vect_scene_to_object'] = '11,11,11'

        with open(directory + affordance_name + '_' + object_name + '.json', 'w') as outfile:
            json.dump(data, outfile)
