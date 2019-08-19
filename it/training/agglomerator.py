import math
import json
import os
import numpy as np
from open3d import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation


class Agglomerator:

    OUTPURDIR = './output/'
    ORIENTATIONS = 8
    
    def __init__(self, it_trainer):

        orientations = [x*(2*math.pi/self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]
        
        agglomerated_pv_points =[]
        agglomerated_pv_vectors =[]
        agglomerated_pv_vdata = []
        
        pv_vdata = np.zeros((it_trainer.SAMPLE_SIZE,3), np.float64)
        pv_vdata[:,0:2] =np.hstack( (it_trainer.pv_norms.reshape(-1,1), it_trainer.pv_mapped_norms.reshape(-1,1) ) )

        for angle in orientations:
            R = z_rotation(angle)
            agglomerated_pv_points.append( np.dot( it_trainer.pv_points, R ) )
            agglomerated_pv_vectors.append( np.dot( it_trainer.pv_vectors, R ) )
            agglomerated_pv_vdata.append(pv_vdata)
        
        self.agglomerated_pv_points = np.asarray(agglomerated_pv_points).reshape(-1,3)
        self.agglomerated_pv_vectors = np.asarray(agglomerated_pv_vectors).reshape(-1,3)
        self.agglomerated_pv_vdata = np.asarray(agglomerated_pv_vdata).reshape(-1,3)
        self.sample_size = it_trainer.SAMPLE_SIZE
        self.env_normal = it_trainer.env_normal
        


    def save_agglomerated_iT(self, affordances_name, scene_name, object_name):
        
        directory = self.OUTPURDIR + affordances_name + "/"

        if not os.path.exists( directory ):
            os.makedirs( directory )
        
        self.__save_info( directory, affordances_name, scene_name, object_name )

        file_name_pattern = directory + "UNew_"+affordances_name+"_"+object_name+"_descriptor_"+str(self.ORIENTATIONS)
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_points)
        o3d.io.write_point_cloud(file_name_pattern+"_points.pcd", pcd, write_ascii = True)

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_vectors)
        o3d.io.write_point_cloud(file_name_pattern+"_vectors.pcd", pcd, write_ascii = True)

        pcd.points = o3d.utility.Vector3dVector(self.agglomerated_pv_vdata)
        o3d.io.write_point_cloud(file_name_pattern+"_vdata.pcd", pcd, write_ascii = True)


    def __save_info(self, directory, affordances_name, scene_name, object_name):

        data = {}
        data['it_descriptor_version'] = 2.0
        data['scene_name'] = scene_name
        data['object_name'] = object_name
        data['sample_size'] = self.sample_size 
        data['orientations'] = self.ORIENTATIONS
        data['env_normal'] = ( str( self.env_normal[0][0] ) + ',' + 
                             str( self.env_normal[0][1] ) + "," + 
                             str( self.env_normal[0][2] ) )
        #data['reference'] = {}
        #data['reference']['idxRefIBS'] = 8  
        #data['reference']['refPointIBS'] = '8,8,8'
        #data['scene_point'] = {}
        #data['scene_point']['idxScenePoint'] = 9
        #data['scene_point']['refPointScene'] = '9,9,9'
        #data['ibs_point_vector'] = {}
        #data['ibs_point_vector']['idx_ref_ibs'] = 10
        #data['ibs_point_vector']['vect_scene_to_ibs'] = '10,10,10'
        #data['obj_point_vector'] = {}
        #data['obj_point_vector']['idx_ref_object'] = 11
        #data['obj_point_vector']['vect_scene_to_object'] = '11,11,11'
        
        with open( directory + affordances_name + '_' + object_name + '.json', 'w' ) as outfile:
            json.dump(data, outfile)


