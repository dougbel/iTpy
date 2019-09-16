import json
import numpy as np
from open3d import open3d as o3d


class Deglomerator:

    def __init__(self, working_path, affordance_name, object_name):
        print(affordance_name + ' ' + object_name)
        self.affordance_name = affordance_name
        self.object_name = object_name
        self.__working_path = working_path
        self.__read_definition()
        self.__readAgglomeratedDescriptor()

    def __read_definition(self):
        self.__definition_file = self.__working_path + '/' + self.affordance_name + "_" + self.object_name + ".json"
        with open(self.__definition_file) as jsonfile:
            self.__definition = json.load(jsonfile)
            self.num_orientations = int(self.__definition['orientations'])
            self.sample_size = int(self.__definition['sample_size'])

    def __readAgglomeratedDescriptor(self):
        base_nameU = self.__working_path + "/UNew_" + self.affordance_name + "_" + self.object_name + "_descriptor_" + str(
            self.__definition['orientations'])

        self.pv_points = np.asarray(o3d.io.read_point_cloud(base_nameU + "_points.pcd").points)
        self.pv_vectors = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vectors.pcd").points)
        self.pv_data = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vdata.pcd").points)
