import json
from os import path
import numpy as np
import open3d as o3d


class Deglomerator:

    def __init__(self, working_path, affordance_name, object_name):
        print(affordance_name + ' ' + object_name)
        self.affordance_name = affordance_name
        self.object_name = object_name
        self.__working_path = working_path
        self.__read_definition()
        self.__readAgglomeratedDescriptor()

    def __read_definition(self):
        self._definition_file = self.__working_path + '/' + self.affordance_name + "_" + self.object_name + ".json"
        with open(self._definition_file) as jsonfile:
            self._definition = json.load(jsonfile)
            self.num_orientations = int(self._definition['orientations'])
            self.sample_size = int(self._definition['sample_size'])
            self.influence_radio = self._definition['max_distances']['obj_influence_radio']

    def __readAgglomeratedDescriptor(self):
        base_nameU = self.__working_path + "/UNew_" + self.affordance_name + "_" + self.object_name + "_descriptor_" + str(
            self._definition['orientations'])

        self.pv_points = np.asarray(o3d.io.read_point_cloud(base_nameU + "_points.pcd").points)
        self.pv_vectors = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vectors.pcd").points)
        self.pv_data = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vdata.pcd").points)

    def object_filename(self):
        obj_filename = path.join(self.__working_path, self.affordance_name + "_" + self.object_name + "_object.ply")
        return obj_filename
