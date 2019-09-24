import math
import json
import copy
import numpy as np

from transforms3d.affines import compose

from it.testing.deglomerator import Deglomerator
from it.testing.results import Analizer, Results


class Tester:

    def __init__(self, path, file):
        self.working_path = path
        self.configuration_file = file
        self.last_position = np.zeros(3)
        self.__read_json()

    def __read_json(self):
        with open(self.configuration_file) as jsonfile:
            data = json.load(jsonfile)

        self.num_it_to_test = len(data['interactions'])
        self.num_orientations = data['parameters']['num_orientations']
        self.num_pv = data['parameters']['num_pv']

        increments = self.num_orientations * self.num_pv
        amount_data = self.num_it_to_test * increments

        self.compiled_pv_begin = np.empty((amount_data, 3), np.float64)
        self.compiled_pv_direction = np.empty((amount_data, 3), np.float64)
        self.compiled_pv_data = np.empty((amount_data, 3), np.float64)
        self.affordances = []
        self.objs_filenames = []
        self.objs_influence_radios = []

        index1 = 0
        index2 = increments

        for affordance in data['interactions']:
            sub_working_path = self.working_path + "/" + affordance['affordance_name']
            it_descriptor = Deglomerator(sub_working_path, affordance['affordance_name'], affordance['object_name'])
            self.compiled_pv_begin[index1:index2] = it_descriptor.pv_points
            self.compiled_pv_direction[index1:index2] = it_descriptor.pv_vectors
            self.compiled_pv_data[index1:index2] = it_descriptor.pv_data
            self.affordances.append([affordance['affordance_name'], affordance['object_name']])
            self.objs_filenames.append(it_descriptor.object_filename())
            self.objs_influence_radios.append(it_descriptor.influence_radio)
            index1 += increments
            index2 += increments
        self.compiled_pv_end = self.compiled_pv_begin + self.compiled_pv_direction

    def get_analyzer(self, scene, position):
        translation = np.asarray(position) - self.last_position
        self.compiled_pv_begin += translation
        self.compiled_pv_end += translation
        self.last_position = position
        # looking for the nearest ray intersections
        (__,
         idx_ray,
         intersections) = scene.ray.intersects_id(
            ray_origins=self.compiled_pv_begin,
            ray_directions=self.compiled_pv_direction,
            return_locations=True,
            multiple_hits=False)

        return Analizer(idx_ray, intersections, self.num_it_to_test, self.objs_influence_radios, self.num_orientations,
                        self.compiled_pv_end)
