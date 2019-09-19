import math
import json
import copy
import numpy as np

from transforms3d.affines import compose

from it.testing.deglomerator import Deglomerator


class Tester:

    def __init__(self, path, file):
        self.working_path = path
        self.configuration_file = file
        self.last_position = np.zeros(3)
        self.__read_json()


    def __read_json(self):
        with open( self.configuration_file ) as jsonfile:
            data =  json.load(jsonfile)

        self.num_it_to_test =len( data['interactions'] )
        self.num_orientations = data['parameters']['num_orientations']
        self.num_pv = data['parameters']['num_pv']

        increments = self.num_orientations * self.num_pv
        amount_data = self.num_it_to_test * increments

        self.compiled_pv_begin = np.empty( (amount_data, 3), np.float64 )
        self.compiled_pv_direction = np.empty( (amount_data, 3), np.float64 )
        self.compiled_pv_data = np.empty( (amount_data, 3), np.float64 )
        self.affordances = []
        self.objs_filenames = []
        self.objs_influence_radios = []

        index1 = 0
        index2 = increments

        for affordance in data['interactions']:
            sub_working_path = self.working_path + "/" + affordance['affordance_name']
            it_descriptor = Deglomerator( sub_working_path, affordance['affordance_name'], affordance['object_name'] )
            self.compiled_pv_begin[index1:index2] = it_descriptor.pv_points
            self.compiled_pv_direction[index1:index2] = it_descriptor.pv_vectors
            self.compiled_pv_data[index1:index2] = it_descriptor.pv_data
            self.affordances.append( [ affordance['affordance_name'] ,affordance['object_name'] ] )
            self.objs_filenames.append(it_descriptor.object_filename())
            self.objs_influence_radios.append(it_descriptor.influence_radio)
            index1 += increments
            index2 += increments
        self.compiled_pv_end = self.compiled_pv_begin + self.compiled_pv_direction


    def measure_scores(self, scene, position):

        #looking for ray intersections
        idx_ray, calculated_intersections = self.intersections_with_scene( scene, position )
        #calculated offline during training iT
        trained_intersections = self.compiled_pv_end[idx_ray]

        intersections_distances = np.linalg.norm(calculated_intersections - trained_intersections, axis=1)

        all_distances = np.empty(len(self.compiled_pv_end))
        all_distances[:] = 'NaN'

        for index, raw_distance in zip(idx_ray, intersections_distances):
            all_distances[index] = raw_distance

        resumed_distances = np.array(list(
                                            map( np.nansum,
                                                np.split(
                                                    all_distances,
                                                    len(self.affordances)* self.num_orientations
                                                )
                                            )
                                        )
                                    ).reshape(len(self.affordances),self.num_orientations)

        missed = np.array(list(
                                map( self._count_nan,
                                    np.split(
                                        all_distances,
                                        len(self.affordances)* self.num_orientations
                                    )
                                )
                            )
                        ).reshape(len(self.affordances),self.num_orientations)

        return all_distances, resumed_distances, missed



    def best_angle_by_affordance(self, scene, position):
        __, resumed_distances, missed = self.measure_scores(scene, position)
        best_scores = np.empty( (len(resumed_distances),3), np.float64 )

        index = 0
        for by_affordance_dist in resumed_distances:
            (score,orientation) = min((sco,ori) for ori,sco in enumerate(by_affordance_dist))
            angle = (2 * math.pi / self.num_orientations  ) * orientation
            best_scores[index] = [orientation,angle,score]
            index +=1

        return best_scores



    def intersections_with_scene(self, scene, position):
        translation = np.asarray(position) - self.last_position
        self.compiled_pv_begin += translation
        self.compiled_pv_end += translation
        self.last_position = position
        #looking for the nearest ray intersections
        ( __,
         idx_ray,
         intersections) = scene.ray.intersects_id(
                ray_origins = self.compiled_pv_begin,
                ray_directions = self.compiled_pv_direction,
                return_locations=True,
                multiple_hits=False )

        return idx_ray, intersections

    def _count_nan(self, a):
        return len(a[np.isnan(a)])