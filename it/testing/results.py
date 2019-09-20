import math
import json
import copy
import numpy as np

from transforms3d.affines import compose

from it.testing.deglomerator import Deglomerator


class Results:
    def __init__(self, all_distances, resumed_distances, missed):
        self.all_distances = all_distances
        self.resumed_distances = resumed_distances
        self.missed = missed


class Analizer:
    results = None

    def __init__(self, idx_ray, calculated_intersections, num_it_to_test, num_orientations, expected_intersections):
        self.idx_ray = idx_ray
        self.calculated_intersections = calculated_intersections
        self.num_it_to_test = num_it_to_test
        self.num_orientations = num_orientations
        self.expected_intersections = expected_intersections

    def measure_scores(self):

        # calculated offline during training iT
        trained_intersections = self.expected_intersections[self.idx_ray]

        intersections_distances = np.linalg.norm(self.calculated_intersections - trained_intersections, axis=1)

        all_distances = np.empty(len(self.expected_intersections))
        all_distances[:] = 'NaN'

        for index, raw_distance in zip(self.idx_ray, intersections_distances):
            all_distances[index] = raw_distance

        resumed_distances = np.array(list(map(np.nansum,
                                              np.split(
                                                  all_distances,
                                                  self.num_it_to_test * self.num_orientations
                                              )))).reshape(self.num_it_to_test, self.num_orientations)

        missed = np.array(list(map(self._count_nan,
                                   np.split(
                                       all_distances,
                                       self.num_it_to_test * self.num_orientations
                                   )))).reshape(self.num_it_to_test, self.num_orientations)
        self.results = Results(all_distances, resumed_distances, missed)
        return self.results.all_distances, self.results.resumed_distances, self.results.missed

    def best_angle_by_distance_by_affordance(self):
        if self.results is None:
            self.measure_scores()

        best_scores = np.empty((self.num_it_to_test, 3), np.float64)

        index = 0
        for by_affordance_dist in self.results.resumed_distances:
            (score, orientation) = min((sco, ori) for ori, sco in enumerate(by_affordance_dist))
            angle = (2 * math.pi / self.num_orientations) * orientation
            best_scores[index] = [orientation, angle, score]
            index += 1

        return best_scores

    def calculated_pvs_intersection(self, num_interaction, orientation):
        num_pv_by_interaction = (self.expected_intersections.shape[0] / self.num_it_to_test)
        num_pv_by_orientation = num_pv_by_interaction / self.num_orientations

        idx_from = num_interaction * num_pv_by_interaction + orientation * num_pv_by_orientation
        idx_to = idx_from + num_pv_by_orientation

        temp_pv_intersections = np.empty(self.expected_intersections.shape)
        temp_pv_intersections[:] = 'NaN'
        temp_pv_intersections[self.idx_ray] = self.calculated_intersections
        idx_intersected = [ray for ray in self.idx_ray if ray >= idx_from and ray < idx_to]
        pv_intersections = temp_pv_intersections[idx_intersected]
        return pv_intersections

    def _count_nan(self, a):
        return len(a[np.isnan(a)])
