import math
import json
import copy
import numpy as np

from transforms3d.affines import compose

from it.testing.deglomerator import Deglomerator


class Results:
    distances = None
    distances_summary = None
    missed = None
    raw_distances = None

    def __init__(self, distances, resumed_distances, missed, raw_distances):
        self.distances = distances
        self.distances_summary = resumed_distances
        self.missed = missed
        self.raw_distances = raw_distances



class Analyzer:
    results = None

    def __init__(self, idx_ray, calculated_intersections, num_it_to_test, influence_radius, num_orientations,
                 expected_intersections):
        self.idx_ray = idx_ray
        self.calculated_intersections = calculated_intersections
        self.num_it_to_test = num_it_to_test
        self.influence_radius = influence_radius
        self.num_orientations = num_orientations
        self.expected_intersections = expected_intersections

    '''def raw_measured_scores(self):

        distances = self._distances_between_calculated_and_expected_intersections()

        distances_summary, missed = self._resume( distances)

        return distances, distances_summary, missed '''

    def measure_scores(self):
        if self.results is None:
            raw_distances = self._distances_between_calculated_and_expected_intersections()
            # avoid consider distances farther than the influence radius
            distances = self._avoid_distances_farther_influence_radius(raw_distances)
            # summarizing distances and count pv missed by distances or because of lack of intersection
            distances_summary, missed = self._resume( distances)
            self.results = Results(distances, distances_summary, missed, raw_distances)

        return self.results.distances, self.results.distances_summary, self.results.missed

    def best_angle_by_distance_by_affordance(self):
        if self.results is None:
            self.measure_scores()

        best_scores = np.empty((self.num_it_to_test, 4), np.float64)

        index = 0
        for by_affordance_dist in self.results.distances_summary:
            (score, orientation) = min((sco, ori) for ori, sco in enumerate(by_affordance_dist))
            angle = (2 * math.pi / self.num_orientations) * orientation
            missing =  self.results.missed[index][orientation]
            best_scores[index] = [orientation, angle, score, missing]
            index += 1

        return best_scores

    def calculated_pvs_intersection(self, num_interaction, orientation):
        num_pv_by_interaction = (self.expected_intersections.shape[0] / self.num_it_to_test)
        num_pv_by_orientation = num_pv_by_interaction / self.num_orientations

        idx_from = num_interaction * num_pv_by_interaction + orientation * num_pv_by_orientation
        idx_to = idx_from + num_pv_by_orientation

        temp_pv_intersections = np.empty(self.expected_intersections.shape)
        temp_pv_intersections[:] = math.nan
        temp_pv_intersections[self.idx_ray] = self.calculated_intersections
        idx_intersected = [ray for ray in self.idx_ray if ray >= idx_from and ray < idx_to]
        pv_intersections = temp_pv_intersections[idx_intersected]
        return pv_intersections

    def _count_nan(self, a):
        return len(a[np.isnan(a)])

    def _resume(self, all_distances):
        resumed_distances = np.array(list(map(np.nansum,
                                              np.split(
                                                  all_distances,
                                                  self.num_it_to_test * self.num_orientations
                                              )))).reshape(self.num_it_to_test, self.num_orientations)

        resumed_missed = np.array(list(map(self._count_nan,
                                   np.split(
                                       all_distances,
                                       self.num_it_to_test * self.num_orientations
                                   )))).reshape(self.num_it_to_test, self.num_orientations)

        return resumed_distances, resumed_missed

    def _distances_between_calculated_and_expected_intersections(self):
        # calculated offline during training iT
        trained_intersections = self.expected_intersections[self.idx_ray]

        intersections_distances = np.linalg.norm(self.calculated_intersections - trained_intersections, axis=1)

        all_distances = np.empty(len(self.expected_intersections))
        all_distances[:] = math.nan

        for index, raw_distance in zip(self.idx_ray, intersections_distances):
            all_distances[index] = raw_distance

        return all_distances

    def _avoid_distances_farther_influence_radius(self, all_distances):
        filter_distances = np.copy(all_distances)
        # avoid consider distances farther than the influence radius
        pv_by_interaction = int(self.expected_intersections.shape[0] / self.num_it_to_test)
        for interaction in range(self.num_it_to_test):
            idx_from = pv_by_interaction * interaction
            idx_to = idx_from + pv_by_interaction
            to_check = filter_distances[idx_from:idx_to]
            to_check[to_check > self.influence_radius[interaction]] = math.nan
            filter_distances[idx_from:idx_to] = to_check

        return filter_distances