import trimesh
import numpy as np
import math
import open3d as o3d


def sample_points_poisson_disk(tri_mesh, number_of_points, init_factor=5):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)

    od3_cloud_poisson = o3d.geometry.sample_points_poisson_disk(o3d_mesh, number_of_points, init_factor)

    return np.asarray(od3_cloud_poisson.points)


def slide_mesh_by_bounding_box(tri_mesh, box_center, box_extension):
    max_x_plane = box_center + np.array([box_extension, 0, 0])
    min_x_plane = box_center - np.array([box_extension, 0, 0])
    max_y_plane = box_center + np.array([0, box_extension, 0])
    min_y_plane = box_center - np.array([0, box_extension, 0])
    max_z_plane = box_center + np.array([0, 0, box_extension])
    min_z_plane = box_center - np.array([0, 0, box_extension])

    extracted = tri_mesh.slice_plane(plane_normal=np.array([-1, 0, 0]), plane_origin=max_x_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([1, 0, 0]), plane_origin=min_x_plane)

    extracted = extracted.slice_plane(plane_normal=np.array([0, -1, 0]), plane_origin=max_y_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([0, 1, 0]), plane_origin=min_y_plane)

    extracted = extracted.slice_plane(plane_normal=np.array([0, 0, -1]), plane_origin=max_z_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([0, 0, 1]), plane_origin=min_z_plane)

    return extracted


def slide_mesh_by_sphere(tri_mesh, sphere_center, sphere_ro, level=16):
    # angle with respect X axis and Z
    angle = [x * 2 * math.pi / level for x in range(level)]
    output_mesh = trimesh.Trimesh(vertices=tri_mesh.vertices,
                                  faces=tri_mesh.faces,
                                  process=False)
    for theta in angle:
        for phi in angle:
            sphere_point = np.array([sphere_ro * math.cos(theta) * math.sin(phi),
                                     sphere_ro * math.sin(theta) * math.sin(phi),
                                     sphere_ro * math.cos(phi)])
            plane_origin = sphere_point + sphere_center
            plane_normal = sphere_center - sphere_point
            output_mesh = output_mesh.slice_plane(plane_normal=plane_normal, plane_origin=plane_origin)

    return output_mesh


def extract_cloud_by_bounding_box(np_cloud, box_center, box_extension):

    max_x_plane = box_center[0] + box_extension
    min_x_plane = box_center[0] - box_extension
    max_y_plane = box_center[1] + box_extension
    min_y_plane = box_center[1] - box_extension
    max_z_plane = box_center[2] + box_extension
    min_z_plane = box_center[2] - box_extension

    extracted = [point for point in np_cloud if point[0] > min_x_plane]
    extracted = [point for point in extracted if point[0] < max_x_plane]
    extracted = [point for point in extracted if point[1] > min_y_plane]
    extracted = [point for point in extracted if point[1] < max_y_plane]
    extracted = [point for point in extracted if point[2] > min_z_plane]
    extracted = [point for point in extracted if point[2] < max_z_plane]

    return np.asarray(extracted)


def extract_cloud_by_sphere(np_cloud, np_sphere_centre, sphere_ro):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud)

    pcd_tree = o3d.geometry.KDTreeFlann(o3d_cloud)

    # it returns 1: number of point returned, 2: idx of point returned, 3: distances
    [__, idx, __] = pcd_tree.search_radius_vector_3d(np_sphere_centre, sphere_ro)

    return idx, np_cloud[idx]


def get_edges(vertices, ridge_vertices, idx_extracted=None):
    # cutting faces in the polygon mesh
    faces_idx_ridges_from = []
    faces_idx_ridges_to = []
    for ridge in ridge_vertices:
        if -1 in ridge:
            continue
        if idx_extracted is not None:
            face_in_boundary = True
            for idx_vertex in ridge:
                if idx_vertex not in idx_extracted:
                    face_in_boundary = False
                    break
            if not face_in_boundary:
                continue

        for i in range(-1, len(ridge) - 1):
            faces_idx_ridges_from.append(ridge[i])
            faces_idx_ridges_to.append(ridge[i + 1])

    edges_from = vertices[faces_idx_ridges_from]
    edges_to = vertices[faces_idx_ridges_to]

    return edges_from, edges_to
