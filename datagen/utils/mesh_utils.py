"""Mesh utilities.

Adapted from the code written by Viraj Mehta and Yurong You.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def read_from_obj(filename):
    """Read the mesh from an obj file.
    """
    vertices = []
    faces = []

    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('UTF-8').strip()
            try:
                vals = line.split()
                if vals[0] == 'v':
                    # Add vertex.
                    vertex = [float(x) for x in vals[1:4]]
                    vertices.append(vertex)
                elif vals[0] == 'f':
                    # Add face.
                    vi = []
                    if vals[1].find('/') == -1:
                        vi = [int(x) - 1 for x in vals[1:]]
                    else:
                        for j in range(1, len(vals)):
                            # Break up like by / to read vert inds, tex coords,
                            # and normal inds.
                            val = vals[j]
                            tokens = val.split('/')
                            for i in range(len(tokens)):
                                if i == 0:
                                    vi.append(int(tokens[i]) - 1)
                                else:
                                    # We omit the texture coordinates and
                                    # normals in the file.
                                    pass
                    faces.append(vi)
            except Exception as e:
                print(e)
                pass

    # TODO(virajm): Can you explain why we need to take the reverse? --by Kuan.
    faces.reverse()

    return np.array(vertices), np.array(faces)


def write_to_obj(filename, vertices, faces):

    faces += 1

    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write('v {0:.6f} {1:.6f} {2:.6f}\n'.format(
                vertex[0], vertex[1], vertex[2]))

        for face in faces:
            f.write('f {0:d} {1:d} {2:d}\n'.format(
                face[0], face[1], face[2]))


def compute_volume(vertices, faces):
    """Compute the volume of the mesh.
    """
    total_volume = 0.0

    for face in faces:
        v0 = vertices[face[0], :]
        v1 = vertices[face[1], :]
        v2 = vertices[face[2], :]
        face_signed_volume = v0.dot(np.cross(v1, v2)) / 6.0
        total_volume += face_signed_volume

    # Correct for flipped faces.
    if total_volume < 0:
        total_volume = -total_volume

    return total_volume


def compute_surface_area(vertices, faces):
    """Compute the surface area of the meash.
    """
    total_area = 0.0

    for face in faces:
        v0 = vertices[face[0], :]
        v1 = vertices[face[1], :]
        v2 = vertices[face[2], :]
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        total_area += face_area

    return total_area


def compute_centroid(vertices, faces):
    """Compute the centroid of the mesh.
    """
    total_area = 0
    centroid = np.zeros((3))

    for face in faces:
        v0 = vertices[face[0], :]
        v1 = vertices[face[1], :]
        v2 = vertices[face[2], :]
        face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        centroid += v0 * face_area
        centroid += v1 * face_area
        centroid += v2 * face_area
        total_area += face_area

    return centroid / (total_area * 3)
