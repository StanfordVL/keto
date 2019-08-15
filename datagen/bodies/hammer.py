"""Hammer generator.
"""

import os
import random

import numpy as np

from bodies.body import Body
from links.link import Link
from links.scad_link import ScadCubeLink
from links.scad_link import ScadCylinderLink
from links.scad_link import ScadPolygonLink
from links.scad_link import ScadBreadLink
from utils.transformations import matrix3_from_euler


FLIP_PROB = 0.5

LATERAL_FRICTION_RANGE = [0.2, 1.0]
SPINNING_FRICTION_RANGE = [0.2, 1.0]
INERTIA_FRICTION_RANGE = [0.2, 1.0]

HANDLE_CONFIG = {
        'mass_range': [0.4, 0.8],
        'size_range': [[0.02, 0.05], [0.02, 0.05], [0.15, 0.30]],
        'lateral_friction_range': LATERAL_FRICTION_RANGE,
        'spinning_friction_range': SPINNING_FRICTION_RANGE,
        'inertia_friction_range': INERTIA_FRICTION_RANGE,
        }

HEAD_CONFIG = {
        'mass_range': [0.8, 1.8],
        'size_range': [[0.03, 0.06], [0.03, 0.06], [0.10, 0.20]],
        'lateral_friction_range': LATERAL_FRICTION_RANGE,
        'spinning_friction_range': SPINNING_FRICTION_RANGE,
        'inertia_friction_range': INERTIA_FRICTION_RANGE,
        }

CMAP = {
        'wood': [0.494, 0.278, 0.164],
        'metal': [0.654, 0.698, 0.761],
        }


def transform_point(point, rotation, translation):
    """Rigid transformation of a point.

    Args:
        point: A 3-dimensional vector.
        rotation: The rotation in Euler angles.
        translation: The translation as a 3-dimensional vector.

    Returns:
        The transformed point as a 3-dimensional vector.
    """
    roll = rotation[0]
    pitch = rotation[1]
    yaw = rotation[2]
    rotation_matrix = matrix3_from_euler(roll, pitch, yaw)

    return rotation_matrix.dot(point) + np.array(translation)


class Hammer(Body):
    """Hammer generator.

    A hammer is defined as a two-part object composed of a handle and a head.
    """

    def __init__(self, name, obj_paths=None, color=None, random_flip=True):
        """Initialize.

        Args:
            name: The name of the body.
            obj_paths: If None, use OpenScad to gnerate objects; otherwise
                sample objects from obj_paths.
            color: The color code.
            random_flip: If true, randomly flip the parts along the three axes.
        """
        self.name = name
        self.random_flip = random_flip

        if color == 'realistic':
            handle_color = CMAP['wood']
            head_color = CMAP['metal']
        else:
            handle_color = None
            head_color = None

        with open('templates/hammer.xml', 'r') as f:
            self.template = f.read()

        if obj_paths is None:
            self.handle_generators = [
                    ScadCubeLink(
                        name='handle', color=handle_color, **HANDLE_CONFIG),
                    ScadCylinderLink(
                        name='handle', color=handle_color, **HANDLE_CONFIG),
                    ScadPolygonLink(
                        name='handle', color=handle_color, **HANDLE_CONFIG),
                    ]

            self.head_generators = [
                    ScadCubeLink(
                        name='head', color=head_color, **HEAD_CONFIG),
                    ScadCylinderLink(
                        name='head', color=head_color, **HEAD_CONFIG),
                    ScadPolygonLink(
                        name='head', color=head_color, **HEAD_CONFIG),
                    ScadBreadLink(
                        name='head', color=head_color, **HEAD_CONFIG),
                    ]
        else:
            self.handle_generators = [
                    Link(
                        name='handle',
                        color=handle_color,
                        obj_paths=obj_paths,
                        **HANDLE_CONFIG)
                    ]

            self.head_generators = [
                    Link(
                        name='head',
                        color=head_color,
                        obj_paths=obj_paths,
                        **HEAD_CONFIG)
                    ]

    def generate(self, path):
        """Generate a body.

        Args:
            path: The folder to save the URDF and OBJ files.
        """
        handle_generator = random.choice(self.handle_generators)
        head_generator = random.choice(self.head_generators)

        # Generate links.
        handle_data = handle_generator.generate(path)
        head_data = head_generator.generate(path)

        # Modify links' positions and orientations.
        rotation, transition = self.sample_head_transformation(
                handle_data, head_data)
        center = [head_data['x'], head_data['y'], head_data['z']]
        center = transform_point(center, rotation, transition)
        head_data['x'] = center[0]
        head_data['y'] = center[1]
        head_data['z'] = center[2]
        head_data['roll'] = rotation[0]
        head_data['pitch'] = rotation[1]
        head_data['yaw'] = rotation[2]

        if self.random_flip:
            if np.random.rand() >= FLIP_PROB:
                handle_data['roll'] = (handle_data['roll'] + np.pi) % (
                        2 * np.pi)

            if np.random.rand() >= FLIP_PROB:
                handle_data['pitch'] = (handle_data['pitch'] + np.pi) % (np.pi)

            if np.random.rand() >= FLIP_PROB:
                handle_data['yaw'] = (handle_data['yaw'] + np.pi) % (2 * np.pi)

            if np.random.rand() >= FLIP_PROB:
                head_data['roll'] = (head_data['roll'] + np.pi) % (2 * np.pi)

            if np.random.rand() >= FLIP_PROB:
                head_data['pitch'] = (head_data['pitch'] + np.pi) % (np.pi)

            if np.random.rand() >= FLIP_PROB:
                head_data['yaw'] = (head_data['yaw'] + np.pi) % (2 * np.pi)

        # Genearte the URDF files.
        handle_urdf = handle_generator.convert_data_to_urdf(handle_data)
        head_urdf = head_generator.convert_data_to_urdf(head_data)
        urdf = self.template.format(
                name=self.name,
                handle_link=handle_urdf,
                head_link=head_urdf,
                handle_name=handle_data['name'],
                head_name=head_data['name'],
                )

        # Write URDF to file.
        urdf_filename = os.path.join(path, '%s.urdf' % (self.name))
        with open(urdf_filename, 'w') as f:
            f.write(urdf)

    def sample_head_transformation(self, handle_data, head_data):
        """Sample the transformation for the head pose.

        Args:
            handle_data: The data dictionary of the handle.
            head_data: The data dictionary of the head.

        Returns:
            rotation: The rotation as Euler angles.
            translation: The translation vector.
        """
        # The orthogonal T-Shape hammer.
        rotation = [0.5 * np.pi, 0, 0]
        translation = [0, 0, 0.5 * handle_data['size_z']]

        return rotation, translation


class TShape(Hammer):
    """T-Sahpe generator.
    """

    def sample_head_transformation(self, handle_data, head_data):
        """Sample the transformation for the head pose.

        See the parent class.
        """
        rotation = [
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.15 * np.pi) + 0.5 * np.pi,
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.15 * np.pi),
                0,
                ]

        translation = [
                0,
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.3 * head_data['size_z']),
                0.5 * handle_data['size_z']
                ]

        return rotation, translation


class LShape(Hammer):
    """L-Sahpe generator.
    """

    def sample_head_transformation(self, handle_data, head_data):
        """Sample the transformation for the head pose.

        See the parent class.
        """
        rotation = [
                0.5 * np.pi + np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.15 * np.pi),
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.15 * np.pi),
                0,
                ]

        translation = [
                0,
                np.random.choice([-1, 1]) * np.random.uniform(
                    0.3 * head_data['size_z'], 0.5 * head_data['size_z']),
                0.5 * handle_data['size_z']
                ]

        return rotation, translation


class XShape(Hammer):
    """X-Sahpe generator.
    """

    def sample_head_transformation(self, handle_data, head_data):
        """Sample the transformation for the head pose.

        See the parent class.
        """
        rotation = [
                0.5 * np.pi + np.random.uniform(-0.15 * np.pi, 0.15 * np.pi),
                np.random.uniform(-0.15 * np.pi, 0.15 * np.pi),
                0,
                ]

        translation = [
                0,
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.3 * head_data['size_z']),
                np.random.choice([-1, 1]) * np.random.uniform(
                    0, 0.3 * handle_data['size_z'])
                ]

        return rotation, translation
