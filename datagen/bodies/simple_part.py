"""Hammer generator.
"""

import os
import random

import numpy as np

from bodies.body import Body
from links.link import Link
from utils.transformations import matrix3_from_euler


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


class SimplePart(Body):
    """Hammer generator.

    A hammer is defined as a two-part object composed of a handle and a head.
    """

    def __init__(self, name, config, color=None):

        self.name = name
        self.config = config

        with open('templates/simple_part.xml', 'r') as f:
            self.template = f.read()

        part_config = config['part']['config']
        self.part_generators = [
            Link(
                 name='part',
                 color=CMAP[color],
                 obj_paths=[config['part']['obj_path']],
                 **part_config)
                ]

    def generate(self, path):
        """Generate a body.

        Args:
            path: The folder to save the URDF and OBJ files.
        """
        part_generator = random.choice(self.part_generators)

        # Generate links.
        part_data = part_generator.generate(path)

        # Modify links' positions and orientations.
        rotation = self.config['part']['rotation']
        transition = self.config['part']['translation']

        center = [part_data['x'], part_data['y'], part_data['z']]
        center = transform_point(center, rotation, transition)

        part_data['x'] = center[0]
        part_data['y'] = center[1]
        part_data['z'] = center[2]
        part_data['roll'] = rotation[0]
        part_data['pitch'] = rotation[1]
        part_data['yaw'] = rotation[2]

        # Genearte the URDF files.
        part_urdf = part_generator.convert_data_to_urdf(part_data)
        urdf = self.template.format(
                name=self.name,
                part_link=part_urdf,
                part_name=part_data['name'],
                )

        # Write URDF to file.
        urdf_filename = os.path.join(path, '%s.urdf' % (self.name))
        with open(urdf_filename, 'w') as f:
            f.write(urdf)
