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


class SimpleHammer(Body):

    def __init__(self, name, config, color=None):

        self.name = name
        self.config = config

        with open('templates/hammer.xml', 'r') as f:
            self.template = f.read()

        handle_config = config['handle']['config']
        self.handle_generators = [
            Link(
                 name='handle',
                 color=CMAP['metal'],
                 obj_paths=[config['handle']['obj_path']],
                 **handle_config)
                ]

        head_config = config['head']['config']
        self.head_generators = [
            Link(
                 name='head',
                 color=CMAP['metal'],
                 obj_paths=[config['head']['obj_path']],
                 **head_config)
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
        rotation_handle = self.config['handle']['rotation']
        transition_handle = self.config['handle']['translation']

        rotation_head = self.config['head']['rotation']
        transition_head = self.config['head']['translation']

        center_handle = [handle_data['x'], handle_data['y'], handle_data['z']]
        center_handle = transform_point(
                center_handle, rotation_handle, transition_handle)

        handle_data['x'] = center_handle[0]
        handle_data['y'] = center_handle[1]
        handle_data['z'] = center_handle[2]
        handle_data['roll'] = rotation_handle[0]
        handle_data['pitch'] = rotation_handle[1]
        handle_data['yaw'] = rotation_handle[2]

        center_head = [head_data['x'], head_data['y'], head_data['z']]
        center_head = transform_point(
                center_head, rotation_head, transition_head)

        head_data['x'] = center_head[0]
        head_data['y'] = center_head[1]
        head_data['z'] = center_head[2]
        head_data['roll'] = rotation_head[0]
        head_data['pitch'] = rotation_head[1]
        head_data['yaw'] = rotation_head[2]

        # Genearte the URDF files.
        handle_urdf = handle_generator.convert_data_to_urdf(handle_data)
        head_urdf = head_generator.convert_data_to_urdf(head_data)

        urdf = self.template.format(
                name=self.name,
                handle_link=handle_urdf,
                handle_name=handle_data['name'],
                head_link=head_urdf,
                head_name=head_data['name'],
                )

        # Write URDF to file.
        urdf_filename = os.path.join(path, '%s.urdf' % (self.name))
        with open(urdf_filename, 'w') as f:
            f.write(urdf)
