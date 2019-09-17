"""Hammer generator.
"""

import os
import random

import numpy as np

from bodies.body import Body
from links.link import Link
from utils.transformations import matrix3_from_euler


FLIP_PROB = 0.5

LATERAL_FRICTION_RANGE = [0.5, 1.0]
SPINNING_FRICTION_RANGE = [0.5, 1.0]
INERTIA_FRICTION_RANGE = [0.5, 1.0]

MAIN_CONFIG = {
        'mass_range': [0.2, 0.4],
        'size_range': [[0.02, 0.04], [0.02, 0.04], [0.20, 0.30]],
        'lateral_friction_range': LATERAL_FRICTION_RANGE,
        'spinning_friction_range': SPINNING_FRICTION_RANGE,
        'inertia_friction_range': INERTIA_FRICTION_RANGE,
        }

PART_CONFIG = {
        'mass_range': [0.04, 0.08],
        'size_range': [[0.02, 0.04], [0.02, 0.04], [0.00, 0.20]],
        'lateral_friction_range': LATERAL_FRICTION_RANGE,
        'spinning_friction_range': SPINNING_FRICTION_RANGE,
        'inertia_friction_range': INERTIA_FRICTION_RANGE,
        }

CMAP = {
        'wood': [0.494, 0.278, 0.164],
        'metal': [0.654, 0.698, 0.761],
        'grey': [0.8, 0.8, 0.8],
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


class General(Body):
    """General object generator.

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
            main_color = CMAP['grey']
            part_color = CMAP['grey']
        else:
            main_color = None
            part_color = None

        with open('templates/general.xml', 'r') as f:
            self.template = f.read()

        self.main_generators = [
                    Link(
                        name='main',
                        color=main_color,
                        obj_paths=obj_paths,
                        **MAIN_CONFIG)
                    ]

        self.part_a_generators = [
                    Link(
                        name='part_a',
                        color=part_color,
                        obj_paths=obj_paths,
                        **PART_CONFIG)
                    ]

        self.part_b_generators = [
                    Link(
                        name='part_b',
                        color=part_color,
                        obj_paths=obj_paths,
                        **PART_CONFIG)
                    ]

    def generate(self, path):
        """Generate a body.

        Args:
            path: The folder to save the URDF and OBJ files.
        """
        main_generator = random.choice(self.main_generators)
        part_a_generator = random.choice(self.part_a_generators)
        part_b_generator = random.choice(self.part_b_generators)

        # Generate links.
        main_data = main_generator.generate(path)
        part_a_data = part_a_generator.generate(path)
        part_b_data = part_b_generator.generate(path)

        # Modify links' positions and orientations.
        rotations, transitions = self.sample_transformation(
                main_data, part_a_data, part_b_data)

        center_a = [part_a_data['x'], part_a_data['y'], part_a_data['z']]
        center_a = transform_point(center_a, rotations[0], transitions[0])

        part_a_data['x'] = center_a[0]
        part_a_data['y'] = center_a[1]
        part_a_data['z'] = center_a[2]

        part_a_data['roll'] = rotations[0][0]
        part_a_data['pitch'] = rotations[0][1]
        part_a_data['yaw'] = rotations[0][2]

        center_b = [part_b_data['x'], part_b_data['y'], part_b_data['z']]
        center_b = transform_point(center_b, rotations[1], transitions[1])

        part_b_data['x'] = center_b[0]
        part_b_data['y'] = center_b[1]
        part_b_data['z'] = center_b[2]

        part_b_data['roll'] = rotations[1][0]
        part_b_data['pitch'] = rotations[1][1]
        part_b_data['yaw'] = rotations[1][2]

        # Genearte the URDF files.
        main_urdf = main_generator.convert_data_to_urdf(main_data)
        part_a_urdf = part_a_generator.convert_data_to_urdf(part_a_data)
        part_b_urdf = part_b_generator.convert_data_to_urdf(part_b_data)

        urdf = self.template.format(
                name=self.name,
                main_link=main_urdf,
                part_a_link=part_a_urdf,
                part_b_link=part_b_urdf,
                main_name=main_data['name'],
                part_a_name=part_a_data['name'],
                part_b_name=part_b_data['name'],
                )

        # Write URDF to file.
        urdf_filename = os.path.join(path, '%s.urdf' % (self.name))
        with open(urdf_filename, 'w') as f:
            f.write(urdf)

    def sample_transformation(self, 
                              main_data, 
                              part_a_data,
                              part_b_data):
        """Sample the transformation for the head pose.

        Args:
            handle_data: The data dictionary of the handle.
            head_data: The data dictionary of the head.

        Returns:
            rotation: The rotation as Euler angles.
            translation: The translation vector.
        """
        # The orthogonal T-Shape hammer.
        rotations = np.random.uniform(
                low=-np.pi, high=np.pi, size=[2, 3]
                ) * np.array([0.4, 0.2, 0.0]) + np.array([np.pi/2, 0, 0])

        mean_size_a = (part_a_data['size_x'] + part_a_data['size_y'] +
                       part_a_data['size_z']) / 3.0
        mean_size_b = (part_b_data['size_x'] + part_b_data['size_y'] +
                       part_b_data['size_z']) / 3.0

        trans_low = [[-mean_size_a * 0.3, 
                      -mean_size_a * 0.3, 
                      -main_data['size_z'] * 0.5],
                     [-mean_size_b * 0.3, 
                      -mean_size_b * 0.3, 
                      -main_data['size_z'] * 0.5]]
        trans_high = [[mean_size_a * 0.3, 
                       mean_size_a * 0.3, 
                       main_data['size_z'] * 0.5],
                      [mean_size_b * 0.3, 
                       mean_size_b * 0.3, 
                       main_data['size_z'] * 0.5]]

        translations = np.random.uniform(
                low=trans_low, high=trans_high) * np.array([0, 1, 1])
        return rotations, translations
