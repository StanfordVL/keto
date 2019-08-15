"""Link generator.
"""
import abc
import os

import numpy as np


class Link(object):
    """Link generator.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 name,
                 size_range,
                 mass_range,
                 lateral_friction_range,
                 spinning_friction_range,
                 inertia_friction_range,
                 color,
                 obj_paths,
                 ):
        """Initialize.

        Args:
            name: Name of the link.
            size_range: The range of the shape size as a numpy array of [3, 2].
            mass_range: The range of the mass of the link.
            lateral_friction_range: The range of the lateral friction.
            spinning_friction_range: The range of the spinning friction.
            inertia_friction_range: The range of the inertia friction.
            color: The color code or the (r, g, b) values.
            obj_paths: Paths from which the objects are sampled.
        """
        with open('templates/link.xml', 'r') as f:
            self.template = f.read()

        self.name = name
        self.size_range = size_range
        self.mass_range = mass_range
        self.lateral_friction_range = lateral_friction_range
        self.spinning_friction_range = spinning_friction_range
        self.inertia_friction_range = inertia_friction_range
        self.color = color
        self.obj_paths = obj_paths

    def generate(self, path=None):
        """Generate a link.

        The center of mass of each mesh should be aligned with the origin.

        Args:
            path: The folder to save the URDF and OBJ files.

        Returns:
            data: Dictionary of the link attributes.
        """
        data = dict()

        data['name'] = self.name

        # Set contact.
        data['mass'] = np.random.uniform(*self.mass_range)

        # Set inertial.
        data['lateral_friction'] = np.random.uniform(
                *self.lateral_friction_range)
        data['spinning_friction'] = np.random.uniform(
                *self.spinning_friction_range)
        data['inertia_scaling'] = np.random.uniform(
                *self.inertia_friction_range)

        # Set mesh.
        data['x'] = 0
        data['y'] = 0
        data['z'] = 0
        data['roll'] = 0
        data['pitch'] = 0
        data['yaw'] = 0
        data['size_x'] = np.random.uniform(*self.size_range[0])
        data['size_y'] = np.random.uniform(*self.size_range[1])
        data['size_z'] = np.random.uniform(*self.size_range[2])
        data['scale_x'] = data['size_x']
        data['scale_y'] = data['size_y']
        data['scale_z'] = data['size_z']

        # Set color.
        color = self.sample_color()
        data['color_r'] = color[0]
        data['color_g'] = color[1]
        data['color_b'] = color[2]

        # Choose and copy mesh.
        obj_file = np.random.choice(self.obj_paths)
        data['filename'] = os.path.join(path, '%s.obj' % (self.name))
        command = 'cp -r {:s} {:s}'.format(obj_file, data['filename'])
        os.system(command)

        return data

    def sample_color(self):
        """Sample the color.

        Returns:
            Color as values of (r, g, b).
        """
        if self.color is None:
            color = np.random.rand(3)
        else:
            color = self.color

        return color

    def convert_data_to_urdf(self, data):
        """Get the urdf text of the link.

        Args:
            data: Dictionary of the link attributes.

        Returns:
            The text of the link data in the URDF format.
        """
        return self.template.format(**data)
