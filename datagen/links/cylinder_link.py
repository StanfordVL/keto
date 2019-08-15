"""Link generator.
"""
import numpy as np

from links.link import LinkGenerator


class CylinderLinkGenerator(LinkGenerator):

    def __init__(self,
                 name,
                 mass_range,
                 lateral_friction_range,
                 spinning_friction_range,
                 inertia_friction_range,
                 scale_range,
                 ):
        """Initialize.
        """
        with open('templates/cylinder_link.xml', 'r') as f:
            self.template = f.read()

        self.name = name
        self.mass_range = mass_range
        self.lateral_friction_range = lateral_friction_range
        self.spinning_friction_range = spinning_friction_range
        self.inertia_friction_range = inertia_friction_range
        self.scale_range = scale_range

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
        data['scale_x'] = np.random.uniform(*self.scale_range[0])
        data['scale_y'] = np.random.uniform(*self.scale_range[1])
        data['scale_z'] = np.random.uniform(*self.scale_range[2])

        return data
