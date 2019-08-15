"""Link generator through OpenScad.
"""
import abc
import os
from sys import platform

import numpy as np

from links.link import Link


ROUND_PROB = 0.7


class ScadLink(Link):
    """Generate link with OpenScad.

    http://www.openscad.org/cheatsheet/index.html
    """

    def __init__(self,
                 name,
                 size_range,
                 mass_range,
                 lateral_friction_range,
                 spinning_friction_range,
                 inertia_friction_range,
                 color,
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
        data['scale_x'] = 1
        data['scale_y'] = 1
        data['scale_z'] = 1

        # Set color.
        color = self.sample_color()
        data['color_r'] = color[0]
        data['color_g'] = color[1]
        data['color_b'] = color[2]

        # Generate mesh use OpenScad.
        data['filename'] = self.run_openscad(path, data)

        return data

    def run_openscad(self, path, data):
        """Run OpenScad command.

        Args:
            path: The folder to save the URDF and OBJ files.
            data: The data dictionary.

        Returns:
            obj_filename: The filename of the OBJ file.
        """
        # Set filenames.
        scad_filename = os.path.join(path, '%s.scad' % (self.name))
        stl_filename = os.path.join(path, '%s.stl' % (self.name))
        obj_filename = os.path.join(path, '%s.obj' % (self.name))
        output_filename = os.path.join(path, '%s' % (self.name))

        # Run OpenScad.
        scad = self.generate_scad(data)

        with open(scad_filename, 'w') as f:
            f.write(scad)

        command = 'openscad -o {:s} {:s}'.format(stl_filename, scad_filename)
        os.system(command)

        # Convert the generated STL file to OBJ file.
        if platform == 'linux' or platform == 'linux2':
            meshconv_bin = './bin/meshconv_linux'
        elif platform == 'darwin':
            meshconv_bin = './bin/meshconv_osx'
        elif platform == 'win32':
            meshconv_bin = './bin/meshconv.exe'
        else:
            raise ValueError

        command = '{:s} -c obj -tri -o {:s} {:s}'.format(
                meshconv_bin, output_filename, stl_filename)
        os.system(command)

        # Return.
        return obj_filename

    @abc.abstractmethod
    def generate_scad(self, data):
        """Randomly generate the OpenScad description data.

        Args:
            data: The data dictionary.

        Returns:
            scad: The description data.
        """
        raise NotImplementedError


class ScadCubeLink(ScadLink):
    """Generate cuboids with OpenScad.

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cube
    """

    def generate_scad(self, data):
        """Randomly generate the OpenScad description data.

        Args:
            data: The data dictionary.

        Returns:
            scad: The description data.
        """
        scad = 'cube([{x:f}, {y:f}, {z:f}], center=true);'.format(
                x=data['size_x'], y=data['size_y'], z=data['size_z'])

        return scad


class ScadCylinderLink(ScadLink):
    """Generate cylinder with OpenScad.

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder
    """

    def generate_scad(self, data):
        """Randomly generate the OpenScad description data.

        Args:
            data: The data dictionary.

        Returns:
            scad: The description data.
        """
        fn = np.random.randint(20, 30)

        scad = ('cylinder(h={h:f}, r1={r1:f}, r2={r2:f}, $fn={fn:d}, '
                'center=true);').format(h=data['size_z'],
                                        r1=0.5 * data['size_x'],
                                        r2=0.5 * data['size_y'],
                                        fn=fn)

        if np.random.rand() < ROUND_PROB:
            scad1 = 'sphere(r={r:f}, $fn={fn:d});'.format(
                    r=0.5 * data['size_x'], fn=fn)
            scad1 = 'translate([0, 0, -%f]) %s ' % (
                    0.5 * data['size_z'], scad1)
            scad = scad + ' ' + scad1

        if np.random.rand() < ROUND_PROB:
            scad2 = 'sphere(r={r:f}, $fn={fn:d});'.format(
                    r=0.5 * data['size_y'], fn=fn)
            scad2 = 'translate([0, 0, %f]) %s ' % (
                    0.5 * data['size_z'], scad2)
            scad = scad + ' ' + scad2

        return scad


class ScadPolygonLink(ScadLink):
    """Generate polygon with OpenScad.

    Polygon is a cylinder with very small number of fragments.

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder
    """

    def generate_scad(self, data):
        """Randomly generate the OpenScad description data.

        Args:
            data: The data dictionary.

        Returns:
            scad: The description data.
        """
        fn = np.random.randint(6, 20)

        scad = ('cylinder(h={h:f}, r1={r1:f}, r2={r2:f}, $fn={fn:d}, '
                'center=true);').format(h=data['size_z'],
                                        r1=0.5 * data['size_x'],
                                        r2=0.5 * data['size_y'],
                                        fn=fn)

        if np.random.rand() < ROUND_PROB:
            scad1 = 'sphere(r={r:f}, $fn={fn:d});'.format(
                    r=0.5 * data['size_x'], fn=fn)
            scad1 = 'translate([0, 0, -%f]) %s ' % (
                    0.5 * data['size_z'], scad1)
            scad = scad + ' ' + scad1

        if np.random.rand() < ROUND_PROB:
            scad2 = 'sphere(r={r:f}, $fn={fn:d});'.format(
                    r=0.5 * data['size_y'], fn=fn)
            scad2 = 'translate([0, 0, %f]) %s ' % (
                    0.5 * data['size_z'], scad2)
            scad = scad + ' ' + scad2

        return scad


class ScadBreadLink(ScadLink):
    """Generate polygon with OpenScad.

    Polygon is a cylinder with very small number of fragments.

    https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Primitive_Solids#cylinder
    """

    def generate_scad(self, data):
        """Randomly generate the OpenScad description data.

        Args:
            data: The data dictionary.

        Returns:
            scad: The description data.
        """
        fn = np.random.randint(6, 40)

        scad1 = ('cylinder(h={h:f}, r={r:f}, $fn={fn:d}, '
                 'center=true);').format(h=data['size_z'],
                                         r=0.5 * data['size_y'],
                                         fn=fn)

        scad2 = 'cube([{x:f}, {y:f}, {z:f}], center=true);'.format(
                x=data['size_x'], y=data['size_y'], z=data['size_z'])
        scad2 = 'translate([0, %f, 0]) %s ' % (0.5 * data['size_y'], scad2)

        scad = scad1 + ' ' + scad2

        return scad
