"""Link.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.simulation.entity import Entity


class Link(Entity):
    """Link of the body."""

    def __init__(self, body, link_ind):
        """Initialize.

        Args:
            body: The body of the joint.
            link_ind: The link index.
        """
        uid = (body.uid, link_ind)
        name = body.physics.get_link_name(uid)

        if isinstance(name, bytes):
            name = name.decode('utf-8')

        Entity.__init__(self, simulator=body.simulator, name=name)
        self._uid = uid
        self._parent = body
        self._index = link_ind

    def set_dynamics(self,
                     mass=None,
                     lateral_friction=None,
                     rolling_friction=None,
                     spinning_friction=None,
                     ):
        """Set dynmamics.
        """
        return self.physics.set_link_dynamics(
                self.uid,
                mass=mass,
                lateral_friction=lateral_friction,
                rolling_friction=rolling_friction,
                spinning_friction=rolling_friction,
                )

    @property
    def parent(self):
        return self._parent

    @property
    def index(self):
        return self._index

    @property
    def pose(self):
        return self.physics.get_link_pose(self.uid)

    @property
    def center_of_mass(self):
        return self.physics.get_link_center_of_mass(self.uid)

    @property
    def mass(self):
        if self._mass is None:
            self._mass = self.physics.get_link_mass(self.uid)
        return self._mass

    @property
    def dynamics(self):
        return self.physics.get_link_dynamics(self.uid)
