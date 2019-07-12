"""The Simulator class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.simulation import physics
from robovat.simulation.body import Body
from robovat.simulation.controllable_body import ControllableBody
from robovat.simulation.constraint import Constraint
from robovat.utils.logging import logger


class Simulator(object):
    """The Simulator class."""

    def __init__(self,
                 physics_backend='BulletPhysics',
                 time_step=1e-3,
                 gravity=[0, 0, -9.8],
                 worker_id=0,
                 max_steps=None,
                 use_visualizer=False):
        """Initialize the simulator.

        Args:
            physics_backend: Name of the physics engine backend.
            time_step: Time step of the simulation.
            gravity: The gravity as a 3-dimensional vector.
            worker_id: The id of the multi-threaded simulation.
            max_steps: Maximum number of simulation steps.
            use_visualizer: Render the simulation use the debugging visualizer
                if True.
        """
        self._gravity = gravity

        self._max_steps = max_steps
        self._num_steps = 0

        # Create the physics backend.
        physics_class = getattr(physics, physics_backend)
        self._physics = physics_class(
                time_step=time_step,
                use_visualizer=use_visualizer,
                worker_id=worker_id)

    def __del__(self):
        """Delete the simulator."""
        del self._physics

    @property
    def physics(self):
        return self._physics

    @property
    def bodies(self):
        return self._bodies

    @property
    def constraints(self):
        return self._constraints

    def reset(self):
        """Reset the simulation."""
        self.physics.reset()
        self.physics.set_gravity(self._gravity)
        self._bodies = dict()
        self._constraints = dict()
        self._num_steps = 0

    def start(self):
        """Start the simulation."""
        self.physics.start()

    def step(self):
        """Take a simulation step."""
        for body in self.bodies.values():
            body.update()

        self.physics.step()

        self._num_steps += 1

        if self._max_steps:
            if self._num_steps > self._max_steps:
                raise ValueError('Simulation has exceeded %d steps.' %
                                 (self._max_steps))

    def add_body(self,
                 filename,
                 pose=None,
                 scale=1.0,
                 is_static=False,
                 is_controllable=False,
                 name=None):
        """Add a body to the simulation.

        Args:
            filename: The path to the URDF or SDF file to be loaded.
            pose: The initial pose as an instance of Pose.
            is_static: If True, set the base of the body to be static.
            is_controllable: If True, the body can apply motor controls.
            name: Used as a reference of the body in this Simulator instance.

        Returns:
            An instance of Body.
        """
        if pose is None:
            pose = [[0, 0, 0], [0, 0, 0]]

        # Create the body.
        if is_controllable:
            body = ControllableBody(
                    simulator=self,
                    filename=filename,
                    pose=pose,
                    scale=scale,
                    is_static=is_static,
                    name=name)
        else:
            body = Body(
                    simulator=self,
                    filename=filename,
                    pose=pose,
                    scale=scale,
                    is_static=is_static,
                    name=name)

        # Add the body to the dictionary.
        self._bodies[body.name] = body

        return body

    def remove_body(self, name):
        """Remove the body.

        Args:
            body: An instance of Body.
        """
        self.physics.remove_body(self._bodies[name].uid)
        del self._bodies[name]

    def add_constraint(self,
                       parent,
                       child,
                       joint_type='fixed',
                       joint_axis=[0, 0, 0],
                       parent_frame_pose=None,
                       child_frame_pose=None,
                       name=None):
        """Add a constraint to the simulation.

        Args:
            parent: The parent entity as an instance of Entity.
            child: The child entity as an instance of Entity.
            joint_type: The type of the joint.
            joint_axis: The axis of the joint.
            parent_frame_pose: The pose of the joint in the parent frame.
            child_frame_pose: The pose of the joint in the child frame.

        Returns:
            An instance of Constraint.
        """
        # Create the constraint.
        constraint = Constraint(
                 parent,
                 child,
                 joint_type,
                 joint_axis,
                 parent_frame_pose,
                 child_frame_pose,
                 name)

        # Add the constraint to the dictionary.
        self._constraints[constraint.name] = constraint

        return constraint

    def receive_robot_commands(self, robot_command):
        """Receive a robot command.

        Args:
            robot_command: An instance of RobotCommand.
        """
        body = self._bodies[robot_command.component]
        command_method = getattr(body, robot_command.command_type)
        command_method(**robot_command.arguments)

    def check_contact(self, entity_a, entity_b=None):
        """Check if the loaded object is stable.

        Args:
            entity_a: The first entity.
            entity_b: The second entity, None for any entities.

        Returns:
            True if they have contacts, False otherwise.
        """
        def _check_contact(entity_a, entity_b=None):
            a_uid = entity_a.uid
            if entity_b is None:
                b_uid = None
            else:
                b_uid = entity_b.uid

            contact_points = self._physics.get_contact_points(
                    a_uid, b_uid)
            has_contact = len(contact_points) > 0

            return has_contact

        if not isinstance(entity_a, (list, tuple)):
            entities_a = [entity_a]
        else:
            entities_a = entity_a

        if not isinstance(entity_b, (list, tuple)):
            entities_b = [entity_b]
        else:
            entities_b = entity_b

        has_contact = False

        for a in entities_a:
            for b in entities_b:
                if _check_contact(a, b):
                    has_contact = True

        return has_contact

    def check_stable(self,
                     body,
                     linear_velocity_threshold,
                     angular_velocity_threshold):
        """Check if the loaded object is stable.

        Args:
            body: An instance of body.

        Returns:
            is_stable: True if the linear velocity and the angular velocity are
            almost zero; False otherwise.
        """
        linear_velocity = np.linalg.norm(body.linear_velocity)
        angular_velocity = np.linalg.norm(body.angular_velocity)

        if linear_velocity_threshold is None:
            has_linear_velocity = False
        else:
            has_linear_velocity = (
                    linear_velocity >= linear_velocity_threshold)

        if angular_velocity_threshold is None:
            has_angular_velocity = False
        else:
            has_angular_velocity = (
                    angular_velocity >= angular_velocity_threshold)

        is_stable = (not has_linear_velocity) and (not has_angular_velocity)

        return is_stable

    def wait_until_stable(self,
                          body,
                          linear_velocity_threshold=0.005,
                          angular_velocity_threshold=0.005,
                          check_after_steps=100,
                          min_stable_steps=100,
                          max_steps=2000):
        """Wait until the objects are stable."""
        logger.debug('Waiting for objects to be stable...')

        if isinstance(body, (list, tuple)):
            body_list = body
        else:
            body_list = [body]

        num_steps = 0
        num_stable_steps = 0

        while(1):
            self.step()
            num_steps += 1

            if num_steps < check_after_steps:
                continue

            # Check if all bodies are stable.
            all_stable = True
            for b in body_list:
                is_stable = self.check_stable(
                        b,
                        linear_velocity_threshold,
                        angular_velocity_threshold)

                if not is_stable:
                    all_stable = False
                    break
        
            if all_stable:
                num_stable_steps += 1

            if (num_stable_steps >= min_stable_steps or num_steps >= max_steps):
                break
