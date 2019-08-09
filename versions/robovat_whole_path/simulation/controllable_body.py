"""The body class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.simulation.body import Body
from robovat.utils.logging import logger


# These default constants are set for the Sawyer robot.
TIMEOUT = 15.0  # TODO(kuanfang): Change it back to 15?
JOINT_POSITION_THRESHOLD = 0.008726640
JOINT_VELOCITY_THRESHOLD = 0.05
POSITION_GAIN = 0.05
VELOCITY_GAIN = 1.0
JOINT_DAMPING = 0.7
STEPS_TO_COMPUTE_IK = 10


class ControllableBody(Body):
    """Body."""

    def __init__(self,
                 simulator,
                 filename,
                 pose,
                 scale=1.0,
                 is_static=False,
                 name=None):
        """Initialize."""
        Body.__init__(self,
                      simulator=simulator,
                      filename=filename,
                      pose=pose,
                      scale=scale,
                      is_static=is_static,
                      name=name)

        self.reset_targets()
        self._max_reaction_forces = [None] * len(self.joints)

    def set_max_reaction_force(self, joint_ind, force):
        """Set the maximum reaction force for a joint.
        """
        self._max_reaction_forces[joint_ind] = force
        self.joints[joint_ind].enable_sensor()

    def reset_targets(self):
        """Reset the control variables.
        """
        self._max_joint_velocities = [
                joint.max_velocity for joint in self.joints
                ]
        self._neutral_joint_positions = None

        self._target_joint_positions = [None] * len(self.joints)
        self._joint_start_time = [None] * len(self.joints)
        self._joint_stop_time = [None] * len(self.joints)
        self._joint_position_threshold = JOINT_POSITION_THRESHOLD
        self._joint_velocity_threshold = JOINT_VELOCITY_THRESHOLD

        self._target_link = None
        self._target_link_pose = None
        self._target_link_poses = []
        self._ik_start_time = None
        self._ik_stop_time = None
        self._ik_num_steps = 0

        self.position_gain = POSITION_GAIN
        self.velocity_gain = VELOCITY_GAIN

    def is_ready(self, joint_inds=None):
        """Check if the body is ready.

        Args:
            joint_inds: The joints to be checked.

        Returns:
            True if all control commands are done, False otherwise.
        """
        if joint_inds is None:
            if self._target_link is not None:
                return False
            elif any(self._target_joint_positions):
                return False
            else:
                return True
        else:
            is_ready = True

            for joint_ind in joint_inds:
                if self._target_link is not None:
                    if joint_ind < self._target_link.index:
                        is_ready = False

                if self._target_joint_positions[joint_ind] is not None:
                    is_ready = False

            return is_ready

    def set_target_joint_positions(self,
                                   joint_positions,
                                   timeout=TIMEOUT,
                                   threshold=JOINT_POSITION_THRESHOLD):
        """Set target joint positions for position control.

        Args:
            joint_positions: The joint positions for each specified joint.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        self.reset_targets()
        start_time = self.physics.time()
        stop_time = start_time + timeout

        if isinstance(joint_positions, (list, tuple)):
            for joint_ind, joint_position in enumerate(joint_positions):
                if joint_position is not None:
                    self._target_joint_positions[joint_ind] = joint_position
                    self._joint_start_time[joint_ind] = start_time
                    self._joint_stop_time[joint_ind] = stop_time
        elif isinstance(joint_positions, dict):
            for key, joint_position in joint_positions.items():
                joint = self.get_joint_by_name(key)
                joint_ind = joint.index
                self._target_joint_positions[joint_ind] = joint_position
                self._joint_start_time[joint_ind] = start_time
                self._joint_stop_time[joint_ind] = stop_time
        else:
            raise ValueError

        self._joint_position_threshold = threshold

    def set_target_link_pose(self,
                             link_ind,
                             link_pose,
                             timeout=TIMEOUT,
                             threshold=JOINT_POSITION_THRESHOLD):
        """Set the target pose of the link for position control.

        Args:
            link_ind: The index of the end effector link.
            link_pose: The pose of the end effector link.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        self.reset_targets()
        self._ik_start_time = self.physics.time()
        self._ik_stop_time = self._ik_start_time + timeout
        self._target_link = self.links[link_ind]
        self._target_link_pose = link_pose
        self._target_link_poses = []
        self._joint_position_threshold = threshold
        self._ik_num_steps = 0

    def set_target_link_poses(self,
                              link_ind,
                              link_poses,
                              timeout=TIMEOUT,
                              threshold=JOINT_POSITION_THRESHOLD):
        """Set the target poses of the link for position control.

        The list of poses should be reached sequentially.

        Args:
            link_ind: The index of the end effector link.
            link_pose: The pose of the end effector link.
            timeout: Seconds to wait for move to finish.
            threshold: Joint position threshold in radians across each joint
                when move is considered successful.
        """
        assert isinstance(link_poses, list)
        self.reset_targets()

        self._ik_start_time = self.physics.time()
        self._ik_stop_time = self._ik_start_time + timeout
        self._target_link = self.links[link_ind]
        self._target_link_pose = None
        self._target_link_poses = link_poses
        self._joint_position_threshold = threshold
        self._ik_num_steps = 0

    def set_neutral_joint_positions(self, joint_positions):
        """Set the neutral joint positions.

        This is used for computing the IK solution.

        Args:
            joint_positions: A list of joint positions.
        """
        self._neutral_joint_positions = joint_positions

    def set_max_joint_velocities(self, joint_velocities):
        """Set the maximal joint velocities for position control.

        Args:
            joint_velocities: The maximal joint velocities.
        """
        # TODO(kuanfang): This functionality is not supported by pybulelt yet.
        if isinstance(joint_velocities, (list, tuple)):
            for joint_ind, joint_velocity in enumerate(joint_velocities):
                if joint_velocity is not None:
                    self._max_joint_velocities[joint_ind] = joint_velocity
        elif isinstance(joint_velocities, dict):
            for key, joint_velocity in joint_velocities.items():
                joint = self.get_joint_by_name(key)
                joint_ind = joint.index
                self._max_joint_velocities[joint_ind] = joint_velocity
        else:
            raise ValueError

    def update(self):
        """Update control and disturbances."""
        # Call the update function of the super class.
        super(ControllableBody, self).update()

        if self._target_link is not None:
            is_ik_done = self._update_ik()

            if is_ik_done:
                self._ik_num_steps = 0
            else:
                self._ik_num_steps += 1

        self._update_position_control()

    def _update_position_control(self):
        """Update the position control."""
        # Return if no target joint position is set.
        if not any(self._target_joint_positions):
            return

        # Set position control arguments.
        target_joint_inds = []
        target_joint_positions = []
        target_joint_velocities = []
        max_joint_velocities = []

        for joint_ind, joint in enumerate(self.joints):
            target_joint_position = self._target_joint_positions[joint_ind]
            if target_joint_position is not None:
                target_joint_inds.append(joint_ind)
                target_joint_positions.append(target_joint_position)
                target_joint_velocities.append(0)
                max_joint_velocity = self._max_joint_velocities[joint_ind]
                max_joint_velocities.append(max_joint_velocity)

        assert len(target_joint_inds) == len(target_joint_positions)
        assert len(target_joint_inds) == len(target_joint_velocities)

        # Run the motor control.
        for i, joint_ind in enumerate(target_joint_inds):
            joint = self.joints[joint_ind]
            self.physics.position_control(
                    joint.uid,
                    target_position=target_joint_positions[i],
                    target_velocity=target_joint_velocities[i],
                    max_velocity=max_joint_velocities[i],
                    position_gain=self.position_gain,
                    velocity_gain=self.velocity_gain)

        # TODO(debug): Debug position control.
        # for i, joint_ind in enumerate(target_joint_inds):
        #     joint = self.joints[joint_ind]
        #     joint.position = target_joint_positions[i]

        # Check if all joints reach the target position.
        is_reached = self.check_reached(target_joint_inds,
                                        target_joint_positions,
                                        target_joint_velocities,
                                        self._joint_position_threshold,
                                        self._joint_velocity_threshold)

        # Reset the target, if the joint is timeout or all targets are reached.
        current_time = self.physics.time()
        for joint_ind in target_joint_inds:
            is_timeout = self.check_joint_timeout(joint_ind, current_time)
            is_safe = self.check_joint_safe(joint_ind)

            # Reset.
            if is_reached or is_timeout or (not is_safe):
                self._target_joint_positions[joint_ind] = None
                self._joint_start_time[joint_ind] = None
                self._joint_stop_time[joint_ind] = None

    def _update_ik(self):
        """Update the inverse kinematics results.

        Returns:
            done: If the IK is done.
        """
        ik_link_ind = self._target_link.index
        joint_inds = range(ik_link_ind)

        any_joint_reached = any(
                [
                    self._target_joint_positions[joint_ind] is None
                    for joint_ind in joint_inds
                ]
                )

        if not (self._ik_num_steps % STEPS_TO_COMPUTE_IK == 0
                or any_joint_reached):
            return False

        # If the current target link pose has been reached, aim for the next.
        if self._target_link_pose is None:
            if len(self._target_link_poses) == 0:
                raise ValueError(
                        'When self._target_link_poses is empty,'
                        'self._target_link should have been set to None')
            else:
                self._target_link_pose = self._target_link_poses[0]
                self._target_link_poses = self._target_link_poses[1:]

                for joint_ind in joint_inds:
                    self._target_joint_positions[joint_ind] = None

        # If use Bullet IK (iterative), check if the IK result converges to the
        # current joint positions; if use other IK solvers, check if the
        # previously computed IK results has been reached.
        joint_positions = self.physics.compute_inverse_kinematics(
                self._target_link.uid,
                self._target_link_pose,
                # upper_limits=self.joint_upper_limits,
                # lower_limits=self.joint_lower_limits,
                # ranges=self.joint_ranges,
                # damping=[JOINT_DAMPING] * len(self.joints),
                neutral_positions=self._neutral_joint_positions)

        # If there is no more target lin poses waiting in the queue, the target
        # velocities are zeros; otherwise velocities do not matter.
        if len(self._target_link_poses) == 0:
            joint_velocities = [0] * len(joint_inds)
            self.position_gain = POSITION_GAIN
            self.velocity_gain = VELOCITY_GAIN
        else:
            self.position_gain = 1.0
            self.velocity_gain = 0.0
            joint_velocities = [None] * len(joint_inds)

        # Check the stop conditions.
        is_converged = self.check_reached(joint_inds,
                                          joint_positions,
                                          joint_velocities,
                                          self._joint_position_threshold,
                                          self._joint_velocity_threshold)
        is_timeout = self.physics.time() >= self._ik_stop_time

        # Warnings.
        if is_timeout:
            logger.warning('Time out for the IK control of link %s.'
                           % self._target_link.name)

        if is_converged or is_timeout:
            # Clear the IK target, if it is done.
            self._target_link_pose = None
            if len(self._target_link_poses) == 0:
                self._target_link = None
                self._ik_start_time = None
                self._ik_stop_time = None
            return True
        else:
            # Set the position control.
            for joint_ind, joint_position in zip(joint_inds, joint_positions):
                self._target_joint_positions[joint_ind] = joint_position
                self._joint_start_time[joint_ind] = self._ik_start_time
                self._joint_stop_time[joint_ind] = self._ik_stop_time
            return False

    def check_reached(self, joint_inds, joint_positions, joint_velocities,
                      joint_position_threshold, joint_velocity_threshold):
        """Check if the specified joint positions are reached.

        Args:
            joint_inds: List of joint indices.
            joint_positions: List of target joint positions.
            joint_positions: List of target joint velocities.

        Returns:
            True is all specified joint positions are reached and corresponding
                joint velocities are close to zero, False otherwise.
        """
        for joint_ind, target_position, target_velocity in zip(
                joint_inds, joint_positions, joint_velocities):
            current_position = self.joints[joint_ind].position
            delta_position = target_position - current_position
            position_reached = (
                    abs(delta_position) < joint_position_threshold)

            if target_velocity is None:
                velocity_reached = True
            else:
                current_velocity = self.joint_velocities[joint_ind]
                delta_velocity = target_velocity - current_velocity
                velocity_reached = (
                        abs(delta_velocity) < joint_velocity_threshold)

            if not (position_reached and velocity_reached):
                return False

        return True

    def check_joint_timeout(self, joint_ind, current_time=None):
        """Check if the joint is timeout.

        Args:
            joint_ind: The joint index.
            current_time: The current simulation time.

        Returns:
            True for timeout, False otherwise.
        """
        if current_time is None:
            current_time = self.physics.time()

        is_timeout = current_time >= self._joint_stop_time[joint_ind]

        if is_timeout:
            target_position = self._target_joint_positions[joint_ind]
            current_position = self.joints[joint_ind].position
            delta_position = target_position - current_position
            current_velocity = self.joint_velocities[joint_ind]
            logger.warning(
                'Time out (%.2f) for the position control of joint %s,'
                'with delta_position = %.3f, velocity = %.3f.'
                % ((self._joint_stop_time[joint_ind] -
                    self._joint_start_time[joint_ind]),
                   self.joints[joint_ind].name,
                   delta_position,
                   current_velocity))

        return is_timeout

    def check_joint_safe(self, joint_ind):
        """Check if the joint is safe.

        Args:
            joint_ind: The joint index.

        Returns:
            True for safe, False otherwise.
        """
        max_force = self._max_reaction_forces[joint_ind]

        if max_force is None:
            return True
        else:
            joint = self.joints[joint_ind]
            reaction_force = joint.reaction_force
            reaction_force_norm = np.linalg.norm(reaction_force[:3])
            is_safe = reaction_force_norm < max_force

            if not is_safe:
                logger.warning('Joint %s has a reaction force %s, '
                               'which is larger than the threshold %.2f.'
                               % (joint_ind, reaction_force_norm, max_force))

            return is_safe
