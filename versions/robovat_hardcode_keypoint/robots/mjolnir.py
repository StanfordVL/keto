"""The class of the Sawyer robot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from robovat.math.pose import Pose
from robovat.robots import robot
from robovat.robots.robot_command import RobotCommand


class Mjolnir(robot.Robot):
    """A flying magic hammer."""

    def __init__(self,
                 simulator,
                 pose=[[0, 0, 0], [0, 0, 0]],
                 config=None):
        """Initialize.
        """
        self.config = config or self.default_config

        if not isinstance(pose, Pose):
            pose = Pose(pose)

        self._simulator = simulator
        self._body = None
        self._constraint = None
        self._initial_pose = pose
        self._time_step = self._simulator.time_step

        self._num_skill_steps = 5

        self.reboot()

    @property
    def pose(self):
        return self._constraint.pose

    def reboot(self):
        """Reboot the robot.
        """
        if self._body is not None:
            self._simulator.remove_body(self._body)

        if self._constraint is not None:
            self._simulator.remove_constraint(self._constraint)

        self._body = self._simulator.add_body(
            filename=self.config.URDF,
            pose=[[0, 0, 0], [0, 0, 0]],
            # pose=self._initial_pose,
            is_static=False,
            is_controllable=False,
            name='mjolnir_body')

        self._constraint = self._simulator.add_constraint(
            parent=self._body,
            child=None,
            parent_frame_pose=[[0, 0, 0], [0, 0, 0]],
            # parent_frame_pose=self._initial_pose,
            child_frame_pose=None,
            max_force=self.config.MAX_FORCE,
            is_controllable=True,
            name='mjolnir_constraint')

        self.move_to(self._initial_pose)

        # print('--')
        # print(self.pose)
        # print(self._body.pose)
        # print(self._constraint.pose)

    def reset(self, pose=None):
        """Reset the robot.

        Args:
            pose: The target pose.       
            """
        self.pose = pose

    def move_to(self,
                pose,
                linear_velocity=None,
                angular_velocity=None,
                timeout=None,
                threshold=None):
        """Move the arm to the specified gripper pose.

        Args:
            pose: The target robot pose.
            timeout: Seconds to wait for move to finish.
            threshold: Position threshold in radians across each joint when move
                is considered successful.
        """
        if linear_velocity is None:
            linear_velocity = self.config.DEFAULT_LINEAR_VELOCITY
            
        if angular_velocity is None:
            angular_velocity = self.config.DEFAULT_ANGULAR_VELOCITY

        if timeout is None:
            timeout = self.config.TIMEOUT

        # Command the position control.
        kwargs = {
                'pose': pose,
                'linear_velocity': linear_velocity,
                'angular_velocity': angular_velocity,
                'timeout': timeout,
                }

        robot_command = RobotCommand(
                component=self._constraint.name,
                command_type='set_target_pose',
                arguments=kwargs)

        self._send_robot_command(robot_command)

    def is_ready(self):
        return self._constraint.is_ready()

    def _send_robot_command(self, robot_command):
        """Send a robot command to the server.

        Args:
            robot_command: An instance of RobotCommand.
        """
        self._simulator.receive_robot_commands(robot_command, 'constraint')
