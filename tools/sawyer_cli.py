#!/usr/bin/env python

"""Command line interface for Sawyer with Kinect2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import os.path
import time
import sys
from builtins import input

import cv2
import numpy as np  # NOQA
import matplotlib.pyplot as plt
import readline

import _init_paths  # NOQA
from robovat.math import Pose
from robovat.robots import sawyer
from robovat.perception.camera import Kinect2
from robovat.simulation import Simulator
from robovat.simulation.camera import BulletCamera
from robovat.utils import time_utils
from robovat.utils.logging import logger


GROUND_POSE = [[0, 0, -0.9], [0, 0, 0]]
TABLE_POSE = [[0.6, 0, 0.0], [0, 0, 0]]
OVERHEAD_POSITIONS = [-0.73, -1.13, 0.82, 1.51, -0.33, 1.34, 1.87]
LIMB_OUT_OF_VIEW_POSITIONS = [-1.5, -1.26, 0.00, 1.98, 0.00, 0.85, 3.3161]
# LIMB_OUT_OF_VIEW_POSITIONS = [-1.5, -1.38, 0.00, 2.18, 0.00, 0.57, 3.3161]

WELCOME = (
        '############################################################\n'
        'RoboVat Command Line Interface\n'
        'Authors: Kuan Fang, Andrey Kurenkov, Viraj Mehta\n'
        '############################################################\n'
        )

HELP = {
        'For now, please read the source code for instructions.'
        }


INTRINSICS = [372.66, 0., 241.20, 0., 368.62, 214.19, 0., 0., 1.]
TRANSLATION = [0, -0.60, 1.5]
ROTATION = [-3.1415, 0, 1.5708]


# Default z offset of the end effector.
Z_OFFSET = 0.4


class EndEffectorClickController: 

    def __init__(self, cli, ax):
        self.cli = cli
        self.depth = None

        self.ax = ax
        plt.ion()
        plt.show()

        self.show_image()

    def __call__(self, event):
        pixel = [event.xdata, event.ydata]
        z = self.depth[int(pixel[0]), int(pixel[1])] - Z_OFFSET
        position = self.cli.camera.deproject_pixel(pixel, z)
        pose = Pose([position, [np.pi, 0, 0]])

        position.z = 0.4  # TODO

        self.cli.robot.move_to_joint_positions(OVERHEAD_POSITIONS)
        while not (self.cli.robot.is_limb_ready() and
                   self.cli.robot.is_gripper_ready()):
            if self.cli.mode == 'sim':
                self.cli.simulator.step()

        print('clicked pixel: %r, moving end effector to: % s'
              % (pixel + [z], position))
        self.cli.robot.move_to_gripper_pose(pose)
        while not (self.cli.robot.is_limb_ready() and
                   self.cli.robot.is_gripper_ready()):
            if self.cli.mode == 'sim':
                self.cli.simulator.step()

        self.show_image()
        plt.scatter(pixel[0], pixel[1], c='r')

        return pixel

    def show_image(self):
        self.depth = self.cli.camera.frames()['depth']
        plt.imshow(self.depth)
        plt.title('Depth Image')
        plt.draw()
        plt.pause(1e-3)


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--mode',
            dest='mode',
            default='sim',
            help='Mode: sim, real.')

    parser.add_argument(
            '--data',
            type=str,
            dest='data_dir',
            default='./data',
            help='The data directory.')

    parser.add_argument(
            '--calib',
            type=str,
            dest='calib_dir',
            default=None,
            help='The calibration directory.')

    parser.add_argument(
            '--output',
            type=str,
            dest='output_dir',
            default=None,
            help='The output directory.')

    parser.add_argument(
            '--fps',
            type=float,
            dest='fps',
            default=5,
            help='The frame rate to record videos using the camera.')

    args = parser.parse_args()

    return args


class SawyerCLI(object):
    """Command line interface for Sawyer with Kinect2.
    """

    def __init__(self, mode, data_dir, calib_dir, output_dir=None, fps=5):
        """Initialize.

        Args:
            mode: 'sim' or 'real'.
            data_dir: The data directory.
            data_dir: The calibration directory.
            output_dir: The output directory.
            fps: The frame rate to record videos using the camera.
        """
        print(WELCOME)

        self.mode = mode
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fps = fps

        # Command line client input history.
        readline.parse_and_bind('tab: complete')
        history_file = os.path.join('.python_history')

        try:
            readline.read_history_file(history_file)
        except IOError:
            pass

        atexit.register(readline.write_history_file, history_file)

        # Set up the environment.
        if self.mode == 'sim':
            print('Setting up the environment in simulation...')
            self.simulator = Simulator(use_visualizer=True)
            self.simulator.reset()
            self.simulator.start()

            # Robot.
            self.robot = sawyer.SawyerSim(
                    self.simulator, joint_positions=LIMB_OUT_OF_VIEW_POSITIONS)

            # Ground.
            path = os.path.join(self.data_dir, 'sim',
                                'planes', 'plane.urdf')
            self.simulator.add_body(path, GROUND_POSE, is_static=True)

            # Table.
            path = os.path.join(self.data_dir, 'sim',
                                'tables', 'table_svl_wooden.urdf')
            self.simulator.add_body(path, TABLE_POSE, is_static=True)

            # Camera.
            # calibration_dir = os.path.join(self.data_dir, 'calibration',
            #                                'kinect')
            self.camera = BulletCamera(
                    simulator=self.simulator,
                    distance=1.0)

            if calib_dir is None:
                intrinsics = INTRINSICS
                translation = TRANSLATION
                rotation = ROTATION
            else:
                intrinsics, translation, rotation = (
                    self.camera.load_calibration(calib_dir))

            self.camera.set_calibration(intrinsics, translation, rotation)

        elif self.mode == 'real':
            print('Setting up the environment in the real simulator...')
            self.robot = sawyer.SawyerReal()
            self.camera = Kinect2(
                    packet_pipeline_mode=0,
                    device_num=0,
                    skip_registration=False,
                    use_inpaint=True)

        else:
            raise ValueError

        # Start the camera camera.
        self.camera.start()

    def start(self):
        """Start the command line client.
        """
        last_recording_time = float('-inf')

        while (1):
            if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
                sys.stdout.flush()
                command = input('Enter a command: ')

                if command == 'quit' or command == 'q':
                    print('Closing the Sawyer client...')
                    break
                else:
                    self.run_command(command)

            if self.mode == 'sim':
                self.simulator.step()

            if self.output_dir is not None:
                t = time.time()
                if t >= last_recording_time + 1 / float(self.fps):
                    self.save_camera_image()
                    last_recording_time = t

    def save_camera_image(self):
        """Save the current camera image.
        """
        timestamp = time_utils.get_timestamp_as_string()

        results = self.camera.frames()
        image = results['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        filename = os.path.join(self.output_dir, '%s.png' % (timestamp))
        cv2.imwrite(filename, image)

    def run_command(self, command):
        """Run the input command.

        Args:
            command: An input string command.
        """
        command = command.replace(',', '').replace('[', '').replace(']', '')
        words = command.split(' ')
        command_type = words[0]

        # Print the help information.
        if command_type == 'help' or command_type == 'h':
            print(HELP)

        # Reset the robot joint positions.
        elif command_type == 'reset' or command_type == 'r':
            self.robot.reset(LIMB_OUT_OF_VIEW_POSITIONS)

        # Move joints to the target positions.
        elif command_type == 'joints' or command_type == 'j':
            joint_positions = [float(ch) for ch in words[1:]]
            print('Moving to joint positions: %s ...' % joint_positions)
            self.robot.move_to_joint_positions(joint_positions)

        # Move the end effector to the target pose.
        elif command_type == 'end_effector' or command_type == 'e':
            pose = [float(ch) for ch in words[1:]]
            if len(pose) == 6 or len(pose) == 7:
                pose = Pose(pose[:3], pose[3:])
            elif len(pose) == 3:
                end_effector_pose = self.robot.end_effector
                pose = Pose(pose, end_effector_pose.orientation)
            else:
                print('The format of the input pose is wrong.')

            print('Moving to end effector pose: %s ...' % pose)
            self.robot.move_to_gripper_pose(pose)

        # Move the end effector to the target pose.
        elif command_type == 'end_effector_line' or command_type == 'el':
            pose = [float(ch) for ch in words[1:]]
            if len(pose) == 6 or len(pose) == 7:
                pose = Pose(pose[:3], pose[3:])
            elif len(pose) == 3:
                end_effector_pose = self.robot.end_effector
                pose = Pose(pose, end_effector_pose.orientation)
            else:
                print('The format of the input pose is wrong.')

            print('Moving to end effector pose: %s ...' % pose)
            self.robot.move_to_gripper_pose(pose, straight_line=True)

        # Open the gripper.
        elif command_type == 'open' or command_type == 'o':
            joint_positions = self.robot.grip(0)

        # Close the gripper.
        elif command_type == 'quit' or command_type == 'q':
            joint_positions = self.robot.grip(1)

        # Print the current robot status.
        elif command_type == 'print' or command_type == 'p':
            joint_positions = self.robot.joint_positions
            joint_positions = [
                    joint_positions['right_j0'],
                    joint_positions['right_j1'],
                    joint_positions['right_j2'],
                    joint_positions['right_j3'],
                    joint_positions['right_j4'],
                    joint_positions['right_j5'],
                    joint_positions['right_j6'],
                    ]
            print('Joint positions: %s' % (joint_positions))

            end_effector_pose = self.robot.end_effector
            print('End Effector position: %s' % (end_effector_pose))

        # Visualize the camera image.
        elif command_type == 'visualize' or command_type == 'v':
            results = self.camera.frames()
            image = results['rgb']
            depth = results['depth']

            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(image)
            plt.title('RGB Image')
            plt.subplot(122)
            plt.imshow(depth)
            plt.title('Depth Image')

            end_effector_pose = self.robot.end_effector
            pixel = self.camera.project_point(end_effector_pose.position)
            plt.scatter(pixel[0], pixel[1], c='r')

            plt.show()

        # Move the gripper to the clicked pixel position.
        elif command_type == 'click' or command_type == 'c':

            fig, ax = plt.subplots(figsize=(20, 10))
            onclick = EndEffectorClickController(self, ax)

            results = self.camera.frames()
            depth = results['depth']
            plt.imshow(depth)
            plt.title('Depth Image')
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

        else:
            print('Unrecognized command: %s' % command)


def main():
    args = parse_args()

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            logger.info('Making directory: %s...' % (args.output_dir))
            os.makedirs(args.output_dir)

    logger.info('Creating the Sawyer command line client...')
    sawyer_cli = SawyerCLI(
            args.mode,
            args.data_dir,
            args.calib_dir,
            args.output_dir,
            args.fps)

    logger.info('Running the Sawyer command line client...')
    sawyer_cli.start()


if __name__ == '__main__':
    main()
