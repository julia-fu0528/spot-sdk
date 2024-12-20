# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple robot state capture tutorial."""

import sys
import time
import json
import os
import numpy as np
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from urdfpy import URDF
import open3d as o3d
import random

from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera
from visualize_robot_state import find_closest_vertices, add_red_dots, compute_forward_kinematics, prepare_trimesh_fk, \
convert_trimesh_to_open3d, create_red_markers, visualize_robot_with_markers, combine_meshes_o3d


def collect_data(output_path, hostname, command, duration=10):
    sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # Make a robot state request
    if command == 'state':
        print("Collecting data\n")
        state = robot_state_client.get_robot_state()
        # create a dictionary with all the keys the same as state bu the values as empty lists
        state_dict = []
        start_time = time.time()

        # try:
            # while True:
        while time.time() - start_time < duration:
            state = robot_state_client.get_robot_state()
            state_dict.append(state)
        
        # save data whe press ctrl+c
        # except KeyboardInterrupt as e:
        print(f"Data collection complete, saved in {output_path}\n")
        np.save(output_path, state_dict)

    elif command == 'hardware':
        print(robot_state_client.get_hardware_config_with_link_info())
    elif command == 'metrics':
        print(robot_state_client.get_robot_metrics())

    return True

def main():
    import argparse

    commands = {'state', 'hardware', 'metrics'}

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('command', choices=list(commands), help='Command to run')
    parser.add_argument('--output_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--duration', type=int, default=10, help='Duration to collect data')
    options = parser.parse_args()

    # Create robot object with an image client.
    hostname = options.hostname
    command = options.command
    output_dir = options.output_dir
    markers_path = options.markers_path
    duration = options.duration
    os.makedirs(output_dir, exist_ok=True)

    robot = URDF.load('spot_description/spot.urdf')
    joint_positions = {joint.name: 0.0 for joint in robot.joints}  # Zero configuration
    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk = prepare_trimesh_fk(robot, link_fk_transforms)
    robot_meshes = convert_trimesh_to_open3d(trimesh_fk)
    # combined_robot_mesh = combine_meshes_o3d(robot_meshes)
    # markers_pos = [
    #     # front
    #     [0.45, 0.06, -0.035],
    #     [0.45, -0.07, -0.035],
    #     # back
    #     [-0.45, 0.05, 0.05],
    #     [-0.45, -0.05, 0.05],
    #     # left
    #     [0.13, 0.14, 0.01],
    #     [-0.13, 0.14, -0.01],
    #     # right
    #     [0.1, -0.15, -0.01],
    #     [-0.13, -0.14, -0.01],
    #     # top
    #     [0.1, 0.05, 0.09],
    #     [-0.12, -0.01, 0.09],
    # ]
    # # LEFT: 24
    # for i in range(8):
    #     x = -0.2 + i / 8 * (0.25 - (-0.2))
    #     for j in range(3):
    #         z = -0.04 + j / 3 * (0.1 - (-0.04))
    #         markers_pos.append([x, 0.105, z])
    # # RIGHT: 24
    # for i in range(8):
    #     x = -0.2 + i / 8 * (0.25 - (-0.2))
    #     for j in range(3):
    #         z = -0.04 + j / 3 * (0.1 - (-0.04))
    #         markers_pos.append([x, -0.105, z])
    # # TOP: 24
    # for i in range(6):
    #     x = -0.37 + i / 6 * (0.04 - (-0.37))
    #     for j in range(3):
    #         y = -0.04 + j/3 * (0.08 - (-0.04))
    #         markers_pos.append([x, y, 0.08])
    # markers_pos.append([-0.2, 0.08, 0.08])
    # markers_pos.append([-0.12, 0.08, 0.08])
    # markers_pos.append([0.01, 0.08, 0.08])
    # markers_pos.append([-0.2, -0.07, 0.08])
    # markers_pos.append([0.01, -0.07, 0.08])
    # markers_pos.append([0.1, -0.07, 0.08])
    # # FRONT: 8
    # for i in range(4):
    #     y = -0.07 + i / 4 * (0.11 - (-0.07))
    #     markers_pos.append([0.39, y, -0.07])
    #     markers_pos.append([0.44, y, 0.04])
    # # BACK: 10
    # for i in range(4):
    #     y = -0.07 + i / 4 * (0.11 - (-0.07))
    #     markers_pos.append([-0.40, y, -0.06])
    #     markers_pos.append([-0.42, y, 0.04])
    # markers_pos.append([-0.42, -0.07, -0.01])
    # markers_pos.append([-0.42, 0.07, -0.01])
    # markers_pos = np.array(markers_pos)
    # num_points = 10000
    # markers_pos, pos_indices = find_closest_vertices(robot_meshes[0], markers_pos, num_points)
    # np.savetxt(markers_path, markers_pos, delimiter=",", comments="")
    markers_pos = np.loadtxt(markers_path, delimiter=",")
    print("Loaded markers positions: ", markers_pos)
    print(f"Total number of markers: {len(markers_pos)}")
    # markers = create_red_markers(markers_pos, radius=0.02)
    # o3d.visualization.draw_geometries(robot_meshes + markers[26:28])
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print(f"marker positions: {marker_positions}")
    # print("DON'T TOUCH YET! COLLECTING NO CONTACT DATA")
    # collect_data(os.path.join(output_dir, f"no_contact.npy"), hostname, command, duration)
    # os.makedirs("data/test1203", exist_ok=True)
    for idx, pos in marker_positions.items():
        if int(idx)<10:
            continue
        # Create marker for current position
        marker = create_red_markers([pos], radius=0.02)[0]
        
        # Create camera parameters facing the marker
        camera_params = create_viewing_parameters(pos)
        
        # Combine geometries
        geometries = robot_meshes + [marker]
        
        print(f"Viewing marker {idx} . Press Ctrl+C in terminal to proceed to next view.")
        
        # Visualize with specific camera view
        visualize_with_camera(geometries, camera_params)
        # user_input = input(f"Is marker position {idx} legit? Enter 'y' for yes, 'n' for no (default: 'y'): \n").strip().lower()
    
        # if user_input == 'n':
        #     print(f"Marker position {idx} deemed not legit. Skipping this marker...\n")
        #     continue
        output_path = os.path.join(output_dir, f"{idx}.npy")
        print(f"YOU CAN TOUCH THE SPOT NOW. Data collection will start in 5 seconds, please make sure you are touching the Spot.\n")
        time.sleep(4)
        collect_data(output_path, hostname, command, duration)
        print(f"Touch Data Collected for marker position {idx}, saved in {output_path}\n")
        

    


if __name__ == '__main__':
    main()
    # if not main(output_path):
    #     sys.exit(1)
