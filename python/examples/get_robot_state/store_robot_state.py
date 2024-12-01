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
from visualize_robot_state import add_red_dots, compute_forward_kinematics, prepare_trimesh_fk, \
convert_trimesh_to_open3d, create_red_markers, visualize_robot_with_markers, combine_meshes_o3d


def collect_data(output_path, hostname, command):
    

    # dict_dir = '/'.join(output_path.split("/")[:-1])

    # dict_dir = "data/20241120"
    # os.makedirs(dict_dir, exist_ok=True)
    sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # Make a robot state request
    if command == 'state':
        state = robot_state_client.get_robot_state()
        # create a dictionary with all the keys the same as state bu the values as empty lists
        state_dict = []
        start_time = time.time()

        # try:
            # while True:
        while time.time() - start_time < 10:
            state = robot_state_client.get_robot_state()
            state_dict.append(state)
        
        # save data whe press ctrl+c
        # except KeyboardInterrupt as e:
        np.save(output_path, state_dict)

    elif command == 'hardware':
        print(robot_state_client.get_hardware_config_with_link_info())
    elif command == 'metrics':
        print(robot_state_client.get_robot_metrics())

    return True

def main(output_dir):
    import argparse

    commands = {'state', 'hardware', 'metrics'}

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('command', choices=list(commands), help='Command to run')
    options = parser.parse_args()

    # Create robot object with an image client.
    hostname = options.hostname
    command = options.command

    robot = URDF.load('spot_description/spot.urdf')
    joint_positions = {joint.name: 0.0 for joint in robot.joints}  # Zero configuration
    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk = prepare_trimesh_fk(robot, link_fk_transforms)
    robot_meshes = convert_trimesh_to_open3d(trimesh_fk)
    # combined_robot_mesh = combine_meshes_o3d(robot_meshes)
    # print(f"Got combined mesh\n")
    # marker_positions = {
    #     "front": [0.445, 0.0, 0.05],  # Front
    #     "back": [-0.42, 0.0, 0.05],  # Back
    #     "left": [0.0, 0.11, 0.0],    # Left
    #     "right": [0.0, -0.11, 0.0],}  # Right
    markers_pos = add_red_dots(robot_meshes[0], 100, radius = 0.01)
    print(f"Total number of markers: {len(markers_pos)}")
    # o3d.visualization.draw_geometries(robot_meshes + markers)
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print(f"marker positions: {marker_positions}")
    # for place, marker in marker_positions.items():
        # red_marker = create_red_markers([marker], radius = 0.01)
        # print(f"YOU CAN TOUCH THE SPOT NOW.\n")
        # o3d.visualization.draw_geometries(robot_meshes + red_marker)
    #     collect_data(output_path, hostname, command)
    #     print(f"Touch Data Collected for {place.upper()}, saved in {output_path}\n")
    # visualize_robot_with_markers(robot_meshes, marker_positions)
    for idx, pos in marker_positions.items():
        # Create marker for current position
        marker = create_red_markers([pos], radius=0.02)[0]
        print(f"YOU CAN TOUCH THE SPOT NOW\n")
        
        # Create camera parameters facing the marker
        camera_params = create_viewing_parameters(pos)
        
        # Combine geometries
        geometries = robot_meshes + [marker]
        
        print(f"Viewing marker {idx} . Press Ctrl+C in terminal to proceed to next view.")
        
        # Visualize with specific camera view
        visualize_with_camera(geometries, camera_params)
        user_input = input(f"Is marker position {idx} legit? Enter 'y' for yes, 'n' for no (default: 'y'): \n").strip().lower()
    
        if user_input == 'n':
            print(f"Marker position {idx} deemed not legit. Skipping this marker...\n")
            continue
        output_path = os.path.join(output_dir, f"{idx}.npy")
        # collect_data(output_path, hostname, command)
        print(f"Touch Data Collected for marker position {idx}, saved in {output_path}\n")
        

    


if __name__ == '__main__':
    output_dir = "data/202412"
    os.makedirs(output_dir, exist_ok=True)
    main(output_dir)
    # if not main(output_path):
    #     sys.exit(1)