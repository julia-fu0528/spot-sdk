# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple robot state capture tutorial."""

import sys
import json
import os
import numpy as np
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient


def main():
    import argparse

    commands = {'state', 'hardware', 'metrics'}

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('command', choices=list(commands), help='Command to run')
    options = parser.parse_args()

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)


    dict_dir = "data/20241119"
    os.makedirs(dict_dir, exist_ok=True)
    # Make a robot state request
    if options.command == 'state':
        state = robot_state_client.get_robot_state()
        # create a dictionary with all the keys the same as state bu the values as empty lists
        state_dict = []
        try:
            while True:
                state = robot_state_client.get_robot_state()
                state_dict.append(state)
        
        # save data whe press ctrl+c
        except KeyboardInterrupt as e:
            np.save(os.path.join(dict_dir, "test.npy"), state_dict)

    elif options.command == 'hardware':
        print(robot_state_client.get_hardware_config_with_link_info())
    elif options.command == 'metrics':
        print(robot_state_client.get_robot_metrics())

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)
