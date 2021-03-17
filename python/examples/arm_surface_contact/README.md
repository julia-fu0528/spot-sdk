<!--
Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.

Downloading, reproducing, distributing or otherwise using the SDK Software
is subject to the terms and conditions of the Boston Dynamics Software
Development Kit License (20191101-BDSDK-SL).
-->

# Arm Surface Contact

This example program shows using the ArmSurfaceContactClient to do an accurate position move of
the end-effector with some amount of force. It demonstrates how to
initialize the SDK to talk to robot, use RobotCommandClient to request a stand, and use 
ArmSurfaceContactClient to request an end-effector trajectory move with some force on the ground.


## Understanding Spot Programming
For your best learning experience, please use the [Quickstart Guide](../../../docs/python/quickstart.md)
found in the SDK's docs/python directory.  That will help you get your Python programming environment set up properly.  

## Common Problems
1. Remember, you will need to launch a software e-stop separately.  The E-Stop programming example is [here](../estop/README.md).
2. Make sure the Motor Enable button on the Spot rear panel is depressed.
3. Make sure Spot is sitting upright, with the battery compartment on the side closest the floor. 

## Setup Dependencies
This example requires the bosdyn API and client to be installed, and must be run using python3. Using pip, these dependencies can be installed using:

```
python3 -m pip install -r requirements.txt
```
## Run the Example
To run the example:
```
python3 arm_surface_contact.py --username USER --password PASSWORD ROBOT_IP
```