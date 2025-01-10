from urdfpy import URDF
import open3d as o3d
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent  # Adjust the number of .parent calls based on your file location
sys.path.append(str(project_root))
from visualize_robot_state import compute_forward_kinematics, prepare_trimesh_fk, convert_trimesh_to_open3d, combine_meshes_o3d




def main():

    robot = URDF.load('fr3_franka_hand.urdf')
    joint_positions = {joint.name: 0.0 for joint in robot.joints}
    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk, _ = prepare_trimesh_fk(robot, link_fk_transforms, folder="franka")
    robot_meshes, _ = convert_trimesh_to_open3d(trimesh_fk)
    combined_robot_mesh = combine_meshes_o3d(robot_meshes)
    o3d.visualization.draw_geometries(robot_meshes)

if __name__ == '__main__':
    main()