
import argparse
from datetime import datetime
import json
import numpy as np
# import pybullet as p
import matplotlib.pyplot as plt
import os
import trimesh
import trimesh.viewer
from pathlib import Path
import xml.etree.ElementTree as ET
import random
from collections import defaultdict
from urdfpy import URDF
from scipy.spatial import cKDTree
import open3d as o3d
from store_robot_state import collect_data

# from pykdl_utils.kdl_kinematics import KDLKinematics


def vis_joint_torques(torque_path):
    # load the npy data
    # torque_path = "data/touch_back.npy"
    state = np.load(torque_path, allow_pickle=True)
    state_dict = {}
    torque_dict = {}
    for i in range(len(state)):
        state_dict[i] = state[i].kinematic_state.joint_states
        torque_dict[i] = []
        for joint in state_dict[i]:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                if not joint_name.startswith("arm"):
                    # Store both the joint name and load value in the dictionary
                    torque_dict[i].append({
                        'name': joint_name,
                        'load': joint.load.value  # Assuming load has a 'value' attribute
                    })

    # Determine the number of entries and the maximum number of joints to dynamically handle varying joint counts
    num_entries = len(torque_dict)
    num_joints = max(len(torque_dict[i]) for i in torque_dict)

    # Initialize torque_data with NaN values in case different entries have different joint counts
    torque_data = np.full((num_entries, num_joints), np.nan, dtype=float)
    joint_names = []

    # Fill torque_data with torque values, ignoring missing joints for each entry
    for i in range(num_entries):
        for j, joint in enumerate(torque_dict[i]):
            torque_data[i, j] = joint['load']
            if i == 0:
                joint_names.append(joint['name'])

    # Create the plot
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(num_entries)

    for j in range(num_joints):
        plt.plot(time_steps, torque_data[:, j], label=f'Joint {joint_names[j]}')

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Torque")
    plt.title(f"Torque Over Time for Each Joint for {torque_path.split('.')[0]}")
    plt.legend()
    plt.grid(True)

    output_dir = "vis"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{torque_path.split('.')[0].split('/')[-1]}.png")
    print(f"save_path: {save_path}")
    plt.savefig(save_path, format="png", dpi=300)  # Save as a PNG file with 300 dpi resolution

    plt.show()

def load_spot():
    robot = URDF.load('spot_description/spot.urdf')
    robot.show()

def add_red_dots(mesh, vertices, output_path, dot_radius=0.01):
    print(f"Number of input coordinates: {len(vertices)}")
    mesh_vertices = np.array(mesh.vertices)

    tree = cKDTree(mesh_vertices)
    distances, closest_indices = tree.query(vertices)

    print(f"Closest mesh vertices found for input coordinates: {len(closest_indices)}")

    red_dots = []
    for index in closest_indices:
        vertex = mesh_vertices[index]
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=dot_radius)
        sphere.visual.vertex_colors = [255, 0, 0, 255]  # Red in RGBA
        # sphere.visual.material = red_material

        # Translate the sphere to the vertex position
        sphere.apply_translation(vertex)
        red_dots.append(sphere)

    print(f"Red dots created: {len(red_dots)}")

    # Combine the red dots with the original mesh (if needed, optional)
    combined_mesh = trimesh.util.concatenate([mesh] + red_dots)
    combined_mesh.export(output_path)
    print(f"Mesh with red dots exported to {output_path}")

    return red_dots

def load_spot_with_red_dots():
    # get the folder path of this file
    folder_path = Path(__file__).parent
    # Paths
    scene_path = os.path.join(folder_path, 'spot_description/meshes/base/visual/body.obj')
    updated_path = os.path.join(folder_path, 'spot_description/meshes/base/visual/body_updated.obj')

    scene = trimesh.load(scene_path, force='mesh')
    if hasattr(scene.visual, 'material'):
        scene.visual.material.alpha = 1.0  # Fully opaque

    # Convert to opaque vertex colors if no material
    if hasattr(scene.visual, 'to_color'):
        scene.visual = scene.visual.to_color()
        scene.visual.vertex_colors[:, 3] = 255  # Fully opaque
    if isinstance(scene, trimesh.Scene):
        # Combine all geometry into a single mesh if needed
        mesh = trimesh.util.concatenate([scene.geometry[g] for g in scene.geometry])
    else:
        mesh = scene

    # Convert Trimesh to Open3D
    o3d_mesh_list = convert_trimesh_to_open3d([mesh])
    if not o3d_mesh_list:
        raise ValueError("Failed to convert Trimesh to Open3D.")
    # Debug Open3D Mesh
    o3d_mesh = o3d_mesh_list[0]

    # Visualize in Open3D
    # o3d.visualization.draw_geometries([o3d_mesh])
    # sampled_points, face_indices = trimesh.sample.sample_surface_even(mesh, count=1000)
    vertices = []
    faces = []

    with open(scene_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex line
                vertices.append([float(v) for v in line.split()[1:]])
            elif line.startswith('f '):  # Face line
                faces.append([int(f.split('/')[0]) - 1 for f in line.split()[1:]])
    vertices = trimesh.points.PointCloud(vertices)  # Convert to a Trimesh PointCloud for easier spatial operations

    # Group vertices by spatial orientation
    groups = defaultdict(list)
    for i, (x, y, z) in enumerate(vertices.vertices):
        if z > 0.08:  # Top
            groups['top'].append(i)
        elif x > 0.4:  # Front
            groups['front'].append(i)
        elif x < -0.41:  # Back
            groups['back'].append(i)
        elif y > 0.1:  # Right
            groups['right'].append(i)
        elif y < -0.1:  # Left
            groups['left'].append(i)

    # Sample 3 vertices from each group
    selected_vertices = []
    chosen_vertices = []

    # for group_name, indices in groups.items():
    #     # Get the vertex coordinates for the current group
    #     group_vertices = np.array([vertices.vertices[i] for i in indices])
    #     print(f"group_vertices: {len(group_vertices)}")
    #     # Determine the axis to sort by based on the face
    #     if group_name in ['top']:
    #         sort_axis = 0  # Sort by X (horizontal spread)
    #     elif group_name in ['left', 'right']:
    #         sort_axis = 1  # Sort by Y (depth spread)
    #     elif group_name in ['front', 'back']:
    #         sort_axis = 2  # Sort by Z (vertical spread)
    #     else:
    #         continue  # Skip if no clear sorting axis is defined

    #     # Sort vertices along the selected axis
    #     sorted_indices = group_vertices[:, sort_axis].argsort()
    #     sorted_group = group_vertices[sorted_indices]

    #     # Evenly select points from the sorted list
    #     num_points = 50  # Number of points to pick
    #     step = max(1, len(sorted_group) // num_points)  # Compute step size
    #     evenly_selected_indices = range(0, len(sorted_group), step)  # Pick points evenly


    #     # Map back to the original vertex indices
    #     # index
    #     selected_vertices.extend([indices[sorted_indices[idx]] for idx in evenly_selected_indices])
    #     # vertex
    #     chosen_vertices.extend([sorted_group[idx] for idx in evenly_selected_indices])

    chosen_vertices = np.array([[0.5, 0.0, -0.1], [-0.42, 0.0, 0.0], [0.0, 0.11, 0.0], [0.0, -0.11, 0.0]])
    robot = URDF.load(os.path.join(folder_path, 'spot_description/spot.urdf'))
    robot_meshes = []
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    # Load the mesh file (e.g., STL or OBJ)
                    mesh_path = os.path.join(folder_path, "spot_description", visual.geometry.mesh.filename)
                    o3d_mesh = convert_trimesh_to_open3d([trimesh.load(mesh_path, force='mesh')])[0]
                    if o3d_mesh== []:
                        print(f"Failed to load mesh from {mesh_path}")
                        continue
                    o3d_mesh.compute_vertex_normals()
                    robot_meshes.append(o3d_mesh)
    o3d.visualization.draw_geometries(robot_meshes)
    for chosen_vertex in chosen_vertices.tolist():
        # Add red dots to the selected vertices
        # red_dots = add_red_dots(mesh, [chosen_vertex], updated_path)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.compute_vertex_normals()

        sphere.paint_uniform_color([1, 0, 0])  # Red
        sphere.translate(chosen_vertex)
        geometry_list = [sphere] + robot_meshes
        o3d.visualization.draw_geometries(geometry_list)


        # print(f"Updated file written to {updated_path}")
        # robot.show()

def convert_trimesh_to_open3d(trimesh_fk):
        o3d_meshes = []
        for tm in trimesh_fk:
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(tm.vertices.copy()),
                triangles=o3d.utility.Vector3iVector(tm.faces.copy())
            )
            o3d_mesh.compute_vertex_normals()
            try:
                o3d_mesh.paint_uniform_color(tm.visual.material.main_color[:3] / 255.)
            except AttributeError:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(tm.visual.vertex_colors[:, :3] / 255.)

            o3d_mesh.transform(trimesh_fk[tm])
            # self.prev_fks.append(trimesh_fk[tm]) # world -> T1

            o3d_meshes.append(o3d_mesh)
        return o3d_meshes

def load_robot_from_urdf(urdf_path):
    """Load robot meshes from a URDF file and convert to Open3D objects."""
    robot = URDF.load(urdf_path)
    robot_meshes = []

    # Convert URDF links to Open3D meshes
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    # Resolve the path to the mesh file
                    mesh_file = os.path.join(os.path.dirname(urdf_path), visual.geometry.mesh.filename)
                    if not os.path.exists(mesh_file):
                        print(f"Mesh file not found: {mesh_file}")
                        continue
                    
                    # Load the mesh with Open3D
                    o3d_mesh = convert_trimesh_to_open3d([trimesh.load(mesh_file, force='mesh')])[0]
                    if o3d_mesh.is_empty():
                        print(f"Failed to load mesh from {mesh_file}")
                        continue
                    
                    o3d_mesh.compute_vertex_normals()
                    
                    # Apply transformation if origin is defined
                    # if visual.origin is not None:
                    #     transform = np.eye(4)
                    #     transform[:3, :3] = visual.origin.rotation
                    #     transform[:3, 3] = visual.origin.xyz
                    #     o3d_mesh.transform(transform)
                    
                    robot_meshes.append(o3d_mesh)
    
    return robot_meshes


def create_red_markers(marker_positions, radius=0.05):
    """Create red spheres as markers at specified positions."""
    red_markers = []
    for pos in marker_positions:
        # Create a sphere and set its position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        sphere.translate(pos)
        red_markers.append(sphere)
    return red_markers


def visualize_robot_with_markers(robot_meshes, markers):
    """Visualize the robot and red markers using Open3D."""
    geometry_list = robot_meshes + markers
    o3d.visualization.draw_geometries(geometry_list)

def compute_joint_transform(joint, joint_position):
    """
    Compute the transformation matrix for a given joint and its position.

    Args:
        joint (urdfpy.Joint): The joint object.
        joint_position (float): The position of the joint (angle for revolute, displacement for prismatic).

    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    transform = np.eye(4)  # Initialize as identity matrix

    # Apply the fixed transform from the joint's origin (if defined)
    if joint.origin is not None:
        origin_transform = np.eye(4)
        origin_transform = joint.origin
        # origin_transform[:3, :3] = joint.origin.rotation  # Rotation matrix from origin
        # origin_transform[:3, 3] = joint.origin.xyz        # Translation vector from origin
        transform = transform @ origin_transform

    # Compute joint-specific transformation
    if joint.joint_type in ["revolute", "continuous"]:
        # Rotation about the joint axis
        axis = np.array(joint.axis)
        angle = joint_position
        rotation_transform = trimesh.transformations.rotation_matrix(angle, axis)
        transform = transform @ rotation_transform
    elif joint.joint_type == "prismatic":
        # Translation along the joint axis
        axis = np.array(joint.axis)
        translation_transform = np.eye(4)
        translation_transform[:3, 3] = axis * joint_position
        transform = transform @ translation_transform
    elif joint.joint_type == "fixed":
        # No additional transformation for fixed joints
        pass
    else:
        raise ValueError(f"Unsupported joint type: {joint.joint_type}")

    return transform

def compute_forward_kinematics(robot, joint_positions):
    """
    Compute forward kinematics for a robot defined in a URDF file.

    Args:
        robot (URDF): The loaded URDF object.
        joint_positions (dict): Dictionary mapping joint names to their positions.

    Returns:
        dict: A dictionary mapping link names to their FK transformation matrices.
    """
    # Initialize the base transform as the identity matrix
    base_transform = np.eye(4)
    link_transforms = {robot.base_link.name: base_transform}

    # Map each joint to its parent and child link names
    joint_map = {joint.child: joint for joint in robot.joints}

    # Recursive function to compute FK for each link
    def compute_transform_for_link(link_name, parent_transform):
        if link_name in joint_map:
            # Get the joint connecting to this link
            joint = joint_map[link_name]
            joint_position = joint_positions.get(joint.name, 0.0)
            # Compute the joint transformation
            # joint_transform = joint.get_transform(joint_position)
            joint_transform = compute_joint_transform(joint, joint_position)
            current_transform = parent_transform @ joint_transform
        else:
            # This is the base link or a link with no parent joint
            current_transform = parent_transform

        # Store the transform for the current link
        link_transforms[link_name] = current_transform

        # Find child links connected via joints
        for joint in robot.joints:
            if joint.parent == link_name:
                compute_transform_for_link(joint.child, current_transform)

    # Start computation from the base link
    compute_transform_for_link(robot.base_link.name, base_transform)

    return link_transforms

def prepare_trimesh_fk(robot, link_fk_transforms):
    """
    Prepare trimesh objects and their transforms for convert_trimesh_to_open3d.
    Returns a dictionary with trimesh objects as keys and FK transforms as values.
    """
    trimesh_fk = {}
    folder_path = Path(__file__).parent

    for link in robot.links:
        for visual in link.visuals:
            if visual.geometry.mesh:
                # Load the mesh
                mesh_file = visual.geometry.mesh.filename
                mesh = trimesh.load(os.path.join(folder_path, 'spot_description', mesh_file), force='mesh')
                if hasattr(mesh.visual, 'material'):
                    mesh.visual.material.alpha = 1.0  # Fully opaque

                # Convert to opaque vertex colors if no material
                if hasattr(mesh.visual, 'to_color'):
                    mesh.visual = mesh.visual.to_color()
                    mesh.visual.vertex_colors[:, 3] = 255  # Fully opaque
                # Assign the FK transform
                transform = link_fk_transforms[link.name]
                trimesh_fk[mesh] = transform

    return trimesh_fk

if __name__ == "__main__":
    # torque_path = "data/20241119/test.npy"
    # vis_joint_torques(torque_path)
    # urdf_path = 'spot_description/spot.urdf'
    # # joint_locations = get_joint_locations(urdf_path)
    # get_joint_positions(urdf_path)


    # load_spot_with_red_dots()
    # urdf_path = 'spot_description/spot.urdf'
    robot = URDF.load('spot_description/spot.urdf')
    joint_positions = {joint.name: 0.0 for joint in robot.joints}  # Zero configuration
    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk = prepare_trimesh_fk(robot, link_fk_transforms)
    robot_meshes = convert_trimesh_to_open3d(trimesh_fk)
    marker_positions = {
        "front": [0.445, 0.0, 0.05],  # Front
        "back": [-0.42, 0.0, 0.02],  # Back
        "right": [0.0, 0.11, 0.0],  # Right
        "left": [0.0, -0.11, 0.0],  # Left
    }
    for place, marker in marker_positions.items():
        red_markers = create_red_markers([marker], radius = 0.01)
        print(f"PLEASE TOUCH SPOT AT {place.upper()}\n")
        o3d.visualization.draw_geometries(robot_meshes + red_markers)
        # get today's date
        # today = datetime.today().strftime('%Y%m%d')
        # collect_data(f"data/{today}/{place}.npy")
        collect_data(f"data/20241120/test.npy")


