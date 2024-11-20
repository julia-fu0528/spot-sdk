import json
import numpy as np
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

def parse_urdf_joints(urdf_file):
    """Parse joint positions from URDF file."""
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    joint_info = {}
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        origin = joint.find('origin')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        if origin is not None:
            xyz = origin.get('xyz')
            if xyz:
                x, y, z = map(float, xyz.split())
                joint_info[joint_name] = {
                    'position': np.array([x, y, z]),
                    'parent': parent,
                    'child': child
                }
    return joint_info



def get_joint_positions(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    joint_positions = {}
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        origin = joint.find('origin')
        if origin is not None:
            xyz = origin.get('xyz')
            if xyz:
                x, y, z = map(float, xyz.split())
                joint_positions[joint_name] = (x, y, z)

    # Print results
    for joint_name, (x, y, z) in joint_positions.items():
        print(f"{joint_name}: x={x}, y={y}, z={z}")
    
    return joint_positions

def find_closest_vertices(scene_path, joint_positions):
    # Load the OBJ file, which may be a Scene
    scene = trimesh.load(scene_path)
    
    # If it's a Mesh (not a Scene), wrap it in a dictionary for consistency
    if isinstance(scene, trimesh.Trimesh):
        scene = {'mesh': scene}

    closest_vertices = {}

    # For each joint, find the closest vertex in any mesh of the scene
    for joint_name, joint_position in joint_positions.items():
        min_distance = float('inf')
        closest_vertex = None

        # Iterate over all meshes in the scene
        for name, mesh in scene.geometry.items():
            # Compute the distance from the joint to each vertex in this mesh
            distances = np.linalg.norm(mesh.vertices - joint_position, axis=1)
            # Find the closest vertex in this mesh
            closest_vertex_idx = np.argmin(distances)
            closest_vertex_in_mesh = mesh.vertices[closest_vertex_idx]
            closest_distance_in_mesh = distances[closest_vertex_idx]

            # Check if this is the closest vertex overall
            if closest_distance_in_mesh < min_distance:
                min_distance = closest_distance_in_mesh
                closest_vertex = closest_vertex_in_mesh

        closest_vertices[joint_name] = closest_vertex

    return closest_vertices

def set_vertex_color_red(scene_path, closest_vertices):
    # Load the OBJ file, which may be a Scene
    scene = trimesh.load(scene_path)

    # If it's a Mesh (not a Scene), wrap it in a dictionary for consistency
    if isinstance(scene, trimesh.Trimesh):
        scene = {'mesh': scene}

    # Iterate over each mesh in the scene
    for mesh_name, mesh in scene.geometry.items():
        # Ensure vertex colors are available
        if not hasattr(mesh.visual, 'vertex_colors'):
            # Convert TextureVisuals to ColorVisuals by assigning white color to all vertices
            mesh.visual = mesh.visual.to_color()
            mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4), dtype=np.uint8) * 255  # white (RGBA)

        # Find and set color for each closest vertex
        for joint_name, closest_vertex in closest_vertices.items():
            # Calculate distances from all vertices to the closest vertex
            distances = np.linalg.norm(mesh.vertices - closest_vertex, axis=1)
            # Find the index of the closest vertex in this mesh
            closest_vertex_idx = np.argmin(distances)

            # Set the color of this vertex to red (RGBA: [255, 0, 0, 255])
            mesh.visual.vertex_colors[closest_vertex_idx] = [255, 0, 0, 255]
    trimesh.viewer.SceneViewer(scene)
    return scene

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
    print(f"Trimesh vertices: {len(mesh.vertices)}")
    print(f"Trimesh faces: {len(mesh.faces)}")

    # Convert Trimesh to Open3D
    o3d_mesh_list = convert_trimesh_to_open3d([mesh])
    if not o3d_mesh_list:
        raise ValueError("Failed to convert Trimesh to Open3D.")
    # Debug Open3D Mesh
    o3d_mesh = o3d_mesh_list[0]
    print(o3d_mesh)

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
    print(f"chosen_vertices: {len(chosen_vertices.tolist())}")
    robot = URDF.load(os.path.join(folder_path, 'spot_description/spot.urdf'))
    for chosen_vertex in chosen_vertices.tolist():
        # Add red dots to the selected vertices
        # red_dots = add_red_dots(mesh, [chosen_vertex], updated_path)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.compute_vertex_normals()
        sphere.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(len(mesh.vertices), 3))
        o3d.visualization.draw_geometries([sphere])

        # sphere.paint_uniform_color([1, 0, 0])  # Red
        print("hi")
        sphere.translate(chosen_vertex)

        o3d.visualization.draw_geometries([sphere, robot])


        # print(f"Updated file written to {updated_path}")
        # robot.show()

def convert_trimesh_to_open3d(trimesh_fk):
        o3d_meshes = []
        for tm in trimesh_fk:
            print(f"tm.vertices: {tm.faces}")
            faces = np.asarray(tm.faces, dtype=np.int32)
            vertices = np.asarray(tm.vertices, dtype=np.float64)

            # Validate faces
            if faces.max() >= len(vertices):
                print("Error: Face indices exceed number of vertices")
                # Either trim the faces or fix the data
                valid_faces = faces[faces.max(axis=1) < len(vertices)]
                faces = valid_faces

            # Create mesh with validated data
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
            try:
                o3d_mesh.paint_uniform_color(tm.visual.material.main_color[:3] / 255.)
            except AttributeError:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(tm.visual.vertex_colors[:, :3] / 255.)

            # o3d_mesh.transform(trimesh_fk[tm])
            # o3d_mesh.transform(tm)
            # self.prev_fks.append(trimesh_fk[tm]) # world -> T1

            o3d_meshes.append(o3d_mesh)
        return o3d_meshes

def urdf_to_open3d(robot, folder_path):
    """
    Converts URDF links to Open3D TriangleMesh objects.
    
    Args:
        robot (URDF): Parsed URDF object.
        folder_path (str): Base path to URDF file and related meshes.
    
    Returns:
        list: List of Open3D geometries representing the URDF.
    """
    o3d_objects = []
    
    for link in robot.links:
        for visual in link.visuals:
            # Get mesh file path
            if visual.geometry.mesh is not None:
                mesh_file = os.path.join(folder_path, visual.geometry.mesh.filename)
                if not os.path.exists(mesh_file):
                    print(f"Mesh file not found: {mesh_file}")
                    continue
                
                # Load the mesh using Open3D
                mesh = o3d.io.read_triangle_mesh(mesh_file)
                mesh.compute_vertex_normals()
                
                # Apply visual origin (translation + rotation) if specified
                if visual.origin is not None:
                    transform = np.eye(4)
                    transform[:3, :3] = visual.origin.rotation
                    transform[:3, 3] = visual.origin.xyz
                    mesh.transform(transform)
                
                # Add the mesh to the list of Open3D objects
                o3d_objects.append(mesh)
    
    return o3d_objects

if __name__ == "__main__":
    # torque_path = "data/20241119/test.npy"
    # vis_joint_torques(torque_path)
    # urdf_path = 'spot_description/spot.urdf'
    # # joint_locations = get_joint_locations(urdf_path)
    # get_joint_positions(urdf_path)


    load_spot_with_red_dots()
    # scene_path = 'spot_description/meshes/base/visual/body.obj'
    # vertex_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Replace with desired indices



    # # load_spot_with_red_dots(joint_names_to_highlight)
    # urdf_path = 'spot_description/spot.urdf'
    # mesh_path = 'spot_description/meshes/base/visual/body.obj'

    # # Get joint positions from the URDF
    # joint_positions = get_joint_positions(urdf_path)
    # # Find the closest vertices on the mesh for each joint position
    # closest_vertices = find_closest_vertices(mesh_path, joint_positions)
    # scene_with_red_vertices = set_vertex_color_red(mesh_path, closest_vertices)
    # scene_with_red_vertices.export('modified_body.obj')

    # # Print the closest vertices for each joint
    # for joint_name, vertex in closest_vertices.items():
    #     print(f"Joint: {joint_name}, Closest Vertex: {vertex}")