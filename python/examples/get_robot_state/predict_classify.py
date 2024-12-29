import os
import time
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree
from natsort import natsorted
from urdfpy import URDF
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.utils.visualizer import SpotVisualizer
from src.utils.helpers import sample_points_from_mesh
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera
from visualize_robot_state import update_meshes_with_fk, combine_meshes_o3d, create_red_markers, compute_forward_kinematics, find_closest_vertices, load_joint_torques, prepare_trimesh_fk, convert_trimesh_to_open3d

def collect_realtime_data(robot_state_client, duration=2):
    """
    Collect real-time data from the robot for a specified duration.
    """
    start_time = time.time()
    data = []

    while time.time() - start_time < duration:
        state = robot_state_client.get_robot_state()
        data.append(state)

    return np.array(data)

def preprocess_realtime_data(data, offset, markers_path, normalize=True):
    """
    Preprocess the real-time data for inference.
    """
    markers_pos = np.loadtxt(markers_path, delimiter=",")
    state_dict = data.kinematic_state.joint_states
    torque_dict = []
    for joint in state_dict:
        joint_name = getattr(joint, 'name', None)
        if joint_name is not None:
            if not joint_name.startswith("arm"):
                torque_dict.append({
                    'name': joint_name,
                    'position': joint.position.value,
                    'load': joint.load.value  
                })
    num_joints = len(torque_dict)
    torque_data = np.full((1, num_joints), np.nan, dtype=float)
    pos_data = np.full((1, num_joints), np.nan, dtype=float)
    joint_names = []
    for j, joint in enumerate(torque_dict):
            torque_data[0, j]  = joint['load']
            pos_data[0, j] = joint['position']
            joint_names.append(joint['name'])
    data = np.hstack((torque_data, pos_data))
    if normalize:
        data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
    data_processed = np.array(data)

    return data_processed

def infer_realtime(model, data, classes):
    """
    Perform real-time inference using the trained model.
    """
    predictions = model.predict(data)

    if predictions.shape[0] == 2:  # Multiple samples
        predicted_classes = [np.argmax(pred) for pred in predictions]
        
        # Find the most common class
        most_common_class_index, count = Counter(predicted_classes).most_common(1)[0]
        predicted_class = classes[most_common_class_index]
        
        # Confidence can be calculated as the proportion of predictions for the most common class
        confidence = count / len(predicted_classes)
    else:  # Single sample
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class = classes[predicted_class_index]

    print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
    return predicted_class, confidence


def visualize_prediction(marker_positions, predicted_class, robot_meshes):
    """
    Visualize the prediction on the robot with markers.
    """
    pos = marker_positions.get(predicted_class)
    marker = create_red_markers([pos], radius=0.04)[0]
    if marker is None:
        print(f"No marker for predicted class: {predicted_class}")
        return

    # Create camera parameters for the marker
    camera_params = create_viewing_parameters(pos)
    geometries = robot_meshes + [marker]
    # Visualize the robot with the predicted marker
    print(f"Visualizing prediction for class: {predicted_class}")
    visualize_with_camera(geometries, camera_params)

def main():
    import argparse
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client import create_standard_sdk
    import bosdyn.client.util

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--data_dir', required=True, help='Path to the directory containing torque data')
    options = parser.parse_args()

     # Load the trained model
    print("Loading the model...")
    model = load_model(options.model_path)
    print("Model loaded successfully.")

    # Initialize robot and client
    sdk = create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

     # Load marker positions
    markers_path = options.markers_path
    markers_pos = np.loadtxt(markers_path, delimiter=",")
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
    simplified_to_full_name = {
        'fl.hx': 'front_left_hip_x',
        'fr.hx': 'front_right_hip_x',
        'hl.hx': 'rear_left_hip_x',
        'hr.hx': 'rear_right_hip_x',
        'fl.hy': 'front_left_hip_y',
        'fr.hy': 'front_right_hip_y',
        'hl.hy': 'rear_left_hip_y',
        'hr.hy': 'rear_right_hip_y',
        'fl.kn': 'front_left_knee',
        'fr.kn': 'front_right_knee',
        'hl.kn': 'rear_left_knee',
        'hr.kn': 'rear_right_knee',
        'arm0.sh0': 'arm_sh0',
        'arm0.sh1': 'arm_sh1',
        'arm0.el0': 'arm_el0',
        'arm0.el1': 'arm_el1',
        'arm0.wr0': 'arm_wr0',
        'arm0.wr1': 'arm_wr1',
        'arm0.f1x': 'arm_f1x',
        'arm0.hr0': 'arm_hr0',
    }
    
    # markers_pos, pos_indices = find_closest_vertices(robot_meshes[0], markers_pos)
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print(f"Loaded marker positions: {len(marker_positions)}")


    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, options.data_dir)
    subdirs = natsorted(os.listdir(torque_dir))
    torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]

   
    vis = o3d.visualization.Visualizer() 
    vis.create_window()
    visualizer = SpotVisualizer(vis=vis)
    # visualizer.visualize()
    radius = 0.04
    # marker = create_red_markers([[0, 0, 0.075]], radius=radius)[0]
    # for robot_mesh in robot_meshes:
    #     vis.add_geometry(robot_mesh)
    total_mesh = o3d.geometry.TriangleMesh()

    for mesh in visualizer.o3d_meshes_default:
       total_mesh += mesh
    point_cloud = total_mesh.sample_points_uniformly(number_of_points=1000000)
    # sampled_points = sample_points_from_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), num_points)

    # o3d.visualization.draw_geometries([point_cloud])
    # vis.add_geometry(point_cloud)
    alpha = 0.1
    sliding_win = 10
    buffer = np.zeros((sliding_win, 101))
    weights = np.power((1 - alpha), np.arange(sliding_win))
    weights = alpha * weights

    original_point_colors = np.asarray(point_cloud.colors).copy()
    # original_vertex_colors = np.asarray(total_mesh.vertex_colors).copy()
   
    # Add the combined mesh to the visualizer
    # vis.add_geometry(total_mesh)
    # body_mesh = robot_meshes[0]
    # body_mesh = body_mesh.filter_smooth_taubin(number_of_iterations=5)
    # body_mesh.compute_vertex_normals()
    pcd_points = np.asarray(point_cloud.points)
    # vertices = np.asarray(total_mesh.vertices)
    # sampled_points = sample_points_from_mesh(np.asarray(robot_meshes[0].vertices), np.asarray(robot_meshes[0].triangles), 10000)
    kdtree = cKDTree(pcd_points)

    

    try:
        while True:
            state = robot_state_client.get_robot_state()

            # Preprocess the data for inference
            processed_data= preprocess_realtime_data(state, np.zeros(12), markers_path)
            print(f"Processed data shape: {processed_data.shape}")
            joint_positions = {joint.name: 0.0 for joint in visualizer.robot.joints}
            joint_states = state.kinematic_state.joint_states
            for joint_info in joint_states:
                joint = visualizer.robot.joint_map[simplified_to_full_name.get(joint_info.name)]
                if joint:
                    joint_positions[simplified_to_full_name.get(joint_info.name)] = joint_info.position.value
                else:
                    print(f"Joint {joint_info['name']} not found in URDF.")
            visualizer.visualize(cfg=joint_positions)
            for mesh in visualizer.o3d_meshes_default:
                vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            # o3d.visualization.draw_geometries(robot_meshes)
            # new_total_mesh = o3d.geometry.TriangleMesh()
            # for mesh in visualizer.o3d_meshes_default:
            #     new_total_mesh += mesh
            # vis.update_geometry(total_mesh)
            # new_points = np.asarray(total_mesh.sample_points_uniformly(number_of_points=1000000).points)
            # point_cloud.points = o3d.utility.Vector3dVector(new_points)
            # point_cloud.estimate_normals()
            # point_cloud = total_mesh.sample_points_uniformly(number_of_points=1000000)
            buffer = np.roll(buffer, 1, axis=0) 
            buffer[0] = model.predict(processed_data)
            predictions = np.dot(weights, buffer).reshape(-1, 101)
            predicted_class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class = classes[predicted_class_index]
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
            # total_mesh.vertex_colors = o3d.utility.Vector3dVector(original_vertex_colors)
            point_cloud.colors = o3d.utility.Vector3dVector(original_point_colors)
            # np.asarray(robot_meshes[0].vertex_colors)[:] = original_vertex_colors
            if predicted_class == "no_contact" :
            # or confidence < 0.1:
                pos = [0, 0, 0]
                print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

            # Visualize the prediction
            # visualize_prediction(marker_positions, predicted_class, robot_meshes)
            else:
                pos = marker_positions.get(predicted_class)

                indices = kdtree.query_ball_point(pos, radius)
                # colors = np.asarray(total_mesh.vertex_colors)
                colors = np.asarray(point_cloud.colors)
                for idx in indices:
                    if 0 <= idx < len(colors):  # Validate index range
                        colors[idx] = [1, 0, 0]
                # total_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                point_cloud.colors = o3d.utility.Vector3dVector(colors)
                # total_mesh.compute_vertex_normals()
                # R = np.eye(3)
                # T = np.eye(4)
                # T[:3, :3] = R 
                # T[:3, 3] = pos
            # marker.translate(pos - marker.get_center(), relative=False)
            # marker.translate(pos, relative=False)
            # vis.update_geometry(point_cloud)

            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        print("Exiting real-time inference...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()