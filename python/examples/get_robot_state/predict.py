import os
import time
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from natsort import natsorted
from urdfpy import URDF
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera
from visualize_robot_state import create_red_markers, compute_forward_kinematics, load_joint_torques, prepare_trimesh_fk, convert_trimesh_to_open3d

def collect_realtime_data(robot_state_client, duration=2):
    """
    Collect real-time data from the robot for a specified duration.
    """
    start_time = time.time()
    data = []

    print("Starting real-time data collection...")
    while time.time() - start_time < duration:
        state = robot_state_client.get_robot_state()
        data.append(state)
        # time.sleep(0.1)  # Sampling rate (adjust as needed)

    print("Real-time data collection complete.")
    return np.array(data)

def preprocess_realtime_data(data, offset, markers_path):
    """
    Preprocess the real-time data for inference.
    """
    markers_pos = np.loadtxt(markers_path, delimiter=",")
    print(f"Loaded marker positions: {len(markers_pos)}")
    
    # Assuming data contains joint torque or similar state information
    # Preprocess data (dummy example here, adapt as needed)
    state_dict = {}
    torque_dict = {}
    for i in range(len(data)):
        state_dict[i] = data[i].kinematic_state.joint_states
        torque_dict[i] = []
        for joint in state_dict[i]:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                if not joint_name.startswith("arm"):
                    torque_dict[i].append({
                        'name': joint_name,
                        'load': joint.load.value  # Assuming load has a 'value' attribute
                    })
    num_entries = len(torque_dict)
    num_joints = max(len(torque_dict[i]) for i in torque_dict)
    torque_data = np.full((num_entries, num_joints), np.nan, dtype=float)
    joint_names = []
    for i in range(num_entries):
        for j, joint in enumerate(torque_dict[i]):
            torque_data[i, j] = joint['load']
            if i == 0:
                joint_names.append(joint['name'])
    # data_processed = np.array([d.joint_state.load_torque for d in data])
    # data_processed = np.expand_dims(data_processed, axis=0)  # Add batch dimension
    min_length = min([len(torque) for torque in torque_data])
    torque_data = [torque[:min_length] for torque in torque_data]
    torque_data = torque_data - offset
    data_processed = np.array(torque_data)

    return data_processed


# def infer_realtime(model, data, classes):
#     """
#     Perform real-time inference using the trained model.
#     """
#     predictions = model.predict(data)
#     predicted_class = classes[np.argmax(np.mean(predictions, axis=0))]
#     confidence = np.max(predictions)

#     return predicted_class, confidence
def infer_realtime(model, data, classes):
    """
    Perform real-time inference using the trained model.
    """
    print("Running real-time inference...")
    
    # Get predictions (shape: [num_samples, num_classes])
    predictions = model.predict(data)
    print(f"Predictions shape: {predictions.shape}")
    
    if len(predictions.shape) == 2:  # Multiple samples
        # mean_probabilities = np.mean(predictions, axis=0)
        # predicted_class_index = np.argmax(mean_probabilities)
        # confidence = np.max(mean_probabilities)
        predicted_classes = [np.argmax(pred) for pred in predictions]
        print(f"Predicted classes: {predicted_classes}")
        
        # Find the most common class
        most_common_class_index, count = Counter(predicted_classes).most_common(1)[0]
        predicted_class = classes[most_common_class_index]
        print(f"predicted class: {predicted_class}")
        
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
    marker = create_red_markers([pos], radius=0.02)[0]
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
    # parser.add_argument('--classes', nargs='+', required=True, help='List of class names')
    options = parser.parse_args()

     # Load the trained model
    print("Loading the model...")
    model = load_model(options.model_path)
    print("Model loaded successfully.")

     # Load marker positions
    markers_path = options.markers_path
    markers_pos = np.loadtxt(markers_path, delimiter=",")
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print(f"Loaded marker positions: {len(marker_positions)}")

    # Initialize robot and client
    sdk = create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    # Load robot meshes for visualization
    robot_obj = URDF.load('spot_description/spot.urdf')
    joint_positions = {joint.name: 0.0 for joint in robot_obj.joints}  # Zero configuration
    link_fk_transforms = compute_forward_kinematics(robot_obj, joint_positions)
    trimesh_fk = prepare_trimesh_fk(robot_obj, link_fk_transforms)
    robot_meshes = convert_trimesh_to_open3d(trimesh_fk)

    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, options.data_dir)
    # get all the files with .npy extension
    torque_files = [f for f in os.listdir(torque_dir) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]

    # no_contact_torque, _, _, _ = load_joint_torques(os.path.join(torque_dir, "no_contact.npy"))
    print(f"Collecting no-contact torque data for offset calculation...")
    no_contact_torque = collect_realtime_data(robot_state_client, 10)
    no_contact_torque = preprocess_realtime_data(no_contact_torque, np.zeros(12), markers_path)
    offset = np.mean(no_contact_torque, axis=0)  # Use no-contact torque as offset


    try:
        while True:
            # Collect real-time data
            print(f"YOU CAN TOUCH THE SPOT NOW. Data collection will start in 5 seconds, please make sure you are touching the Spot.\n")
            time.sleep(5)
            data = collect_realtime_data(robot_state_client, 5)

            # Preprocess the data for inference
            processed_data = preprocess_realtime_data(data, offset, markers_path)

            # Perform inference
            predicted_class, confidence = infer_realtime(model, processed_data, classes)
            print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
            if predicted_class == "no_contact":
                print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
                continue
            # Visualize the prediction
            visualize_prediction(marker_positions, predicted_class, robot_meshes)

            # Add user prompt to continue or exit
            user_input = input("Press 'Enter' to collect data again or 'q' to quit: ").strip().lower()
            if user_input == 'q':
                break
    except KeyboardInterrupt:
        print("Exiting real-time inference...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()