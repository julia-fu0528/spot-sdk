from datetime import datetime
import os
import random
import sys
from natsort import natsorted
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

from visualize_robot_state import load_joint_torques, load_joint_positions, vis_joint_torques


def load_data(torque_dir, class_to_index):
    # torque_data = []
    training_data = []
    labels = []

    # no_contact_torque, _, _, _ = load_joint_torques(os.path.join(torque_dir, "no_contact.npy"))
    # no_contact_torque = 2 * ((no_contact_torque - no_contact_torque.min()) / (no_contact_torque.max() - no_contact_torque.min())) - 1
    # offset = np.mean(no_contact_torque, axis=0)  # Use no-contact torque as offsett
    min_length = np.inf
    print(f"torque_dir: {torque_dir}")
    for i, torque_file in enumerate(natsorted(os.listdir(torque_dir))):
        if torque_file.endswith(".npy"):
            torque_path = os.path.join(torque_dir, torque_file)
            class_name = torque_file.split(".")[0]  # Extract label from the filename
            if class_name in classes:
                torque, _, _, _ = load_joint_torques(torque_path)
                if torque.shape[0]  < min_length:
                    min_length = torque.shape[0]
                    print(f"new min length: {min_length}")
                joint_angle, _, _, _ = load_joint_positions(torque_path)
                # torque = torque - offset  # Subtract offset
                # normalized_torque = 2 * ((torque - torque.min()) / (torque.max() - torque.min())) - 1
                # data = np.hstack((normalized_torque, joint_angle))
                data = np.hstack((torque, joint_angle))
                # print(f"torque file: {torque_file}, data.shape: {data.shape}")
                normalized_data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
                # torque_data.append(normalized_torque)
                training_data.append(normalized_data)
                # print(f"training_data.shape: {training_data[-1].shape}")
                # torque_data.append(torque) 
                # labels.append(class_name * len(torque))
                labels.append([class_to_index[class_name]] * len(torque))  # Map class name to index
            
    # get the min length of the torque data
    # min_length = min([len(torque) for torque in torque_data]),
    # torque_data = [torque[:min_length] for torque in torque_data]
    training_data = [data[:min_length] for data in training_data]
    labels = [label[:min_length] for label in labels]
    # torque_data = pad_sequences(torque_data, padding='post', dtype='float32')
    # torque_data = np.array(torque_data)
    training_data = np.array(training_data)
    labels = np.array(labels)
    # print(f"labels: {labels}")
    print(f"training_data.shape: {training_data.shape}")
    # sys.exit()
    return training_data, labels

def preprocess(torque_dir, classes):
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    # count the number of directories in torque_dir
    num_dir = len([name for name in os.listdir(torque_dir) if os.path.isdir(os.path.join(torque_dir, name))])
    print(f"num_dir: {num_dir}")
    num_dir = 19
    # randomly pick a number from 0 to num_dir
    # val_idx = random.randint(0, num_dir)
    val_indices = random.sample(range(num_dir), 2)
    train_dirs = []
    dirs = natsorted(os.listdir(torque_dir))
    dirs = dirs[:10] + dirs[-10:]
    for idx, dir in enumerate(dirs):
        if os.path.isdir(os.path.join(torque_dir, dir)):
            # if idx == val_idx:
            if idx in val_indices:
                val_dir = os.path.join(torque_dir, dir)
            else:
                train_dirs.append(os.path.join(torque_dir, dir))
    for train_dir in train_dirs:
        training_data, labels = load_data(train_dir, class_to_index)
        if 'X_train' in locals():
            # min_len = min(X_train.shape[1], training_data.shape[1])
            # print(f"min_len: {min_len}")
            X_train = np.concatenate((X_train, training_data), axis=1)
            y_train = np.concatenate((y_train, labels), axis=1)
        else:
            X_train = training_data
            y_train = labels
    X_val, y_val = load_data(val_dir, class_to_index)
    X_all, y_all = np.append(X_train, X_val, axis=1), np.append(y_train, y_val, axis=1)
    print(f"X_all.shape: {X_all.shape}, y_all.shape: {y_all.shape}")
    len_data = X_all.shape[1]
    train_idx = random.sample(range(len_data), int(0.8 * len_data))
    X_train, y_train = X_all[:, train_idx, :], y_all[:, train_idx]
    val_idx = list(set(range(len_data)) - set(train_idx))
    X_val, y_val = X_all[:, val_idx, :], y_all[:, val_idx]
    # sys.exit()
    # torque_data, labels = load_data(torque_dir, class_to_index)

    # Split the second dimension (time steps)
    # time_steps = torque_data.shape[1]
    # split_index = int(0.8 * time_steps)

    # train_idx = random.sample(range(time_steps), split_index)
    # val_idx = list(set(range(time_steps)) - set(train_idx))
    
    # Split and reshape data
    # X_train = torque_data[:, train_idx, :]
    # X_val = torque_data[:, val_idx, :]
    
    # Reshape y data to match X
    # y_train = labels_one_hot[:, train_idx, :].reshape(-1, labels_one_hot.shape[-1])
    # y_val = labels_one_hot[:, val_idx, :].reshape(-1, labels_one_hot.shape[-1])
    return X_train, X_val, y_train, y_val


def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),      # Define input shape explicitly
        Flatten(),                     # Flatten the input
        Dense(64, activation='relu'), # Hidden layer with 128 neurons
        Dense(128, activation='relu'),  # Hidden layer with 64 neurons
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model
    

def train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir):
    # Create the model
    input_shape = X_train.shape[1:]  # Shape of a single input sample
    num_classes = len(classes)       # Number of output classes
    model = create_model(input_shape, num_classes)
    model.summary()



    ground_truth_coords_list = [coordinates[str(i)] for i in range(len(coordinates))]
    ground_truth_coords_tensor = tf.convert_to_tensor(ground_truth_coords_list, dtype=tf.float32)

    
    def named_euclidean_distance(y_true, y_pred):
        return euclidean_distance(y_true, y_pred, ground_truth_coords_tensor)
    named_euclidean_distance.__name__ = 'euclidean_distance'
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),# 5e-4
        # optimizer='adam',
        loss='categorical_crossentropy',  # Cross-entropy loss for classification
        metrics=['accuracy', named_euclidean_distance]
    )
    

    # Create TensorBoard callback with timestamped log directory
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1,  # Generate histogram visualizations for layer weights every epoch
        write_graph=True,  # Visualize the model graph
        write_images=True,  # Visualize layer activations
        update_freq='epoch',  # Update logs at the end of each epoch
        profile_batch='500,520'  # Profile performance for batches 500 to 520
    )

    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=70,                     # Number of epochs
        batch_size=32,                 # Batch size
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path, 
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    # Save the trained model
    model.save(model_path)

    return model, history

def predict(model_path, classes):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict on new data
    new_sample = np.load("data/20241120/front.npy")  # Example file
    new_sample = np.expand_dims(new_sample, axis=0)  # Add batch dimension
    predictions = model.predict(new_sample)

    # Decode the predictions
    predicted_class = classes[np.argmax(predictions)]

def euclidean_distance(y_true, y_pred, ground_truth_coords):
    true_coord_index = tf.argmax(y_true, axis=-1)  # Index of the ground truth
    pred_coord_index = tf.argmax(y_pred, axis=-1)  # Index of the predicted class

    # Get the corresponding coordinates
    true_coord = tf.gather(ground_truth_coords, true_coord_index)  # Shape: [batch_size, 3]
    pred_coord = tf.gather(ground_truth_coords, pred_coord_index)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(true_coord - pred_coord), axis=-1)))




def plot_confusion_matrix(model, X_val, y_val, classes, save_path=None):
    """
    Plot and optionally save confusion matrix
    """
    # Get predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Create figure and plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    # Show the plot
    plt.show()

def plot_tsne(X, y, classes, save_path=None):
    """
    Create and save t-SNE visualization of the data
    
    Args:
        X: Input features
        y: One-hot encoded labels
        classes: List of class names
        save_path: Optional path to save the plot
    """
    # Convert one-hot encoded labels back to class indices
    y_classes = np.argmax(y, axis=1)
    
    # Perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=y_classes, 
                         cmap='tab10',
                         alpha=1)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i/10), 
                                 label=classes[i], markersize=10)
                      for i in range(len(classes))]
    plt.legend(handles=legend_elements, title="Classes")
    
    # Add labels and title
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Data')
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type = str, required=True, help='Session for data collection')
    parser.add_argument('--data_dir', required=True, help='Output directory for data')
    parser.add_argument('--model_dir', required=True, help='Output directory for model')
    parser.add_argument('--log_dir', required=True, help='Output directory for logs')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--plots_dir', required=True, help='Output directory for plots')

    options = parser.parse_args()

    session = options.session
    data_dir = options.data_dir
    model_dir = options.model_dir
    log_dir = options.log_dir
    plots_dir = options.plots_dir
    markers_path = options.markers_path

    markers_pos = np.loadtxt(markers_path, delimiter=",")
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}

    # Path to the directory containing `.npy` files
    # current folder path 
    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, data_dir, session)
    # get all the files with .npy extension
    subdirs = natsorted(os.listdir(torque_dir))
    torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]
    # classes = ['no_contact', '95', '24', '8']
    print(f"class: {classes}")
    coordinates = {}
    print(f"class: {classes}")
    for c in classes:
        if marker_positions.get(c) is None:
            coordinates['100'] = np.array([0, 0, 0])
            # coordinates.append(np.array([0, 0, 0]))
        else:
            # coordinates.append(marker_positions.get(c))
            coordinates[c] = marker_positions.get(c)
    model_dir = os.path.join(folder_path, model_dir, session)
    log_dir = os.path.join(folder_path, log_dir, session)
    plots_dir = os.path.join(folder_path, plots_dir, session)  # New directory for plots


    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)


    model_path = os.path.join(model_dir, "5_class_model.keras")
    best_model_path = os.path.join(model_dir, 'best_model.keras')


    # classes = ['no_contact', 'front', 'back', 'left', 'right']  # Define your classes
    print(f"torque_dir: {torque_dir}")
    X_train, X_val, y_train, y_val = preprocess(torque_dir, classes)
    # add no contact
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    extra_dirs = [name for name in natsorted(os.listdir(torque_dir)) if os.path.isdir(os.path.join(torque_dir, name))][10:20]
    val_indices = random.sample(range(10, 20), 2)
    train_dirs = []
    val_dirs = []
    for idx, dir in enumerate(natsorted(os.listdir(torque_dir))[10:20]):
        if os.path.isdir(os.path.join(torque_dir, dir)):
            # if idx == val_idx:
            if idx+10 in val_indices:
                val_dirs.append(os.path.join(torque_dir, dir))
            else:
                train_dirs.append(os.path.join(torque_dir, dir))
    X_train = [sequence for sequence in X_train] 
    y_train = [sequence for sequence in y_train]
    X_val = [sequence for sequence in X_val]
    y_val = [sequence for sequence in y_val]
    for train_dir in train_dirs:
        classes = ['no_contact']
        training_data, labels = load_data(train_dir, class_to_index)
        X_train[-1] = np.concatenate((X_train[-1], training_data[0]), axis=0)
        y_train[-1] = np.concatenate((y_train[-1], labels[0]), axis=0)
    for val_dir in val_dirs:
        val_data, val_labels = load_data(val_dir, class_to_index)
        X_val[-1] = np.concatenate((X_val[-1], val_data[0]), axis=0)
        y_val[-1] = np.concatenate((y_val[-1], val_labels[0]), axis=0)


    # One-hot encode labels
    classes = [f.split('.')[0] for f in torque_files]
    X_train_nc = X_train[-1]
    y_train_nc = to_categorical(y_train[-1], num_classes = len(classes))
    X_val_nc = X_val[-1]
    y_val_nc = to_categorical(y_val[-1], num_classes = len(classes))
    X_train = np.array(X_train[:-1])
    y_train = np.array(y_train[:-1])
    X_val = np.array(X_val[:-1])
    y_val = np.array(y_val[:-1])
    
    y_train = to_categorical(y_train, num_classes=len(classes))
    y_val = to_categorical(y_val, num_classes=len(classes))
    
    # Reshape X data
    X_train = X_train.reshape(-1, X_train.shape[-1])  # Combine batch and time dimensions
    X_val = X_val.reshape(-1, X_val.shape[-1])
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    print(f"X_train_nc.shape: {X_train_nc.shape}, y_train_nc.shape: {y_train_nc.shape}")

    X_train = np.concatenate((X_train, X_train_nc), axis=0)
    X_val = np.concatenate((X_val, X_val_nc), axis=0)
    y_train = y_train.reshape(-1, y_train.shape[-1])
    y_train = np.concatenate((y_train, y_train_nc), axis=0)
    y_val = y_val.reshape(-1, y_val.shape[-1])
    y_val = np.concatenate((y_val, y_val_nc), axis=0)
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    
    
     # Create t-SNE visualization before training
    # print("Creating t-SNE visualization of training data...")
    # tsne_path = os.path.join(plots_dir, 'tsne_visualization.png')
    # plot_tsne(X_train, y_train, classes, save_path=tsne_path)

    model, history = train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir)
    
    # Evaluate the model on validation set
    val_loss, val_accuracy, val_euclidean = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Validation Euclidean distance: {val_euclidean:.2f}")

    confusion_matrix_path = os.path.join(plots_dir, f'confusion_matrix.png')
    plot_confusion_matrix(model, X_val, y_val, classes, save_path=confusion_matrix_path)