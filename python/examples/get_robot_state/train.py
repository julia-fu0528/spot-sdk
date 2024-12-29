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
from tensorflow.keras.layers import Dense, Flatten, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

from visualize_robot_state import load_joint_torques, load_joint_positions, vis_joint_torques


def load_data(torque_dir, classes, classify=False, num_classes = None, seq = False):
    # in classification task, 2nd argument is class_to_index 
    # in regression task, 2nd argument is coordinates, in both cases are dicts w str keys

    training_data = []
    labels = []

    min_length = np.inf
    for i, torque_file in enumerate(natsorted(os.listdir(torque_dir))):
        if torque_file.endswith(".npy"):
            torque_path = os.path.join(torque_dir, torque_file)
            class_name = torque_file.split(".")[0]  # Extract label from the filename
            if class_name == 'no_contact':
                if not classify:
                    class_name = '100'
            if class_name in classes.keys():
                label = classes[class_name]
                torque, _, _, _ = load_joint_torques(torque_path)
                if torque.shape[0]  < min_length:
                    min_length = torque.shape[0]
                    print(f"new min length: {min_length}")
                joint_angle, _, _, _ = load_joint_positions(torque_path)
                data = np.hstack((torque, joint_angle))
                normalized_data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
                training_data.append(normalized_data)
                if classify:
                    if num_classes is None:
                        raise Exception("num_classes must be provided for classification")
                    label = to_categorical([label], num_classes=num_classes)
                labels.append(np.tile(label, (len(torque), 1)))
                
            
    # get the min length of the torque data
    training_data = [data[:min_length] for data in training_data]
    print(f"min_length: {min_length}")
    print(f"training data len: {training_data[0].shape}") # 101, 557, 24 //todo
    labels = [label[:min_length] for label in labels]
    print(f"labels len: {labels[0].shape}") # 101, 557
    training_data = np.array(training_data)
    labels = np.array(labels)
    # print(f"labels: {labels}")
    print(f"training_data.shape: {training_data.shape}")
    print(f"labels shape: {labels.shape}")
    return training_data, labels


def preprocess(torque_dir, classes, classify=False, num_classes=None):
    if classify and num_classes is None:
        raise Exception("num_classes must be provided for classification")
    # count the number of directories in torque_dir
    num_dir = len([name for name in os.listdir(torque_dir) if os.path.isdir(os.path.join(torque_dir, name))])
    print(f"num_dir: {num_dir}")
    num_dir = 9
    # randomly pick a number from 0 to num_dir
    val_indices = random.sample(range(num_dir), 4)
    train_dirs = []
    dirs = natsorted(os.listdir(torque_dir))
    dirs = dirs[:10] + dirs[-11:]
    # dirs = dirs[:5]
    for idx, dir in enumerate(dirs):
        if os.path.isdir(os.path.join(torque_dir, dir)):
            if idx in val_indices:
                val_dir = os.path.join(torque_dir, dir)
            else:
                train_dirs.append(os.path.join(torque_dir, dir))
    for train_dir in train_dirs:
        training_data, labels = load_data(train_dir, classes, classify, num_classes)
        if 'X_train' in locals():
            X_train = np.concatenate((X_train, training_data), axis=1)
            y_train = np.concatenate((y_train, labels), axis=1)
        else:
            X_train = training_data
            y_train = labels
    X_val, y_val = load_data(val_dir, classes, classify, num_classes)
    X_all, y_all = np.append(X_train, X_val, axis=1), np.append(y_train, y_val, axis=1)
    print(f"X_all.shape: {X_all.shape}, y_all.shape: {y_all.shape}")
    # len_data = X_all.shape[1]
    # train_idx = random.sample(range(len_data), int(0.8 * len_data))
    # X_train, y_train = X_all[:, train_idx, :], y_all[:, train_idx]
    # val_idx = list(set(range(len_data)) - set(train_idx))
    # X_val, y_val = X_all[:, val_idx, :], y_all[:, val_idx]

    return X_train, X_val, y_train, y_val


def preprocess_seq(torque_dir, classes):
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    # count the number of directories in torque_dir
    num_dir = len([name for name in os.listdir(torque_dir) if os.path.isdir(os.path.join(torque_dir, name))])
    print(f"num_dir: {num_dir}")
    num_dir = 20
    # randomly pick a number from 0 to num_dir
    # val_idx = random.randint(0, num_dir)
    val_indices = random.sample(range(num_dir), 2)
    train_dirs = []
    dirs = natsorted(os.listdir(torque_dir))
    dirs = dirs[:10] + dirs[-11:]
    # dirs = dirs[:5]
    for idx, dir in enumerate(dirs):
        if os.path.isdir(os.path.join(torque_dir, dir)):
            # if idx == val_idx:
            if idx in val_indices:
                val_dir = os.path.join(torque_dir, dir)
            else:
                train_dirs.append(os.path.join(torque_dir, dir))
    for train_dir in train_dirs:
        training_data, labels = load_data(train_dir, class_to_index, num_classes)
        if 'X_train' in locals():
            # min_len = min(X_train.shape[1], training_data.shape[1])
            # print(f"min_len: {min_len}")
            X_train = np.concatenate((X_train, training_data), axis=1)
            y_train = np.concatenate((y_train, labels), axis=1)
        else:
            X_train = training_data
            y_train = labels
    X_val, y_val = load_data(val_dir, class_to_index, num_classes)
    X_all, y_all = np.append(X_train, X_val, axis=1), np.append(y_train, y_val, axis=1)
    print(f"X_all.shape: {X_all.shape}, y_all.shape: {y_all.shape}")
    len_data = X_all.shape[1]
    train_idx = random.sample(range(len_data), int(0.8 * len_data))
    X_train, y_train = X_all[:, train_idx, :], y_all[:, train_idx]
    val_idx = list(set(range(len_data)) - set(train_idx))
    X_val, y_val = X_all[:, val_idx, :], y_all[:, val_idx]


    sequence_length = 3
    X_train_seq = []
    y_train_seq = []
    for i in range(len(X_train[0]) - sequence_length + 1):
       X_train_seq.append(X_train[:, i:i+sequence_length, :])
       y_train_seq.append(y_train[:, i+sequence_length-1])
    
    X_train = np.array(X_train_seq)
    y_train = np.array(y_train_seq)

    X_val_seq = []
    y_val_seq = []
    for i in range(len(X_val[0]) - sequence_length + 1):
       X_val_seq.append(X_val[:, i:i+sequence_length, :])
       y_val_seq.append(y_val[:, i+sequence_length-1])
    X_val = np.array(X_val_seq)
    y_val = np.array(y_val_seq)

    return X_train, X_val, y_train, y_val

def create_model(input_shape, classify, out_dim):
    if classify:
        model = Sequential([
            Input(shape=input_shape),      # Define input shape explicitly
            Flatten(),                     # Flatten the input
            Dense(128, activation='relu'), # Hidden layer with 128 neurons
            Dense(256, activation='relu'),  # Hidden layer with 64 neurons
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(out_dim, activation='softmax') 
        ])
    else:
        model = Sequential([
            Input(shape=input_shape),      # Define input shape explicitly
            Flatten(),                     # Flatten the input
            Dense(128, activation='relu'), # Hidden layer with 128 neurons
            Dense(256, activation='relu'),  # Hidden layer with 64 neurons
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(out_dim)  
        ])
    return model
    

def create_model_conv(input_shape, num_classes):
   model = Sequential([
       Input(shape=(3, input_shape)),  # 3 timesteps, each with input_shape features
       Conv1D(64, kernel_size=2, activation='relu'),
       Conv1D(128, kernel_size=2, activation='relu'),
       GlobalAveragePooling1D(),
       Dense(256, activation='relu'),
       Dense(num_classes, activation='softmax')
   ])
   return model


def train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir, classify, ground_truth_coords):

    # Create the model
    input_shape = X_train.shape[1:]  # Shape of a single input sample
    num_classes = len(classes)       # Number of output classes
    if classify:
        out_dim = num_classes
    else:
        out_dim = 3
    model = create_model(input_shape, classify, out_dim)
    model.summary()

    ground_truth_coords_list = [coordinates[str(i)] for i in range(len(coordinates))]
    ground_truth_coords_tensor = tf.convert_to_tensor(ground_truth_coords_list, dtype=tf.float32)

    def named_euclidean_distance(y_true, y_pred):
        return euclidean_distance(y_true, y_pred, classify, ground_truth_coords_tensor)
    named_euclidean_distance.__name__ = 'euclidean_distance'
    
    if classify:

        # Compile the model 
        model.compile(
            # optimizer=tf.keras.optimizers.Adam(learning_rate=5e-3),# 5e-4
            optimizer='adam',
            loss='categorical_crossentropy',  # Cross-entropy loss for classification
            metrics=['accuracy', named_euclidean_distance]
        )

    else: # regression
        def named_regression_accuracy(y_true, y_pred):
            return regression_accuracy(y_true, y_pred, ground_truth_coords_tensor)
        named_regression_accuracy.__name__ = 'regression_accuracy'

        # Compile the model 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),# 5e-4
            # optimizer='adam',
            loss='mse',  # Cross-entropy loss for classification
            metrics=[named_regression_accuracy, named_euclidean_distance]
            # metrics=[named_regression_accuracy, 'mae', euclidean_distance]
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
        epochs=40,                     # Number of epochs
        batch_size=128,                 # Batch size
        callbacks=[
            tensorboard_callback,
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                best_model_path, 
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    print(history.history['euclidean_distance'])
    print(history.history['val_euclidean_distance'])
    # Save the trained model
    model.save(model_path)

    return model, history


def euclidean_distance(y_true, y_pred, classify, ground_truth_coords = None):
    if classify:
        true_coord_index = tf.argmax(y_true, axis=-1)  # Index of the ground truth
        pred_coord_index = tf.argmax(y_pred, axis=-1)  # Index of the predicted class

        # Get the corresponding coordinates
        true_coord = tf.gather(ground_truth_coords, true_coord_index)  # Shape: [batch_size, 3]
        pred_coord = tf.gather(ground_truth_coords, pred_coord_index)

        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(true_coord - pred_coord), axis=-1)))
    else: # regression
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))

# def regression_accuracy(y_true, y_pred, ground_truth_coords):
#     print(f"y_true type: {type(y_true)}")
#     print(f"y_pred type: {type(y_pred)}")
#     y_true = tf.reshape(y_true, [-1, 3])
#     y_pred = tf.reshape(y_pred, [-1, 3])
#     y_pred_expanded = tf.expand_dims(y_pred, axis=1)  # Shape: [batch_size, 1, 3]
#     y_true_expanded = tf.expand_dims(y_true, axis=1)  # Shape: [batch_size, 1, 3]

#     distances = tf.sqrt(tf.reduce_sum(tf.square(y_pred_expanded - y_true_expanded), axis=-1))  # Shape: [batch_size, num_classes]

#     nearest_indices = tf.argmin(distances, axis=1)  # Shape: [batch_size]

#     nearest_coords = tf.gather(ground_truth_coords, nearest_indices)  # Shape: [batch_size, 3]

#     correct_predictions = tf.reduce_all(tf.equal(nearest_coords, y_true), axis=-1)  # Shape: [batch_size]

#     return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def regression_accuracy(y_true, y_pred, ground_truth_coords, tolerance=1e-3):

    # Ensure ground_truth_coords is a tensor
    ground_truth_coords = tf.convert_to_tensor(ground_truth_coords, dtype=tf.float32)
    # Reshape y_true and y_pred
    y_true = tf.reshape(y_true, [-1, 3])
    y_pred = tf.reshape(y_pred, [-1, 3])

    # Expand dimensions for broadcasting
    y_pred_expanded = tf.expand_dims(y_pred, axis=1)  # Shape: [1, 3]
    y_true_expanded = tf.expand_dims(y_true, axis=1)  # Shape: [1, 3]
    ground_truth_coords_expanded = tf.expand_dims(ground_truth_coords, axis=0)  # Shape: [1, num_coords, 3]

    print(f"y_pred_expanded.shape: {y_pred_expanded.shape}")
    print(f"y_true_expanded.shape: {y_true_expanded.shape}")
    print(f"ground_truth_coords_expanded.shape: {ground_truth_coords_expanded.shape}")

    # Compute pairwise Euclidean distances
    distances = tf.sqrt(tf.reduce_sum(tf.square(y_pred_expanded - ground_truth_coords_expanded), axis=-1))  # Shape: [batch_size, num_coords]

    # Find the index of the nearest ground truth coordinate
    nearest_indices = tf.argmin(distances, axis=1)  # Shape: [batch_size]

    # Get the nearest ground truth coordinate
    nearest_coords = tf.gather(ground_truth_coords, nearest_indices)  # Shape: [batch_size, 3]

    # Compute distance between nearest_coords and y_true
    # errors = tf.sqrt(tf.reduce_sum(tf.square(nearest_coords - y_true), axis=-1))  # Shape: [batch_size]

    # # Check if the error is within the tolerance
    # correct_predictions = errors <= tolerance  # Shape: [batch_size]
    correct_predictions = tf.reduce_all(tf.equal(nearest_coords, y_true), axis=-1)  # Shape: [batch_size]

    # Calculate accuracy as the mean of correct predictions
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def predict(model_path, classes):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Predict on new data
    new_sample = np.load("data/20241120/front.npy")  # Example file
    new_sample = np.expand_dims(new_sample, axis=0)  # Add batch dimension
    predictions = model.predict(new_sample)

    # Decode the predictions
    # predicted_class = classes[np.argmax(predictions)]
    predicted_coordinates = predictions[0]


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
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')


    options = parser.parse_args()

    classify = options.classify

    session = options.session
    data_dir = options.data_dir
    markers_path = options.markers_path

    if classify:
        model_dir = os.path.join("classify", options.model_dir)
        log_dir = os.path.join("classify", options.log_dir)
        plots_dir = os.path.join("classify", options.plots_dir)
    else:
        model_dir = os.path.join("regression", options.model_dir)
        log_dir = os.path.join("regression", options.log_dir)
        plots_dir = os.path.join("regression", options.plots_dir)

 
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
    num_classes = len(classes)
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


    print(f"torque_dir: {torque_dir}")
    if classify:
        class_to_label = {cls: idx for idx, cls in enumerate(classes)}
    else:
        class_to_label = coordinates
    
    X_train, X_val, y_train, y_val = preprocess(torque_dir, class_to_label, classify, num_classes)
    print(f"y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}")

    # Add no contact
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    extra_dirs = [name for name in natsorted(os.listdir(torque_dir)) if os.path.isdir(os.path.join(torque_dir, name))][10:20]
    val_indices = random.sample(range(10, 20), 2)
    train_dirs = []
    val_dirs = []
    for idx, dir in enumerate(natsorted(os.listdir(torque_dir))[10:20]):
        if os.path.isdir(os.path.join(torque_dir, dir)):
            if idx+10 in val_indices:
                val_dirs.append(os.path.join(torque_dir, dir))
            else:
                train_dirs.append(os.path.join(torque_dir, dir))
    X_train = [sequence for sequence in X_train] 
    y_train = [sequence for sequence in y_train]
    X_val = [sequence for sequence in X_val]
    y_val = [sequence for sequence in y_val]
    for train_dir in train_dirs:
        if classify:
            training_data, labels = load_data(train_dir, {'no_contact': 100}, classify, num_classes)
        else:
            training_data, labels = load_data(train_dir, {'100': np.array([0, 0, 0])}, classify)
        X_train[-1] = np.concatenate((X_train[-1], training_data[0]), axis=0)
        y_train[-1] = np.concatenate((y_train[-1], labels[0]), axis=0)
    for val_dir in val_dirs:
        if classify:   
            val_data, val_labels = load_data(val_dir, {'no_contact': 100}, classify, num_classes)
        else:
            val_data, val_labels = load_data(val_dir, {'100': np.array([0, 0, 0])}, classify)
        X_val[-1] = np.concatenate((X_val[-1], val_data[0]), axis=0)
        y_val[-1] = np.concatenate((y_val[-1], val_labels[0]), axis=0)


    # One-hot encode labels
    classes = [f.split('.')[0] for f in torque_files]
    X_train_nc = X_train[-1]
    # y_train_nc = to_categorical(y_train[-1], num_classes = len(classes))
    y_train_nc = y_train[-1]
    X_val_nc = X_val[-1]
    # y_val_nc = to_categorical(y_val[-1], num_classes = len(classes))
    y_val_nc = y_val[-1]
    X_train = np.array(X_train[:-1])
    y_train = np.array(y_train[:-1])
    X_val = np.array(X_val[:-1])
    y_val = np.array(y_val[:-1])
    
    # classification
    # y_train = to_categorical(y_train, num_classes=len(classes))
    # y_val = to_categorical(y_val, num_classes=len(classes))

    # regression
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    print(f"y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}")
    
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

    model, history = train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir, classify, coordinates)
    
    # Evaluate the model on validation set classification
    # val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    # regression
    val_loss, val_accuracy,val_euclidean = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(f"Validation Euclidean Distance: {val_euclidean:.4f}")

    confusion_matrix_path = os.path.join(plots_dir, f'confusion_matrix.png')
    plot_confusion_matrix(model, X_val, y_val, classes, save_path=confusion_matrix_path)