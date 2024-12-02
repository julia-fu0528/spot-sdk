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

from visualize_robot_state import load_joint_torques, vis_joint_torques


def load_data(torque_dir, class_to_index):
    torque_data = []
    labels = []

    no_contact_torque, _, _, _ = load_joint_torques(os.path.join(torque_dir, "no_contact.npy"))
    offset = np.mean(no_contact_torque, axis=0)  # Use no-contact torque as offset
    
    for torque_file in natsorted(os.listdir(torque_dir)):
        if torque_file.endswith(".npy"):
            torque_path = os.path.join(torque_dir, torque_file)
            class_name = torque_file.split(".")[0]  # Extract label from the filename
            if class_name in classes:
                torque, _, _, _ = load_joint_torques(torque_path)
                torque = torque - offset  # Subtract offset
                torque_data.append(torque)
                # labels.append(class_name * len(torque))
                labels.append([class_to_index[class_name]] * len(torque))  # Map class name to index

    # get the min length of the torque data
    min_length = min([len(torque) for torque in torque_data])
    torque_data = [torque[:min_length] for torque in torque_data]
    labels = [label[:min_length] for label in labels]
    # torque_data = pad_sequences(torque_data, padding='post', dtype='float32')
    torque_data = np.array(torque_data)
    labels = np.array(labels)
    # print(f"labels: {labels}")

    return torque_data, labels

def preprocess(torque_dir, classes):
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    torque_data, labels = load_data(torque_dir, class_to_index)
    print(f"labels: {labels}")

    # One-hot encode labels
    labels_one_hot = to_categorical(labels, num_classes=len(classes))

    # Split the second dimension (time steps)
    time_steps = torque_data.shape[1]
    split_index = int(0.8 * time_steps)

    train_idx = random.sample(range(time_steps), split_index)
    val_idx = list(set(range(time_steps)) - set(train_idx))
    
    # Split and reshape data
    X_train = torque_data[:, train_idx, :]
    X_val = torque_data[:, val_idx, :]
    
    # Reshape X data
    X_train = X_train.reshape(-1, X_train.shape[-1])  # Combine batch and time dimensions
    X_val = X_val.reshape(-1, X_val.shape[-1])
    
    # Reshape y data to match X
    y_train = labels_one_hot[:, train_idx, :].reshape(-1, labels_one_hot.shape[-1])
    y_val = labels_one_hot[:, val_idx, :].reshape(-1, labels_one_hot.shape[-1])

    return X_train, X_val, y_train, y_val


def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),      # Define input shape explicitly
        Flatten(),                     # Flatten the input
        Dense(128, activation='relu'), # Hidden layer with 128 neurons
        Dense(64, activation='relu'),  # Hidden layer with 64 neurons
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model
    

def train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir):
    # Create the model
    input_shape = X_train.shape[1:]  # Shape of a single input sample
    num_classes = len(classes)       # Number of output classes
    model = create_model(input_shape, num_classes)
    model.summary()

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Cross-entropy loss for classification
        metrics=['accuracy']
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
        epochs=50,                     # Number of epochs
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
                         alpha=0.6)
    
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
    parser.add_argument('--plots_dir', required=True, help='Output directory for plots')

    options = parser.parse_args()

    session = options.session
    print(f"session: {session}")
    data_dir = options.data_dir
    model_dir = options.model_dir
    log_dir = options.log_dir
    plots_dir = options.plots_dir

    # Path to the directory containing `.npy` files
    # current folder path 
    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, data_dir, session)
    # get all the files with .npy extension
    # torque_dir = natsorted(os.listdir(torque_dir))
    torque_files = [f for f in os.listdir(torque_dir) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]
    print(f"torque_classes: {classes}")
    model_dir = os.path.join(folder_path, model_dir, session)
    log_dir = os.path.join(folder_path, log_dir, session)
    plots_dir = os.path.join(folder_path, plots_dir, session)  # New directory for plots


    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)


    model_path = os.path.join(model_dir, "5_class_model.keras")
    best_model_path = os.path.join(model_dir, 'best_model.keras')


    # classes = ['no_contact', 'front', 'back', 'left', 'right']  # Define your classes
    X_train, X_val, y_train, y_val = preprocess(torque_dir, classes)

     # Create t-SNE visualization before training
    print("Creating t-SNE visualization of training data...")
    tsne_path = os.path.join(plots_dir, 'tsne_visualization.png')
    plot_tsne(X_train, y_train, classes, save_path=tsne_path)

    model, history = train(X_train, X_val, y_train, y_val, model_path, best_model_path, log_dir)
    
    # Evaluate the model on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    # predict(model_path, classes)

    confusion_matrix_path = os.path.join(plots_dir, f'confusion_matrix.png')
    plot_confusion_matrix(model, X_val, y_val, classes, save_path=confusion_matrix_path)