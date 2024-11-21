import os
import random
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda


from visualize_robot_state import load_joint_torques, vis_joint_torques


def load_data(torque_dir, class_to_index):
    torque_data = []
    labels = []
    
    for torque_file in os.listdir(torque_dir):
        if torque_file.endswith(".npy"):
            torque_path = os.path.join(torque_dir, torque_file)
            class_name = torque_file.split(".")[0]  # Extract label from the filename
            if class_name in classes:
                torque, _, _, _ = load_joint_torques(torque_path)
                torque_data.append(torque)
                labels.append([class_to_index[class_name]] * len(torque))  # Map class name to index


    # get the min length of the torque data
    min_length = min([len(torque) for torque in torque_data])
    torque_data = [torque[:min_length] for torque in torque_data]
    labels = [label[:min_length] for label in labels]
    # torque_data = pad_sequences(torque_data, padding='post', dtype='float32')
    torque_data = np.array(torque_data)
    labels = np.array(labels)

    return torque_data, labels

def preprocess(torque_dir, classes):
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    torque_data, labels = load_data(torque_dir, class_to_index)
    print(f"torque_data.shape: {torque_data.shape}")
    print(f"labels.shape: {labels.shape}")

    # One-hot encode labels
    labels_one_hot = to_categorical(labels, num_classes=len(classes))
    print(f"labels_one_hot.shape: {labels_one_hot.shape}")

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
    

def train(X_train, X_val, y_train, y_val, model_path, best_model_path):
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
    
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,                     # Number of epochs
        batch_size=32,                 # Batch size
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True)
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


if __name__ == "__main__":
    # Path to the directory containing `.npy` files
    # current folder path 
    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, "data/20241120")
    model_dir = os.path.join(folder_path, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "5_class_model.keras")
    best_model_path = os.path.join(model_dir, 'best_model.keras')


    classes = ['no_contact', 'front', 'back', 'left', 'right']  # Define your classes
    X_train, X_val, y_train, y_val = preprocess(torque_dir, classes)

    model, history = train(X_train, X_val, y_train, y_val, model_path, best_model_path)
    
    # Evaluate the model on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    # predict(model_path, classes)