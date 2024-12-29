import os
import random
import numpy as np
import torch
from pathlib import Path
from natsort import natsorted
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from visualize_robot_state import load_joint_torques, load_joint_positions


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
    print(f"training data len: {training_data[0].shape}") # 101, 557, 24
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
    num_dir = 20
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
    len_data = X_all.shape[1]
    train_idx = random.sample(range(len_data), int(0.8 * len_data))
    X_train, y_train = X_all[:, train_idx, :], y_all[:, train_idx]
    val_idx = list(set(range(len_data)) - set(train_idx))
    X_val, y_val = X_all[:, val_idx, :], y_all[:, val_idx]

    return X_train, X_val, y_train, y_val


# Dataset
class TorqueDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# DataModule
class TorqueDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=128):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TorqueDataset(self.X_train, self.y_train)
        self.val_dataset = TorqueDataset(self.X_val, self.y_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


# Model
class TorqueModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, classify=False, learning_rate=1e-3):
        super().__init__()
        self.classify = classify
        self.learning_rate = learning_rate
        if classify:
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
                nn.Softmax(dim=1)
            )
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
            self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)
        return {"val_loss": loss, "preds": preds, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Confusion Matrix Plot
def plot_confusion_matrix(preds, targets, classes, save_path=None):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = torch.argmax(targets, dim=1).cpu().numpy()
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Main Training Function
def main():
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
    
    input_dim = X_train.shape[1]

    # Create DataModule
    data_module = TorqueDataModule(X_train, y_train, X_val, y_val, batch_size=128)

    # Create Model
    if classify:
        out_dim = num_classes
    else:
        out_dim = 3
    model = TorqueModel(input_dim=input_dim, output_dim=out_dim, classify=classify)

    # Logger and Callbacks
    logger = TensorBoardLogger("tb_logs", name="torque_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best_model")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test or Validate the model
    trainer.validate(model, data_module)


if __name__ == "__main__":
    main()
