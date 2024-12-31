from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import numpy as np
import lightning as L
import sklearn.metrics as metrics
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

class JointNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, classify) -> None:
        super().__init__()
        if classify:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                # nn.Linear(256, 512),
                # nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Softmax(dim=1)
            )
        else:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
    
    def forward(self, x):
        return self.network(x)

class LitSpot(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, markers_path, classify) -> None:
        super().__init__()
        self.model = JointNetwork(input_dim, output_dim, classify)
        # self.device = device  

        self.classify = classify
        self.learning_rate = 2e-3

        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        self.marker_positions['100'] = np.array([0, 0, 0])

        self.rev_marker_positions = {
            tuple(np.round(v.astype(np.float32), decimals=4)): k for k, v in self.marker_positions.items()}

        self.marker_posarray = np.array(list(self.marker_positions.values()))
    
    def forward(self, x):
        return self.model(x)

    def _get_device(self):
        return next(self.parameters()).device

    def training_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)
        y_hat = self(x) # shape batch_size by 3

        threshold = 0.1
        if self.classify:
            # if self.device == "gpu":
            #     y_hat_idx = torch.argmax(y_hat, dim=1).cpu()
            #     y_idx = torch.argmax(y, dim=1).cpu()
            #     loss = F.cross_entropy(y_hat.cpu(), y_idx)
            # else:
            y_hat_idx = torch.argmax(y_hat, dim=1)
            y_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(y_hat, y_idx)

            y_hat_label = y_hat_idx.float()         # Convert to float if needed
            y_label = y_idx.float()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx]) 
            
            # acc = torch.sum(y_hat_label == y_label).item() / y_label.shape[0]
            # acc = metrics.accuracy_score(y_label, y_hat_label)
            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)
            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

        else:
            loss = F.mse_loss(y_hat, y)
            # if self.device == "gpu":
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()
            # else:
            #     y_hat_pos = y_hat.detach().numpy()
            #     y_pos = y.numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(
                self.marker_posarray[None, :, :] - y_hat_pos[:, None, :],
                axis=2)
            # get closest marker index
            min_indices = np.argmin(distances, axis=1) 
            min_indices = torch.tensor(min_indices, dtype=torch.long)
            # get closest class indices
            y_label = []
            for i, pos in enumerate(y_pos):
                rounded_pos = tuple(np.round(pos, decimals=4))
                # get class index from coord using reverse mapping
                y_label.append(int(self.rev_marker_positions.get(rounded_pos)))
                appended = int(self.rev_marker_positions.get(rounded_pos))
            y_label = torch.tensor(y_label, dtype=torch.long).to(device)
            # calculate accuracy based on whether classes are the same
            # acc = torch.sum(min_indices == y_label).item() / y_label.shape[0]

            min_coords = self.marker_posarray[min_indices]
            # acc = np.sum(min_coords == y_pos) / len(y_pos)
            # threshold = 1e-5
            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            # correct = np.linalg.norm(y_pos - min_coords, axis=1) < threshold
            acc = np.mean(correct)

        self.log("train/train_loss", loss, on_epoch = True, prog_bar=True)
        self.log("train/train_acc", acc, on_epoch = True, prog_bar=True)
        self.log("train/train_dist", euclidean_distance.mean(), on_epoch = True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        device = self._get_device()
        x, y = batch["joint_data"].float().to(device), batch["contact_label"].float().to(device)
        y_hat = self(x)

        threshold = 0.1
        if self.classify:
            # if self.device == "gpu":
            #     y_hat_idx = torch.argmax(y_hat, dim=1).cpu()
            #     y_idx = torch.argmax(y, dim=1).cpu()
            #     loss = F.cross_entropy(y_hat.cpu(), y_idx)
            # else:
            y_hat_idx = torch.argmax(y_hat, dim=1)
            y_idx = torch.argmax(y, dim=1)
            loss = F.cross_entropy(y_hat, y_idx)

            y_hat_label = y_hat_idx.float()         
            y_label = y_idx.float()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx])
            # acc = torch.sum(y_hat_label == y_label).item() / y_label.shape[0]
            # acc = metrics.accuracy_score(y_label, y_hat_label)
            correct = np.linalg.norm(y_pos - y_hat_pos, axis=1) < threshold
            acc = np.mean(correct)

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

        else:
            loss = F.mse_loss(y_hat, y)
            # if self.device == "gpu":
            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()
            # else:
            #     y_hat_pos = y_hat.detach().numpy()
            #     y_pos = y.numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(
                self.marker_posarray[None, :, :] - y_hat_pos[:, None, :],
                axis=2)
            # get closest marker index
            min_indices = np.argmin(distances, axis=1) 
            min_indices = torch.tensor(min_indices, dtype=torch.long)
            # get closest class indices
            y_label = []
            for i, pos in enumerate(y_pos):
                rounded_pos = tuple(np.round(pos, decimals=4))
                # get class index from coord using reverse mapping
                y_label.append(int(self.rev_marker_positions.get(rounded_pos)))
                appended = int(self.rev_marker_positions.get(rounded_pos))
            y_label = torch.tensor(y_label, dtype=torch.long).to(device)
            # calculate accuracy based on whether classes are the same
            # acc = torch.sum(min_indices == y_label).item() / y_label.shape[0]

            min_coords = self.marker_posarray[min_indices]
            # acc = np.sum(min_coords == y_pos) / len(y_pos)
            # threshold = 1e-5
            correct = np.linalg.norm(y_pos - min_coords, axis=1) < threshold
            acc = np.mean(correct)

        self.log("val/val_loss", loss, on_epoch = True, prog_bar=True)
        self.log("val/val_acc", acc, on_epoch = True, prog_bar=True)
        self.log("val/val_dist", euclidean_distance.mean(), on_epoch = True, prog_bar=True)

        return loss
    

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.logged_metrics.get("val/val_loss")
        val_acc = self.trainer.logged_metrics.get("val/val_acc")
        val_dist = self.trainer.logged_metrics.get("val/val_dist")
        print(f"Epoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Euclidean Distance: {val_dist:.4f}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def predict(self, inputs):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            return self(inputs)