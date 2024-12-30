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
    def __init__(self, input_dim: int, output_dim: int, classify=False) -> None:
        super().__init__()
        if classify:
            self.network = nn.Sequential(
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
        else:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
    
    def forward(self, x):
        return self.network(x)

class LitSpot(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, markers_path, classify=False) -> None:
        super().__init__()

        self.model = JointNetwork(input_dim, output_dim, classify)


        self.classify = classify
        self.learning_rate = 1e-3

        markers_pos = np.loadtxt(markers_path, delimiter=',')
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
        self.marker_positions['100'] = np.array([0, 0, 0])

        self.marker_posarray = np.array(list(self.marker_positions.values()))
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["joint_data"].float(), batch["contact_label"].float()
        y_hat = self(x) # shape batch_size by 3

        if self.classify:
            # print(f"y_hat: {y_hat.numpy().shape}")
            # print(f"y: {y.numpy().shape}")

            # y_hat_idx = np.argmax(y_hat.detach().numpy(), axis=1)
            y_hat_idx = torch.argmax(y_hat, dim=1)
            # y_idx = np.argmax(y.numpy(), axis=1)
            y_idx = torch.argmax(y, dim=1) 

            # y_hat_label = torch.tensor(y_hat_idx).float()
            # y_label = torch.tensor(y_idx).float()
            y_hat_label = y_hat_idx.float()         # Convert to float if needed
            y_label = y_idx.float()

            loss = F.cross_entropy(y_hat, y_idx)
            # print(f"loss requires grad:{loss.requires_grad}")

            # sys.exit()

            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx]) 

            acc = metrics.accuracy_score(y_label, y_hat_label)
            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

        else:
            loss = F.mse_loss(y_hat, y)

            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(
                self.marker_posarray[None, :, :] - y_hat_pos[:, None, :],
                axis=2)
            min_indices = np.argmin(distances, axis=1) 
            min_coords = self.marker_posarray[min_indices]
            # acc = np.sum(min_coords == y_pos) / len(y_pos)
            threshold = 1e-5
            correct = np.linalg.norm(y_pos - min_coords, axis=1) < threshold
            acc = np.mean(correct)

        self.log("train/train_loss", loss, on_epoch = True, prog_bar=True)
        self.log("train/train_acc", acc, on_epoch = True, prog_bar=True)
        self.log("train/train_dist", euclidean_distance.mean(), on_epoch = True, prog_bar=True)

        # return {"loss": loss, "acc": acc, "dist": euclidean_distance.mean()}
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["joint_data"].float(), batch["contact_label"].float()
        y_hat = self(x)

        if self.classify:
            # y_hat_idx = np.argmax(y_hat.detach().numpy(), axis=1)
            # y_idx = np.argmax(y.numpy(), axis=1)

            # y_hat_label = torch.tensor(y_hat_idx).float()
            # y_label = torch.tensor(y_idx).float()
            y_hat_idx = torch.argmax(y_hat, dim=1)  
            y_idx = torch.argmax(y, dim=1)          

            y_hat_label = y_hat_idx.float()         
            y_label = y_idx.float()

            loss = F.cross_entropy(y_hat, y_idx)

            # y_hat_pos = np.array([self.marker_positions.get(str(i)) for i in y_hat_idx])
            # y_pos = np.array([self.marker_positions.get(str(i)) for i in y_idx])
            y_hat_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_hat_idx])
            y_pos = np.array([self.marker_positions.get(str(i.item())) for i in y_idx])

            acc = metrics.accuracy_score(y_label, y_hat_label)
            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

        else:
            loss = F.mse_loss(y_hat, y)

            y_hat_pos = y_hat.detach().cpu().numpy()
            y_pos = y.cpu().numpy()

            euclidean_distance = np.linalg.norm(y_pos - y_hat_pos, axis=1)

            # calculate the distance from y_hat_pos to all the marker positions
            distances = np.linalg.norm(
                self.marker_posarray[None, :, :] - y_hat_pos[:, None, :],
                axis=2)
            min_indices = np.argmin(distances, axis=1) 
            min_coords = self.marker_posarray[min_indices]
            # acc = np.sum(min_coords == y_pos) / len(y_pos)
            threshold = 1e-5
            correct = np.linalg.norm(y_pos - min_coords, axis=1) < threshold
            acc = np.mean(correct)

        self.log("val/val_loss", loss, on_epoch = True, prog_bar=True)
        self.log("val/val_acc", acc, on_epoch = True, prog_bar=True)
        self.log("val/val_dist", euclidean_distance.mean(), on_epoch = True, prog_bar=True)

        # return {"val_loss": loss, "val_acc": acc, "val_dist": euclidean_distance.mean()}
        return loss
    
    # def on_train_epoch_end(self, outputs):
    # # def on_train_epoch_end(self):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     # avg_loss = self.trainer.logged_metrics.get("train/train_loss")
    #     train_acc = torch.stack([x['acc'] for x in outputs]).mean()
    #     # train_acc = self.trainer.logged_metrics.get("train/train_acc")
    #     train_dist = torch.stack([x['dist'] for x in outputs]).mean()
    #     # train_dist = self.trainer.logged_metrics.get("train/train_euclidean_distance")
    #     print(f"avg_loss: {avg_loss}")
    #     print(f"train_acc: {train_acc}")
    #     print(f"train_dist: {train_dist}")
    #     print(f"self.current_epoch: {self.current_epoch}")
    #     print(f"Epoch {self.current_epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Train Euclidean Distance: {train_dist:.4f}")

    def on_validation_epoch_end(self):
        # avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_loss = self.trainer.logged_metrics.get("val/val_loss")
        val_acc = self.trainer.logged_metrics.get("val/val_acc")
        val_dist = self.trainer.logged_metrics.get("val/val_dist")
        print(f"Epoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Euclidean Distance: {val_dist:.4f}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer