import argparse
import os
from natsort import natsorted
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from network import LitSpot
from dataset import SpotDataModule


def main(num_classes, markers_path, classify = False):
    seed_everything(42)

    if classify:
        tb_logger = TensorBoardLogger("gouger_logs", name="classification")
    else:
        tb_logger = TensorBoardLogger("gouger_logs", name="regression")

    data_module = SpotDataModule(batch_size=256)
    if classify:
        output_dim = num_classes
    else:
        output_dim = 3
    model = LitSpot(input_dim = 24, output_dim = output_dim, markers_path = markers_path, classify=classify)

    checkpoint_callback = ModelCheckpoint(monitor="val/val_acc", mode="max", save_last=True, filename="best")
    early_stop_callback = EarlyStopping(monitor="val/val_acc", patience=100, mode="max")

    trainer = Trainer(
        accelerator="gpu", max_epochs=40,
        logger=[tb_logger],
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(model, data_module)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--session', type = str, required=True, help='Session for data collection')
    parser.add_argument('--data_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')

    options = parser.parse_args()

    classify = options.classify
    session = options.session
    data_dir = options.data_dir
    markers_path = options.markers_path

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

    main(num_classes, markers_path, classify)
    