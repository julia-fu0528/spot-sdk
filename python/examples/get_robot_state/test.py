import numpy as np
import sys
import torch
import torch.nn.functional as F

# from visualize_robot_state import load_joint_torques, load_joint_positions

# alist = []
# torque_path = "data/gouger1209/0/0.npy"
# torque, _, _, _ = load_joint_torques("data/gouger1209/0/0.npy")
# joint_angle, _, _, _ = load_joint_positions("data/gouger1209/0/0.npy")
# data = np.hstack((torque, joint_angle))
# for i in data:
#     alist.append(i)
# for i in data:
#     alist.append(i)

label = 1
label = F.one_hot(torch.tensor(label), num_classes=10)
labels = np.tile(label, (5, 1))
print(f"label:{labels}")
blist = []
for i in range(5):
    blist.append(label)
print(f"blist:{len(blist)}")
for i in range(5):
    blist.append(label)
print(f"blist:{len(blist)}")
print(f"blist:{blist[0].shape}")


