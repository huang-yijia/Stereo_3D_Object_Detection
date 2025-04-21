# check_weights.py

import torch

# Load the checkpoint
ckpt_path = "transformer_depth_kitti.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")

# If the file is a plain state_dict (not wrapped), print all parameter keys
print("All parameter keys in the checkpoint:")
for key in state_dict.keys():
    print(f"  {key} \t→ shape: {state_dict[key].shape}")
