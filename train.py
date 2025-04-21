from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
import torch.nn.functional as F
import os, cv2
import numpy as np
import torch

from transformer_depth_model import TransformerDepthEstimator

class KITTIDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.left_images = sorted(os.listdir(os.path.join(root_dir, 'image_2')))
        self.right_images = sorted(os.listdir(os.path.join(root_dir, 'image_3')))
        self.disps = sorted(os.listdir(os.path.join(root_dir, 'disp_occ_0')))
        self.root = root_dir
        self.transform = transform
        self.focal = 721.5377
        self.baseline = 0.54

    def __len__(self):
        return min(len(self.left_images), len(self.right_images), len(self.disps))
    
    def __getitem__(self, idx):
        left = cv2.imread(os.path.join(self.root, 'image_2', self.left_images[idx]))
        right = cv2.imread(os.path.join(self.root, 'image_3', self.right_images[idx]))
        disp_path = os.path.join(self.root, 'disp_occ_0', self.disps[idx])
        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)

        if disp is None:
            raise ValueError(f"Failed to load disparity map at {disp_path}")

        disp = disp.astype(np.float32) / 256.0
        mask = disp > 0
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[mask] = (self.focal * self.baseline) / disp[mask]

        if self.transform:
            left = self.transform(left)
            right = self.transform(right)

            depth = cv2.resize(depth, (left.shape[2], left.shape[1]), interpolation=cv2.INTER_NEAREST)

        depth = torch.from_numpy(depth).unsqueeze(0).float()
        return left, right, depth


def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for left, right, gt_depth in dataloader:
            left, right, gt_depth = left.to(device), right.to(device), gt_depth.to(device)

            pred_depth = model(left, right)
            loss = F.l1_loss(pred_depth, gt_depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 512)),
        T.ToTensor()
    ])

    dataset = KITTIDepthDataset("/home/hyj/workplace/KITTI/training", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = TransformerDepthEstimator()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloader, optimizer, device='cpu')

    torch.save(model.state_dict(), "transformer_depth_kitti.pth")