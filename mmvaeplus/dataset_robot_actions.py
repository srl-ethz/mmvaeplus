import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RobotActionsDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True).item()
        self.hand_pose = torch.tensor(self.data['pose'], dtype=torch.float32)
        # TODO: add shape to MANO input
        self.hand_shape = torch.tensor(self.data['shape'], dtype=torch.float32)
        self.faive_angles = torch.tensor(self.data['faive_angles'], dtype=torch.float32)
        self.onedof_pose = torch.tensor(self.data['simple_gripper'], dtype=torch.float32)
        
        # normalize data, possibly save these?
        self.hand_pose_mean = self.hand_pose.mean()
        self.hand_pose_std = self.hand_pose.std()
        self.faive_angles_mean = self.faive_angles.mean()
        self.faive_angles_std = self.faive_angles.std()
        self.onedof_pose_mean = self.onedof_pose.mean()
        self.onedof_pose_std = self.onedof_pose.std()

        self.dataset_stats = {
            'hand_pose': {
                'mean': self.hand_pose_mean,
                'std': self.hand_pose_std
            },
            'faive_angles': {
                'mean': self.faive_angles_mean,
                'std': self.faive_angles_std
            },
            'onedof_pose': {
                'mean': self.onedof_pose_mean,
                'std': self.onedof_pose_std
            }
        }

        self.hand_pose = (self.hand_pose - self.hand_pose_mean) / self.hand_pose_std
        self.faive_angles = (self.faive_angles - self.faive_angles_mean) / self.faive_angles_std
        self.onedof_pose = (self.onedof_pose - self.onedof_pose_mean) / self.onedof_pose_std


        assert len(self.hand_pose) == len(self.faive_angles) == len(self.onedof_pose), "Data lengths do not match"
        assert torch.isnan(self.hand_pose).any() == False, "NaN in hand_pose"
        assert torch.isnan(self.faive_angles).any() == False, "NaN in faive_angles"
        assert torch.isnan(self.onedof_pose).any() == False, "NaN in onedof_pose"

    def __len__(self):
        return len(self.hand_pose)

    def __getitem__(self, idx):

        return {
            'hand_pose': self.hand_pose[idx],
            'faive_angles': self.faive_angles[idx],
            '1dof_pose': self.onedof_pose[idx]
        }

def get_robot_actions_dataloaders(data_path, batch_size, shuffle=True, split_ratio=0.8, device="cuda"):
    dataset = RobotActionsDataset(data_path)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader, test_loader, dataset.dataset_stats
