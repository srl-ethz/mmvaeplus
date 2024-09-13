import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RobotActionsDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True).item()
        self.hand_pose = torch.tensor(self.data['hand_pose'], dtype=torch.float32)
        self.faive_angles = torch.tensor(self.data['faive_angles'], dtype=torch.float32)
        self.onedof_pose = torch.tensor(self.data['1dof_pose'], dtype=torch.float32)
        
        assert len(self.hand_pose) == len(self.faive_angles) == len(self.onedof_pose), "Data lengths do not match"

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

    return train_loader, test_loader
