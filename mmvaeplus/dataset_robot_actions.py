import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer



class RobotActionsDataset(Dataset):
    def __init__(self, data_path, normalization_type='z_score'):
        self.data = np.load(data_path, allow_pickle=True).item()
        self.hand_pose = torch.tensor(self.data['pose'], dtype=torch.float32)
        self.faive_angles = torch.tensor(self.data['faive_angles'], dtype=torch.float32)
        self.onedof_pose = torch.tensor(self.data['simple_gripper'], dtype=torch.float32)
        print(self.hand_pose.shape, self.faive_angles.shape, self.onedof_pose.shape)

        self.normalization_type = normalization_type
        self.dataset_stats = {}

        self.normalize_data()

        assert len(self.hand_pose) == len(self.faive_angles) == len(self.onedof_pose), "Data lengths do not match"
        assert torch.isnan(self.hand_pose).any() == False, "NaN in hand_pose"
        assert torch.isnan(self.faive_angles).any() == False, "NaN in faive_angles"
        assert torch.isnan(self.onedof_pose).any() == False, "NaN in onedof_pose"

    def normalize_data(self):
        if self.normalization_type == 'z_score':
            self.z_score_normalization()
        elif self.normalization_type == 'minmax':
            self.minmax_normalization()
        elif self.normalization_type == 'robust':
            self.robust_normalization()
        elif self.normalization_type == 'quantile':
            self.quantile_normalization()
        else:
            raise ValueError("Unknown normalization type")

    def z_score_normalization(self):
        for name, data in [('hand_pose', self.hand_pose), ('faive_angles', self.faive_angles), ('onedof_pose', self.onedof_pose)]:
            mean = data.mean(dim=0)
            std = data.std(dim=0)
            std[std < 1e-6] = 1e-6  # Avoid division by zero
            normalized_data = (data - mean) / std
            setattr(self, name, normalized_data)
            self.dataset_stats[name] = {'mean': mean, 'std': std}

    def minmax_normalization(self):
        for name, data in [('hand_pose', self.hand_pose), ('faive_angles', self.faive_angles), ('onedof_pose', self.onedof_pose)]:
            min_val = data.min(dim=0).values
            max_val = data.max(dim=0).values
            range_val = max_val - min_val
            range_val[range_val < 1e-6] = 1e-6  # Avoid division by zero
            normalized_data = (data - min_val) / range_val
            setattr(self, name, normalized_data)
            self.dataset_stats[name] = {'min': min_val, 'max': max_val}

    def robust_normalization(self):
        for name, data in [('hand_pose', self.hand_pose), ('faive_angles', self.faive_angles), ('onedof_pose', self.onedof_pose)]:
            scaler = RobustScaler()
            normalized_data = torch.tensor(scaler.fit_transform(data.numpy()), dtype=torch.float32)
            setattr(self, name, normalized_data)
            self.dataset_stats[name] = {'center': torch.tensor(scaler.center_), 'scale': torch.tensor(scaler.scale_)}

    def quantile_normalization(self):
        for name, data in [('hand_pose', self.hand_pose), ('faive_angles', self.faive_angles), ('onedof_pose', self.onedof_pose)]:
            transformer = QuantileTransformer(output_distribution='normal')
            normalized_data = torch.tensor(transformer.fit_transform(data.numpy()), dtype=torch.float32)
            setattr(self, name, normalized_data)
            self.dataset_stats[name] = {'quantiles': torch.tensor(transformer.quantiles_)}

    def __len__(self):
        return len(self.hand_pose)

    def __getitem__(self, idx):
        return {
            'hand_pose': self.hand_pose[idx],
            'faive_angles': self.faive_angles[idx],
            '1dof_pose': self.onedof_pose[idx]
        }

def get_robot_actions_dataloaders(data_path, batch_size, normalization_type='z_score', shuffle=True, split_ratio=0.8, device="cuda"):
    dataset = RobotActionsDataset(data_path, normalization_type)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    kwargs = {'num_workers': 16, 'pin_memory': True, 'prefetch_factor':32} if device == "cuda" else {}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader, test_loader, dataset.dataset_stats
