import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.distributions as dist
from .mmvaeplus import MMVAEplus
from .vae_robot_actions import RobotAction11d, RobotAction45d, RobotAction1d
from mmvaeplus.utils import Constants
from mmvaeplus.dataset_robot_actions import get_robot_actions_dataloaders
import json
import os
import pickle

def build_model(checkpoint_dir, epoch, device='cuda'):
    args = torch.load(os.path.join(checkpoint_dir, f'args.rar'), map_location=device)
    model = RobotActions(args)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model_{epoch}.rar'), map_location=device))
    model = model.to(device)
    with open(os.path.join(checkpoint_dir, f'dataset_stats.pkl'), 'rb') as f:
        dataset_stats = pickle.load(f)
    return model, dataset_stats

class RobotActions(MMVAEplus):
    """
    MMVAEplus subclass for Robot Actions Experiment
    """
    def __init__(self, params):
        super(RobotActions, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,
            params,
            RobotAction11d,
            RobotAction45d,
            RobotAction1d
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False)  # logvar
        ])
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.modelName = 'MMVAEplus_RobotActions'

        # Fix model names for individual models to be saved
        for idx, vae in enumerate(self.vaes):
            vae.modelName = f'VAE_RobotAction_{vae.input_dim}d'

        self.params = params

        self.calculate_llik_scaling()

    def calculate_llik_scaling(self):
        # Calculate the "size" of each modality
        sizes = [vae.input_dim for vae in self.vaes]
        
        # Choose the median size as the reference
        reference_size = sorted(sizes)[1]
        
        # Calculate scaling factors
        scalings = [reference_size / size for size in sizes]
        
        # Apply scaling factors to each VAE
        for vae, scaling in zip(self.vaes, scalings):
            # print(f'Scaling factor for {vae.input_dim}d: {scaling}')
            vae.llik_scaling = scaling

    @property
    def pz_params(self):
        """
        Returns: Parameters of prior distribution for shared latent code
        """
        if self.params.priorposterior == 'Normal':
            return self._pz_params[0], F.softplus(self._pz_params[1]) + Constants.eta
        else:
            return self._pz_params[0], F.softmax(self._pz_params[1], dim=-1) * self._pz_params[1].size(-1) + Constants.eta

    @property
    def pw_params(self):
        """
        Returns: Parameters of prior distribution for modality-specific latent code
        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params[0], F.softplus(self._pw_params[1]) + Constants.eta
        else:
            return self._pw_params[0], F.softmax(self._pw_params[1], dim=-1) * self._pw_params[1].size(-1) + Constants.eta

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # Implement the data loading for robot actions
        data_path = '/mnt/data/erbauer/retargeting/retargeted_hand_dataset_combined_grab.npy'  # Update this path
        train_loader, test_loader, dataset_stats = get_robot_actions_dataloaders(
            data_path, batch_size, shuffle=shuffle, split_ratio=0.8, device=device
        )
        return train_loader, test_loader, dataset_stats


    def self_and_cross_modal_generation(self, data, num=3,N=10):
        """
        Self- and cross-modal generation.
        Args:
            data: Input
            num: Number of samples to be considered for generation -> max. number of different modalities
            N: Number of generations

        Returns:
            Generations
        """
        recon_triess = [[[] for i in range(num)] for j in range(num)]
        outputss = [[[] for i in range(num)] for j in range(num)]
        for i in range(N):
            recons_mat = super(RobotActions, self).self_and_cross_modal_generation([d[:num] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                      recon = recon.squeeze(0).cpu()
                      recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                # outputss[r][o] = make_grid(torch.cat([data[r][:num].cpu()]+recon_triess[r][o]), nrow=num)
                outputss[r][o] = (data[r][:num].cpu(), recon_triess[r][o])
        return outputss
    
    def targeted_generation(self, data, src_modality, target_modality = 'all'):
        """
        Targeted generation.
        Args:
            data: Input (batch_size, in_modality_dim)
            src_modality: Source modality (0,1,2) ('gc_angles', 'mano_pose', 'simple_gripper')
            target_modality: Target modality (0,1,2) ('gc_angles', 'mano_pose', 'simple_gripper')

        Returns:
            Generations (batch_size, out_modality_dim)
        """
        # inputs: pad non-src modalities with zeros
        device = data.device
        inputs = [torch.zeros((data.shape[0], vae.input_dim), device=device) for vae in self.vaes]
        inputs[src_modality] = data
        recons_mat = super(RobotActions, self).self_and_cross_modal_generation(inputs)

        if isinstance(target_modality, int):
            return recons_mat[src_modality][target_modality]
        elif target_modality == 'all':
            return recons_mat


