import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from .mmvaeplus import MMVAEplus
from .vae_robot_actions import RobotAction11d, RobotAction45d, RobotAction1d
from utils import Constants
from dataset_robot_actions import get_robot_actions_dataloaders

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
        data_path = '/mnt/data/erbauer/retargeting/retargeted_hand_dataset_combined_test.npy'  # Update this path
        train_loader, test_loader = get_robot_actions_dataloaders(
            data_path, batch_size, shuffle=shuffle, split_ratio=0.8, device=device
        )
        return train_loader, test_loader

    def self_and_cross_modal_generation(self, data, num=10, N=10):
        """
        Self- and cross-modal generation for robot actions.
        Args:
            data: Input data (list of tensors for each modality)
            num: Number of samples to be considered for generation
            N: Number of generations

        Returns:
            List of lists containing generated samples for each modality
        """
        recon_tries = [[[] for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        outputs = [[[] for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for _ in range(N):
            recons_mat = super(RobotActions, self).self_and_cross_modal_generation([d[:num] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    recon = recon.cpu()
                    recon_tries[r][o].append(recon)

        for r, recons_list in enumerate(recons_mat):
            for o, _ in enumerate(recons_list):
                outputs[r][o] = torch.stack(recon_tries[r][o])

        return outputs
