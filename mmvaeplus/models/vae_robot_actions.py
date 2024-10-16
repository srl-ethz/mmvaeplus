import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from mmvaeplus.utils import Constants
from .base_vae import VAE
from .encoder_decoder_blocks.mlp_robot_actions import RobotActionEncoder, RobotActionDecoder

class RobotActionVAE(VAE):
    def __init__(self, input_dim, enc_hidden_dim, dec_hidden_dim, params):
        super(RobotActionVAE, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,  # prior
            dist.Normal,  # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,  # posterior
            RobotActionEncoder(input_dim, params.latent_dim_w, params.latent_dim_z, params.priorposterior, enc_hidden_dim, params.num_hidden_layers),  # Encoder
            RobotActionDecoder(params.latent_dim_w + params.latent_dim_z, input_dim, dec_hidden_dim, params.num_hidden_layers),  # Decoder
            params
        )
        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=True)
        ])
        self.input_dim = input_dim
        self.llik_scaling = 1.0
        self.params = params

    @property
    def pw_params_aux(self):
        if self.params.priorposterior == 'Normal':
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + Constants.eta
        else:
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(-1) + Constants.eta

class RobotAction11d(RobotActionVAE):
    def __init__(self, params):
        super(RobotAction11d, self).__init__(11, params.faive_enc_hidden_dim, params.faive_dec_hidden_dim, params)
        self.modelName = 'RobotAction11d'

class RobotAction45d(RobotActionVAE):
    def __init__(self, params):
        super(RobotAction45d, self).__init__(45, params.mano_enc_hidden_dim, params.mano_dec_hidden_dim, params)
        self.modelName = 'RobotAction45d'

class RobotAction1d(RobotActionVAE):
    def __init__(self, params):
        super(RobotAction1d, self).__init__(1, params.gripper_enc_hidden_dim, params.gripper_dec_hidden_dim, params)
        self.modelName = 'RobotAction1d'
