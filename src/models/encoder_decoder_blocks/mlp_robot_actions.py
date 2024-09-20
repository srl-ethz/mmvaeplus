import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(MLP, self).__init__()
        layers = []

        if len(hidden_dims) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        elif len(hidden_dims) == 1:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dims[0]))
            layers.append(nn.Linear(hidden_dims[0], output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dims[0]))
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dims[i]))

            layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class RobotActionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim_w, latent_dim_z, dist, hidden_dim=2048, num_hidden_layers=2):
        super(RobotActionEncoder, self).__init__()
        self.dist = dist
        self.input_dim = input_dim
        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z

        # Separate MLPs for w and z
        self.mlp_w_mu = MLP(input_dim, latent_dim_w, [hidden_dim]*num_hidden_layers)
        self.mlp_w_lv = MLP(input_dim, latent_dim_w, [hidden_dim]*num_hidden_layers)
        self.mlp_z_mu = MLP(input_dim, latent_dim_z, [hidden_dim]*num_hidden_layers)
        self.mlp_z_lv = MLP(input_dim, latent_dim_z, [hidden_dim]*num_hidden_layers)

    def forward(self, x):
        mu_w = self.mlp_w_mu(x)
        lv_w = self.mlp_w_lv(x)
        mu_z = self.mlp_z_mu(x)
        lv_z = self.mlp_z_lv(x)

        if self.dist == 'Normal':
            return torch.cat((mu_w, mu_z), dim=-1), \
                   torch.cat((F.softplus(lv_w) + Constants.eta,
                              F.softplus(lv_z) + Constants.eta), dim=-1)
        else:
            return torch.cat((mu_w, mu_z), dim=-1), \
                   torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta,
                              F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta), dim=-1)

class RobotActionDecoder(nn.Module):
    def __init__(self, latent_dim_u, output_dim, hidden_dim=2048, num_hidden_layers=2):
        super(RobotActionDecoder, self).__init__()
        self.latent_dim_u = latent_dim_u
        self.output_dim = output_dim

        self.decoder = MLP(latent_dim_u, output_dim, [hidden_dim]*num_hidden_layers)

    def forward(self, u):
        # returning mean and length scale, hardcoded?
        return self.decoder(u), torch.tensor(0.01).to(u.device)
