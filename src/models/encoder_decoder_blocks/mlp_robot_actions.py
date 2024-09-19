import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 128], output_dim=None):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)

class RobotActionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim_w, latent_dim_z, dist):
        super(RobotActionEncoder, self).__init__()
        self.dist = dist
        self.input_dim = input_dim
        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z

        # Separate MLPs for w and z
        self.mlp_w_mu = MLP(input_dim, [2048, 2048], latent_dim_w)
        self.mlp_w_lv = MLP(input_dim, [2048, 2048], latent_dim_w)
        self.mlp_z_mu = MLP(input_dim, [2048, 2048], latent_dim_z)
        self.mlp_z_lv = MLP(input_dim, [2048, 2048], latent_dim_z)

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
    def __init__(self, latent_dim_u, output_dim):
        super(RobotActionDecoder, self).__init__()
        self.latent_dim_u = latent_dim_u
        self.output_dim = output_dim

        self.decoder = MLP(latent_dim_u, [2048, 2048], output_dim)

    def forward(self, u):
        # returning mean and length scale, hardcoded?
        return self.decoder(u), torch.tensor(0.01).to(u.device)

class RobotActionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim_w, latent_dim_z, dist):
        super(RobotActionVAE, self).__init__()
        self.latent_dim_u = latent_dim_w + latent_dim_z
        self.encoder = RobotActionEncoder(input_dim, latent_dim_w, latent_dim_z, dist)
        self.decoder = RobotActionDecoder(self.latent_dim_u, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
