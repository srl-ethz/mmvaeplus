import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Constants

class RobotActionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim_w, latent_dim_z, dist):
        super(RobotActionEncoder, self).__init__()
        self.dist = dist
        self.input_dim = input_dim
        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim_z

        # Separate branches for w and z, for now just simple linear layers
        self.fc_mu_w = nn.Linear(input_dim, latent_dim_w)
        self.fc_lv_w = nn.Linear(input_dim, latent_dim_w)
        self.fc_mu_z = nn.Linear(input_dim, latent_dim_z)
        self.fc_lv_z = nn.Linear(input_dim, latent_dim_z)

    def forward(self, x):
        # check for nans
        assert not torch.isnan(x).any(), "NaN in x"
        print(f'Input shape: {x.shape}')
        print(f'Expecting shape: {self.input_dim}')

        mu_w = self.fc_mu_w(x)
        lv_w = self.fc_lv_w(x)
        mu_z = self.fc_mu_z(x)
        lv_z = self.fc_lv_z(x)

        # check for nans
        assert not torch.isnan(mu_w).any(), "NaN in mu_w"
        assert not torch.isnan(lv_w).any(), "NaN in lv_w"
        assert not torch.isnan(mu_z).any(), "NaN in mu_z"
        assert not torch.isnan(lv_z).any(), "NaN in lv_z"

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

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_u, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

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
