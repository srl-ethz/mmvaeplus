import torch
from torch import nn
import torch.nn.functional as F
from mmvaeplus.utils import Constants

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


# Classes
class Enc(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latentDim_w, latentDim_z, dist):
        super(Enc, self).__init__()
        self.dist = dist
        self.embedding = nn.Linear(vocabSize, embeddingDim)
        self.enc_w = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True)
        )
        self.enc_z = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 8, fBase * 16, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1_w = nn.Linear(fBase * 16 * 16, latentDim_w)
        self.c2_w = nn.Linear(fBase * 16 * 16, latentDim_w)

        self.c1_z = nn.Conv2d(fBase * 16, latentDim_z, 4, 1, 0, bias=True)
        self.c2_z = nn.Conv2d(fBase * 16, latentDim_z, 4, 1, 0, bias=True)

        # self.ll_c1_z = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_z)])
        # self.ll_c2_z = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_z)])
        # self.ll_c1_w = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_w)])
        # self.ll_c2_w = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_w)])

    def forward(self, x):
        x_emb = self.embedding(x).unsqueeze(1)
        e_w = self.enc_w(x_emb)
        e_w = e_w.view(-1, fBase * 16 * 16)
        mu_w, lv_w = self.c1_w(e_w), self.c2_w(e_w)
        e_z = self.enc_z(x_emb)
        mu_z, lv_z = self.c1_z(e_z).squeeze(), self.c2_z(e_z).squeeze()
        # mu_z, lv_z = self.c1_z(e_z).squeeze().unsqueeze(0), self.c2_z(e_z).squeeze().unsqueeze(0)
        if self.dist == 'Normal':
            return torch.cat((mu_w, mu_z), dim=-1), \
                torch.cat((F.softplus(lv_w) + Constants.eta, F.softplus(lv_z) + Constants.eta), dim=-1)
        else:
            return torch.cat((mu_w, mu_z), dim=-1), \
                torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta,
                           F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta), dim=-1)


class Dec(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latentDim_w, latentDim_z):
        super(Dec, self).__init__()
        self.dec_w = nn.Sequential(
            nn.ConvTranspose2d(latentDim_w, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_z = nn.Sequential(
            nn.ConvTranspose2d(latentDim_z, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 8, fBase * 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_h = nn.Sequential(
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=True),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

        self.latent_dim_w = latentDim_w
        self.latent_dim_z = latentDim_z

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, u):
        # z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        w, z = torch.split(u, [self.latent_dim_w, self.latent_dim_z], dim=-1)
        z = z.unsqueeze(-1).unsqueeze(-1)
        hz = self.dec_z(z.view(-1, *z.size()[-3:]))
        w = w.unsqueeze(-1).unsqueeze(-1)
        hw = self.dec_w(w.view(-1, *w.size()[-3:]))
        h = torch.cat((hw, hz), dim=1)
        out = self.dec_h(h)
        out = out.view(*z.size()[:-3], *out.size()[1:]).view(-1, embeddingDim)
        # The softmax is key for this to work
        ret = [self.softmax(self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize))]
        return ret