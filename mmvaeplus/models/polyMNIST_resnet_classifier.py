# PolyMNIST resnet classifier model specification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Constants
dataSize = torch.Size([3, 28, 28])

def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out

class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class EncC(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim):
        super().__init__()
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc_mu = nn.Linear(self.nf0*s0*s0, ndim)
        #self.ll = nn.Linear(ndim, 10)
        #self.fc_lv = nn.Linear(self.nf0*s0*s0, ndim)

    def forward(self, x):
        # batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(out.size()[0], self.nf0*self.s0*self.s0)
        #lv = self.fc_lv(out)
        #if self.params.priorposterior == 'Normal':
        #    return self.fc_mu(out), F.softplus(lv) + Constants.eta
        #else:
        #    return self.fc_mu(out), F.softmax(lv, dim=-1)*lv.size(-1) + Constants.eta
        #return self.ll(self.fc_mu(out))
        return self.fc_mu(out)

# Classes
class MMNetwork(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim, nmods, dev):
        super().__init__()
        self.encoders = [EncC(ndim).to(dev) for i in range(nmods)]
        self.nmods = nmods
        self.ll = nn.Sequential(*[nn.ReLU(inplace=True), nn.Linear(ndim*nmods, ndim*2),
                                  nn.ReLU(inplace=True), nn.Linear(ndim*2, 10)])

    def forward(self, x):
        # batch_size = x.size(0)
        repres = [self.encoders[m](x[m]) for m in range(self.nmods)]
        repres = torch.cat(repres, dim=-1)
        logits = self.ll(repres)
        return logits
