from torch import nn
from .layers import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class):
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid,n_class)
    def forward(self, x, G):

        x1 = F.relu(self.hgc1(x, G))
        return x1
