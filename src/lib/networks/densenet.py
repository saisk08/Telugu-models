from torch import nn
from utils.layers import res_block, conv_block


def conv(ic, oc, **kwargs): return conv_block(ic, oc, **kwargs)


def dense(c): return res_block(c, dense=True)


def conv_and_dense(ic, oc, **kwargs):
    return nn.Sequential(
        conv(ic, oc, **kwargs),
        dense(oc)
    )


class Telnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_and_res(1, 8, kernel_size=5, stride=2, padding=2),  # 16
            conv_and_res(8, 16),  # 8
            conv_and_res(16, 32),  # 4
            conv_and_res(32, 24),  # 2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)


class Siameserdm(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.feats = nn.Sequential(
            conv_and_dense(1, 8),  # 16
            conv_and_dense(8, 16),  # 8
            conv_and_dense(16, 32),  # 4
            conv_and_dense(32, 64)  # 2
        )
        self.classifier = nn.Sequential(
            conv(64, 1),  # 1
            nn.Flatten()
        )

    def forward(self, x):
        x = self.feats(x)
        x = self.classifier(x)
        return x


class Siamesecat(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.net = self.create(out)

    def conv(self, ic, oc): return conv_block(
        ic, oc, kernel_szie=3, stride=2, padding=1)

    def create(self, out):
        return nn.Sequential(
            conv_and_dense(1, 8),  # 16
            conv_and_dense(8, 16),  # 8
            conv_and_dense(16, 64),  # 4
            conv_and_dense(64, 128),  # 2
            conv_and_dense(256, 300),  # 1
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)
