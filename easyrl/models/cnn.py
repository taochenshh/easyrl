import torch
import torch.nn as nn


class NatureDQNCNN(nn.Module):
    def __init__(self, in_channels=3, format='NCHW'):
        # input height = width = 64
        super().__init__()
        if format not in ['NCHW', 'NHWC']:
            raise TypeError(f'Unsupported image data format: {format}')
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self._need_permute = format != 'NCHW'

    def forward(self, x):
        if self._need_permute:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        x = self.convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x


class ImpalaCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 dropout=0.0,
                 batch_norm=False,
                 impala_size='small',
                 format='NCHW'):
        # https://arxiv.org/pdf/1802.01561.pdf
        # https://github.com/openai/coinrun/blob/master/coinrun/policies.py#L9
        super().__init__()
        if format not in ['NCHW', 'NHWC']:
            raise TypeError(f'Unsupported image data format: {format}')
        if impala_size not in ['small', 'large']:
            raise ValueError(f'impala_size should be '
                             f'one of \'small\' or \'large\'! '
                             f'Got {impala_size} instead!')
        if impala_size == 'small':
            channel_groups = [16, 32, 32]
        else:
            channel_groups = [32, 64, 64, 64, 64]
        self.convs = nn.ModuleList()
        for ch in channel_groups:
            self.convs.append(
                ImpalaConvBlock(in_channels=in_channels,
                                out_channels=ch,
                                dropout=dropout,
                                batch_norm=batch_norm)
            )
            self.convs.append(
                nn.MaxPool2d(kernel_size=3,
                             stride=2,
                             padding=1)
            )
            self.convs.append(
                ImpalaResidualBlock(num_channels=ch,
                                    dropout=dropout,
                                    batch_norm=batch_norm)
            )
            self.convs.append(
                ImpalaResidualBlock(num_channels=ch,
                                    dropout=dropout,
                                    batch_norm=batch_norm)
            )
            in_channels = ch
        self.fcs = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU()
        )
        self._need_permute = format != 'NCHW'

    def forward(self, x):
        if self._need_permute:
            x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        for layer in self.convs:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x


class ImpalaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=0.0, batch_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1)
        )
        if dropout > 0.0:
            self.layers.append(
                nn.Dropout2d(p=dropout)
            )
        if batch_norm:
            self.layers.append(
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImpalaResidualBlock(nn.Module):
    def __init__(self, num_channels,
                 dropout=0.0, batch_norm=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            ImpalaConvBlock(in_channels=num_channels,
                            out_channels=num_channels,
                            dropout=dropout,
                            batch_norm=batch_norm),
            nn.ReLU(),
            ImpalaConvBlock(in_channels=num_channels,
                            out_channels=num_channels,
                            dropout=dropout,
                            batch_norm=batch_norm)
        )

    def forward(self, x):
        out = self.layers(x)
        new_out = out + x
        return new_out
