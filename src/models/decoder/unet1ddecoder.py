# ref: https://github.com/bamps53/kaggle-dfl-3rd-place-solution/blob/master/models/cnn_3d.py
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
            '''
            注意力机制通常用于增强或弱化模型某些部分的影响力，以提高其性能和精确度。
            
            SEModule正是通过重调节不同特征通道的重要性来实现这一点。
            1 压缩（Squeeze）：
            首先，输入的特征图通过全局平均池化来进行压缩。
            这意味着每个通道的特征图将被缩减为一个单独的数值，这个数值代表了该通道整体的平均激活水平。
            例如，如果输入特征图的大小是 C×H×W（其中 C 是通道数，H 和 W 分别是高度和宽度），
            则全局平均池化后的输出大小是 C×1×1。
            2 激励（Excitation）：
            全局平均池化的输出接着被送入一个小型的全连接网络。
            这个网络通常包含两个全连接层，第一个层会减少通道数（通常是输入通道数的一小部分，比如 C/r，其中 r 是缩减比例），
            接着是一个激活函数（如 ReLU）。
            然后第二个全连接层会将通道数从缩减后的大小恢复到原始通道数 C，
            通常接一个 Sigmoid 激活函数，以产生范围在 0 到 1 之间的权重。
            3 重新加权（Re-Calibration）：
            这个小型全连接网络输出的权重（每个通道一个权重）用于调整原始输入特征图的每个通道。
            这个过程可以看作是对每个通道的重要性进行重新加权。
            具体而言，每个通道的特征图乘以对应的权重（通过 Sigmoid 激活的输出），从而增强重要的特征通道，抑制不那么重要的通道。
            '''
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity,
        )
        '''
        mid_channels作用
        1 逐步改变特征维度：通过逐步改变特征图的通道数，网络能够更加平滑地从输入特征到输出特征进行转换。
        这可以帮助网络更有效地学习从输入到输出所需的特征变换。
        2 控制模型复杂度：通过调整 mid_channels，我们可以控制网络层的复杂度和参数数量。
        例如，如果直接从一个很大的通道数减少到一个很小的通道数，这可能需要非常大的卷积核（即有很多参数），这可能会导致过拟合。
        通过引入中间通道数，可以减少每个卷积层的参数数量。
        3 增加非线性：两次连续的卷积操作（通常伴随着激活函数）能够增加网络的非线性能力。这意味着网络可以捕获更复杂的特征和模式。
        '''

    def forward(self, x):
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d, se=False, res=False
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


'''
bilinear 参数决定了上采样的方法。
上采样是将特征图尺寸放大的过程，通常用于解码器部分以重建更高分辨率的输出。
bilinear=True：双线性插值
bilinear=False：转置卷积

双线性插值是一种插值方法，它在两个轴上进行线性插值，以生成新的像素或特征值。
双线性插值通常在计算上更高效，但可能不如转置卷积灵活，因为它不涉及学习参数。
转置卷积是一种特殊的卷积操作，它可以增加特征图的空间尺寸（高度和宽度）。
转置卷积可以学习如何最佳地上采样，但如果不当使用，可能会引入伪影（如棋盘效应）。
'''
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
            # 由于双线性插值不会改变通道数，因此需要额外的卷积操作（self.conv）来调整通道数。
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


'''
批标准化 (BatchNorm):
应用方式：批标准化对小批量（batch）中的数据进行标准化。它计算整个小批量数据在某一特定特征维度上的均值和标准差。
优点：在训练时可以提高收敛速度，使网络对初始权重的选择更加鲁棒。
缺点：对小批量的大小较为敏感。当小批量很小或者是动态变化的（例如在序列模型中），其效果可能会受到影响。
常用场景：在卷积神经网络（CNN）和某些全连接层中效果较好。

层标准化 (LayerNorm):
应用方式：层标准化对单个数据点的所有特征进行标准化。它在特征维度上计算单个数据点的均值和标准差。
优点：不依赖于小批量的大小，因此适用于小批量大小不一或者动态变化的情况，如循环神经网络（RNN）中。
缺点：在某些情况下，它可能不如批标准化有效，特别是在批量较大且稳定时。
常用场景：在循环神经网络（RNN）和Transformer模型中效果较好。
'''
def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])


class UNet1DDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = True,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration // 2)
        )
        #最大池化层使用 scale_factor 来缩小特征图的尺寸        
        '''
        每个 Down 层的 length 减少一半的原因是由于最大池化操作。池化操作后的特征图长度（即时间步的数量）减少到原始长度的一半。

        Down 层在 U-Net 架构或其他类似的编码器-解码器网络中充当编码器的一部分。
        在这些网络中，编码器部分负责逐渐提取输入数据的特征，同时减少其空间维度。这通常是通过一系列卷积和池化操作实现的，
        其中 Down 层正是这个过程的关键组成部分。
        在编码器中，每个 Down 层通常会执行以下操作：
        1 应用池化：通过池化操作（如最大池化），减小数据的空间维度（例如，减少时间步长、缩小图像尺寸等）。
        这有助于网络抽象和概括数据特征，同时减少计算量和内存需求。
        2 增加特征通道数：随着网络向更深层次发展，增加每层的通道数（即深度）。
        这样做可以增加网络的容量，使其能够学习和表示更复杂的特征。
        Down 层在编码器中起到了关键的作用，它们帮助网络逐步理解和抽象输入数据的特征，为后续的解码过程（即数据重建或特定任务的输出）打下基础。
        '''
        self.down2 = Down(
            128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration // 4)
        )
        self.down3 = Down(
            256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration // 8)
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration)
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """

        # 1D U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # classifier
        logits = self.cls(x)  # (batch_size, n_classes, n_timesteps)
        return logits.transpose(1, 2)  # (batch_size, n_timesteps, n_classes)
