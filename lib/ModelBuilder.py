import torch
import torch.nn as nn
import numpy as np
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

# def _get_padding(padding_type, kernel_size):
#     assert padding_type in ['SAME', 'VALID']
#     if padding_type == 'SAME':
#         return tuple((k - 1) // 2 for k in kernel_size))
#     # return tuple(0 for _ in kernel_size)

def get_padding(padding_type, kernel_size):
    assert padding_type in ['same', 'valid']
    if padding_type == 'same':
        return (kernel_size-1)//2
    return tuple(0 for _ in kernel_size)


def build_double_conv(inp_channels: int, output_channels: int, dropout_frac: float = 0.1):
    conv = nn.Sequential(
        nn.ZeroPad2d(get_padding('same', 3)),
        nn.Conv2d(inp_channels, output_channels, kernel_size=(3,3)),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_frac, inplace=True),
        nn.ZeroPad2d(get_padding('same', 3)),
        nn.Conv2d(output_channels, output_channels, kernel_size=(3,3)),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_tensor(orignal_tensor, target_tensor):
    orignal_size = orignal_tensor.size()[2]
    target_size = target_tensor.size()[2]
    delta = orignal_size - target_size
    delta = delta // 2
    return orignal_tensor[:, :, delta:orignal_size - delta, delta:orignal_size - delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.MaxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = build_double_conv(2, 64)
        self.down_conv_2 = build_double_conv(64, 128)
        self.down_conv_3 = build_double_conv(128, 256)
        self.down_conv_4 = build_double_conv(256, 512)
        self.down_conv_5 = build_double_conv(512, 1024)

        ## Second Part of the architecture
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(2,2),
            stride=2)
        self.up_conv_1 = build_double_conv(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(2,2),
            stride=2)
        self.up_conv_2 = build_double_conv(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2,2),
            stride=2)
        self.up_conv_3 = build_double_conv(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2,2),
            stride=2)
        self.up_conv_4 = build_double_conv(128, 64)

        ## Output
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, image):
        ## Encoder
        x1 = self.down_conv_1(image)  ## Skip connection
        print(x1.shape)
        x2 = self.MaxPool_2x2(x1)
        print(x2.shape)

        x3 = self.down_conv_2(x2)  ## Skip connection
        print(x3.shape)
        x4 = self.MaxPool_2x2(x3)
        print(x4.shape)

        x5 = self.down_conv_3(x4)  ## Skip connection
        print(x5.shape)
        x6 = self.MaxPool_2x2(x5)
        print(x6.shape)

        x7 = self.down_conv_4(x6)  ## Skip connection
        print(x7.shape)
        x8 = self.MaxPool_2x2(x7)
        print(x8.shape)

        x9 = self.down_conv_5(x8)
        print(x9.shape)

        ## Decoder
        x = self.up_transpose_1(x9)
        print(x.shape)
        y = crop_tensor(x7, x)
        print(y.shape)
        x = self.up_conv_1(torch.cat([x, y], axis=1))
        print(x.shape)

        x = self.up_transpose_2(x)
        print(x.shape)
        y = crop_tensor(x5, x)
        print(y.shape)
        x = self.up_conv_2(torch.cat([x, y], axis=1))
        print(x.shape)

        x = self.up_transpose_3(x)
        print(x.shape)
        y = crop_tensor(x3, x)
        print(y.shape)
        x = self.up_conv_3(torch.cat([x, y], axis=1))
        print(x.shape)

        x = self.up_transpose_4(x)
        print(x.shape)
        y = crop_tensor(x1, x)
        print(y.shape)
        x = self.up_conv_4(torch.cat([x, y], axis=1))
        print(x.shape)

        ## Output
        x = self.output(x)
        print(x.shape)
        return x

if __name__ == "__main__":
    random_image = torch.randn((10, 2, 256, 256))
    model = UNet()
    model.forward(random_image)