import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out
    
class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.upsample1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor=2)
        self.upsample2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor=2)

        self.conv2 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.res_blocks(out)
        out = self.pixel_shuffle1(self.upsample1(out))
        out = self.pixel_shuffle2(self.upsample2(out))
        out = self.conv2(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        # Điều chỉnh kích thước đầu vào của lớp FC
        self.fc1 = nn.Linear(256 * 70 * 60, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = x.view(x.size(0), -1)  # Flatten tensor
        x = self.lrelu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x