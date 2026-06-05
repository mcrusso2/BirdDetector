
import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    # Expansion factor for output channels (1 for BasicBlock)
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # Second 3x3 convolution
        # TODO: Define the second 3x3 convolutional layer (conv2) and its batch normalization layer (bn2)
        # Hint: Both input and output channels are `out_channel`. Set stride to 1, padding to 1, and bias to False.
        # self.conv2 =  # [YOUR CODE HERE]
        # self.bn2 =  # [YOUR CODE HERE]
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # Downsample layer for shortcut connection (if dimensions mismatch)
        self.downsample = downsample

    def forward(self, x):
        # The input x typically has the shape [N, C_in, H_in, W_in]
        # N: Batch Size, C: Channels, H: Height, W: Width
        identity = x

        if self.downsample is not None:
            # If downsampling is applied (e.g., stride=2), spatial dimensions are halved and channel depth changes.
            # Shape change: [N, C_in, H_in, W_in] -> [N, C_out, H_out, W_out]
            identity = self.downsample(x)

        # Passes through the first convolutional layer, batch normalization, and ReLU activation.
        # Shape change: [N, C_in, H_in, W_in] -> [N, C_out, H_out, W_out]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Passes through the second convolutional layer and batch normalization.
        # The stride here is always 1, so the spatial dimensions remain unchanged.
        # Shape remains: [N, C_out, H_out, W_out]
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection (Residual addition)
        # The shapes of `out` and `identity` must be exactly identical for element-wise addition.
        # Shape remains: [N, C_out, H_out, W_out] + [N, C_out, H_out, W_out]
        out += identity

        # Final activation
        out = self.relu(out)

        # Return the final output
        # Final Shape: [N, C_out, H_out, W_out]
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # Initial input channels

        self.groups = groups
        self.width_per_group = width_per_group

        # Stage 0: Initial 7x7 convolution and max pooling
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 1-4: Residual block sequences
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # TODO:
        # Complete the definition of layer3.
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # Stage 5: Classification head
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize network weights (Kaiming normal for ReLU networks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # Configure downsample 1x1 conv if spatial size or channel count changes
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # First block in the layer (handles downsampling if needed)
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion

        # Remaining blocks in the layer
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial input shape: [N, 3, H, W]
        # (e.g., [N, 3, 224, 224] for a standard ImageNet input)

        # Stage 0: Stem (Initial feature extraction and aggressive downsampling)
        x = self.conv1(x)  # Shape: [N, 64, H/2, W/2] (Spatial dimensions halved due to stride=2)
        x = self.bn1(x)  # Shape remains: [N, 64, H/2, W/2]
        x = self.relu(x)  # Shape remains: [N, 64, H/2, W/2]
        x = self.maxpool(x)  # Shape: [N, 64, H/4, W/4] (Spatial dimensions halved again due to stride=2)

        # Stages 1-4: Residual Body (Feature processing)
        # As we go deeper, spatial dimensions are halved and channel depth is doubled.
        x = self.layer1(x)  # Shape remains: [N, 64, H/4, W/4]  (No downsampling in layer1)
        x = self.layer2(x)  # Shape: [N, 128, H/8, W/8]         (Downsampled, channels doubled)
        x = self.layer3(x)  # Shape: [N, 256, H/16, W/16]       (Downsampled, channels doubled)
        x = self.layer4(x)  # Shape: [N, 512, H/32, W/32]       (Downsampled, channels doubled)
        # E.g., for a 224x224 input, shape here is [N, 512, 7, 7]

        # Stage 5: Classification Head
        if self.include_top:
            x = self.avgpool(x)  # Global Average Pooling compresses spatial dimensions to 1x1
            # Shape: [N, 512, 1, 1]

            x = torch.flatten(x, 1)  # Flatten (B, C, 1, 1) -> (B, C)  (Note: B and N both represent Batch Size)
            # Shape: [N, 512]

            x = self.fc(x)  # Fully Connected Layer maps features to output classes
            # Final Shape: [N, num_classes]

        return x

def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-5c106cde.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)




#
#
# class Bottleneck(nn.Module):
#     """
#     Note: In the original paper, stride=2 is applied to the first 1x1 conv.
#     In the PyTorch official implementation, stride=2 is applied to the 3x3 conv.
#     This provides roughly a 0.5% top-1 accuracy boost (ResNet v1.5).
#     """
#     # Expansion factor for output channels (4 for Bottleneck)
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None,
#                  groups=1, width_per_group=64):
#         super(Bottleneck, self).__init__()
#
#         width = int(out_channel * (width_per_group / 64.)) * groups
#
#         # 1x1 conv: squeeze channels
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width)
#
#         # 3x3 conv: process spatial information
#         self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = nn.BatchNorm2d(width)
#
#         # 1x1 conv: unsqueeze channels back to out_channel * expansion
#         self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         # Skip connection (Residual addition)
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# def resnet50(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet50-19c8e357.pth
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnet101(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return ResNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
#
#
# def resnext101_32x8d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
#     groups = 32
#     width_per_group = 8
#     return ResNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
