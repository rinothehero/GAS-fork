import torch
import torch.nn as nn
import numpy as np


def model_selection(cifar=False, mnist=False, fmnist=False, cinic=False, cifar100=False, SVHN=False, split=False, twoLogit=False, resnet=False):
    num_classes = 10
    if cifar100:
        num_classes = 100
    if split:
        server_local_model = None
        if cifar or cinic or cifar100 or SVHN:
            if resnet:
                # Use ResNet-18 for Split Learning
                user_model = ResNet18DownCifar()
                server_model = ResNet18UpCifar(num_classes=num_classes)
                if twoLogit:
                    server_local_model = ResNet18UpCifar(num_classes=num_classes)
            else:
                # Use AlexNet for Split Learning (default)
                user_model = AlexNetDownCifar()
                server_model = AlexNetUpCifar(num_classes=num_classes)
                if twoLogit:
                    server_local_model = AlexNetUpCifar(num_classes=num_classes)
        elif mnist or fmnist:
            user_model = AlexNetDown()
            server_model = AlexNetUp()
            if twoLogit:
                server_local_model = AlexNetUp()
        else:
            user_model = None
            server_model = None
        if twoLogit:
            return user_model, server_model, server_local_model
        else:
            return user_model, server_model
    else:
        if cifar or cinic or cifar100 or SVHN:
            if resnet:
                # Use complete ResNet-18 (not split)
                model = ResNet18Cifar(num_classes=num_classes)
            else:
                # Use complete AlexNet (default)
                model = AlexNetCifar(num_classes=num_classes)
        elif mnist or fmnist:
            model = AlexNet()
        else:
            model = None

        return model

def inversion_model(feature_size=None):
    model = custom_AE(input_nc=64, output_nc=3, input_dim=8, output_dim=32)
    return model


class custom_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            #TODO: change to Conv2d
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class AlexNetCifar(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetCifar, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256*3*3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x, return_features=False):
        out = self.conv(x)
        features = out.view(-1, 256*3*3)
        out = self.fc(features)
        if return_features:
            return out, features
        else:
            return out

class AlexNetDownCifar(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDownCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class AlexNetUpCifar(nn.Module):
    def __init__(self, width_mult=1, num_classes=10):
        super(AlexNetUpCifar, self).__init__()
        self.model2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256*3*3)
        x = self.classifier(x)
        return x

class AlexNetDown(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class AlexNetUp(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNetUp, self).__init__()
        self.model2 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.model2(x)
        x = x.view(-1, 256*3*3)
        x = self.classifier(x)
        return x


# ============== ResNet for CIFAR-10 ==============

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Cifar(nn.Module):
    """ResNet-18 for CIFAR-10 (complete model, not split)"""
    def __init__(self, num_classes=10):
        super(ResNet18Cifar, self).__init__()
        self.in_channels = 64

        # Initial convolution (smaller for CIFAR-10's 32x32 images)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet18DownCifar(nn.Module):
    """Client-side ResNet-18 for CIFAR-10 (split after layer2)"""
    def __init__(self):
        super(ResNet18DownCifar, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # First two ResNet layers (client side)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: [B, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # After conv1: [B, 64, 32, 32]

        x = self.layer1(x)
        # After layer1: [B, 64, 32, 32]

        x = self.layer2(x)
        # After layer2: [B, 128, 16, 16]
        # This is the activation sent to server

        return x


class ResNet18UpCifar(nn.Module):
    """Server-side ResNet-18 for CIFAR-10 (receives layer2 output)"""
    def __init__(self, num_classes=10):
        super(ResNet18UpCifar, self).__init__()
        self.in_channels = 128

        # Remaining ResNet layers (server side)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: [B, 128, 16, 16] from client

        x = self.layer3(x)
        # After layer3: [B, 256, 8, 8]

        x = self.layer4(x)
        # After layer4: [B, 512, 4, 4]

        x = self.avgpool(x)
        # After avgpool: [B, 512, 1, 1]

        x = torch.flatten(x, 1)
        # After flatten: [B, 512]

        x = self.fc(x)
        # Output: [B, num_classes]

        return x
