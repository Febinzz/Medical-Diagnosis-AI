import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            **kwargs
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj
    ):
        super().__init__()

        self.branch1 = ConvBlock(
            in_channels,
            num1x1,
            kernel_size=1
        )

        self.branch2 = nn.Sequential(
            ConvBlock(
                in_channels,
                num3x3_reduce,
                kernel_size=1
            ),
            ConvBlock(
                num3x3_reduce,
                num3x3,
                kernel_size=3,
                padding=1
            )
        )

        self.branch3 = nn.Sequential(
            ConvBlock(
                in_channels,
                num5x5_reduce,
                kernel_size=1
            ),
            ConvBlock(
                num5x5_reduce,
                num5x5,
                kernel_size=5,
                padding=2
            )
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvBlock(
                in_channels,
                pool_proj,
                kernel_size=1
            )
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat(
            [b1, b2, b3, b4],
            dim=1
        )


class Auxiliary(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AvgPool2d(
            kernel_size=5,
            stride=3
        )

        self.conv = ConvBlock(
            in_channels,
            128,
            kernel_size=1
        )

        self.fc1 = nn.Linear(
            2048,
            1024
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.7
        )

        self.fc2 = nn.Linear(
            1024,
            num_classes
        )

    def forward(self, x):

        x = self.avgpool(x)

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


class Inception(nn.Module):

    def __init__(
        self,
        in_channels=3,
        use_auxiliary=True,
        num_classes=3
    ):
        super().__init__()

        self.use_auxiliary = use_auxiliary

        self.conv1 = ConvBlock(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.conv2 = ConvBlock(
            64,
            192,
            kernel_size=3,
            padding=1
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.inception3a = InceptionBlock(
            192,64,96,128,16,32,32
        )

        self.inception3b = InceptionBlock(
            256,128,128,192,32,96,64
        )

        self.inception4a = InceptionBlock(
            480,192,96,208,16,48,64
        )

        self.inception4b = InceptionBlock(
            512,160,112,224,24,64,64
        )

        self.inception4c = InceptionBlock(
            512,128,128,256,24,64,64
        )

        self.inception4d = InceptionBlock(
            512,112,144,288,32,64,64
        )

        self.inception4e = InceptionBlock(
            528,256,160,320,32,128,128
        )

        self.inception5a = InceptionBlock(
            832,256,160,320,32,128,128
        )

        self.inception5b = InceptionBlock(
            832,384,192,384,48,128,128
        )

        if self.use_auxiliary:
            self.aux1 = Auxiliary(
                512,
                num_classes
            )

            self.aux2 = Auxiliary(
                528,
                num_classes
            )

        self.avgpool = nn.AvgPool2d(
            kernel_size=7,
            stride=1
        )

        self.dropout = nn.Dropout(
            0.4
        )

        self.fc = nn.Linear(
            1024,
            num_classes
        )

    def forward(self, x):

        aux1 = None
        aux2 = None

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.maxpool(x)

        x = self.inception4a(x)

        if self.training and self.use_auxiliary:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.training and self.use_auxiliary:
            aux2 = self.aux2(x)

        x = self.inception4e(x)

        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.fc(x)

        return x, aux1, aux2