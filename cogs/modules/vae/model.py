import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3trans(in_planes, out_planes, stride=1, groups=1, dilation=1, output_padding=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, output_padding=output_padding, bias=False)


def conv1x1trans(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.elu = nn.ELU(inplace=True) #elu
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.elu(out)


        return out

class SEBasicBlockTrans(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, output_padding=1, norm_layer=None, r=16):
        super(SEBasicBlockTrans, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.elu = nn.ELU(inplace=True) #elu
        if stride != 1:
            self.conv2 = conv3x3trans(inplanes, planes, stride, dilation=dilation, output_padding=output_padding)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # add SE operation
        out = self.se(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.elu(out)


        return out

class vEncoderSE(nn.Module):
    def __init__(self, *, ch1=64, ch2=16, ch3=4, out_ch=1024,
                 in_channels, patch_size, **ignore_kwargs):
        super().__init__()
        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.norm_layer = nn.BatchNorm2d

        self.downsample1 = nn.Sequential(
            conv1x1(in_channels, self.ch1, 1),
            self.norm_layer(self.ch1),
        )

        self.downsample2 = nn.Sequential(
            conv1x1(self.ch1, self.ch2, 1),
            self.norm_layer(self.ch2),
        )

        self.downsample3 = nn.Sequential(
            conv1x1(self.ch2, self.ch3, 1),
            self.norm_layer(self.ch3),
        )

        self.layer1 = SEBasicBlock(in_channels, self.ch1, downsample=self.downsample1)
        self.layer2 = SEBasicBlock(self.ch1, self.ch2, downsample=self.downsample2)
        self.layer3 = SEBasicBlock(self.ch2, self.ch3, r=1, downsample=self.downsample3)

        self.fl = nn.Flatten()

        self.mean_branch = nn.Linear(self.ch3 * self.patch_size * self.patch_size, self.out_ch)
        self.stddev_branch = nn.Linear(self.ch3 * self.patch_size * self.patch_size, self.out_ch)


    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h3 = self.fl(h3)

        h_mean = self.mean_branch(h3)
        h_std = self.stddev_branch(h3)

        return h_mean, h_std

class DecoderSE(nn.Module):
    def __init__(self, *, ch1=64, ch2=16, ch3=4, out_ch=1024,
                 in_channels, patch_size, **ignore_kwargs):
        super().__init__()
        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.norm_layer = nn.BatchNorm2d

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(self.ch3, self.ch2, kernel_size=1,
                               stride=1, bias=False),
            self.norm_layer(self.ch2),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(self.ch2, self.ch1, kernel_size=1,
                               stride=1, bias=False),
            self.norm_layer(self.ch1),
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(self.ch1, in_channels, kernel_size=1,
                               stride=1, bias=False),
            self.norm_layer(in_channels),
        )

        self.layer1 = SEBasicBlockTrans(self.ch3, self.ch2, stride=1, upsample=self.upsample1)
        self.layer2 = SEBasicBlockTrans(self.ch2, self.ch1, stride=1, upsample=self.upsample2)
        self.layer3 = SEBasicBlockTrans(self.ch1, in_channels, stride=1, upsample=self.upsample3)
        self.fc2 = nn.Linear(self.ch3 * self.patch_size * self.patch_size + self.out_ch,
                             self.ch3 * self.patch_size * self.patch_size)

        self.unfl = nn.Unflatten(1, (self.ch3, self.patch_size, self.patch_size))

    def forward(self, x):
        print(x.shape)
        h = self.fc2(x)
        h = self.unfl(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)

        return h
