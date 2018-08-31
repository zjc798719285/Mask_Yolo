import torch as th
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo





model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x, x1, x2, x3, x4


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2)
        )
    def forward(self, input):
        x = self.conv(input)
        x = nn.ReLU6()(x + input)
        return x


class conv_1x1(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x










class MultiResolutionFusion(nn.Module):
    def __init__(self, low_ch, high_ch):
        super(MultiResolutionFusion, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(nn.Conv2d(low_ch + high_ch, high_ch, kernel_size=3, dilation=2, stride=1, padding=2),
                                  nn.BatchNorm2d(high_ch),
                                  nn.ReLU6(),
                                  nn.Conv2d(high_ch, high_ch, kernel_size=3, dilation=2, stride=1, padding=2),
                                  nn.BatchNorm2d(high_ch),
                                  nn.ReLU6()
                                  )

    def forward(self, low_x, high_x):
        low_x = self.upsample(low_x)
        x = th.cat((low_x, high_x), 1)
        x = self.conv(x)
        return x






class up(nn.Module):
    def __init__(self, low_ch, high_ch):
        super(up, self).__init__()
        self.conv_low_1x1 = conv_1x1(low_ch, high_ch)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_high = double_conv(high_ch)
        self.conv_cat = double_conv(high_ch)


    def forward(self, low_x, high_x):
        low_x = self.conv_low_1x1(low_x)
        global_x = nn.Sigmoid()(th.mean(th.mean(low_x, 3, keepdim=True), 2, keepdim=True))
        low_x = self.upsample(low_x)
        high_x = self.conv_high(high_x)
        high_x = high_x * global_x
        cat_x = low_x + high_x
        cat_x = self.conv_cat(cat_x)
        return cat_x


# class up(nn.Module):
#     def __init__(self, low_ch, high_ch):
#         super(up, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(high_ch, low_ch, kernel_size=3, dilation=2, stride=2, padding=2),
#                                    nn.BatchNorm2d(low_ch),
#                                    nn.ReLU6())
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.conv2 = nn.Sequential(nn.Conv2d(low_ch, high_ch, kernel_size=3, dilation=2, stride=1, padding=2),
#                                   nn.BatchNorm2d(high_ch),
#                                   nn.ReLU6(),
#                                   nn.Conv2d(high_ch, high_ch, kernel_size=3, dilation=2, stride=1, padding=2),
#                                   nn.BatchNorm2d(high_ch),
#                                   nn.ReLU6()
#                                   )
#     def forward(self, low_x, high_x):
#         high_x = self.conv1(high_x)
#         x = low_x + high_x
#         x = self.upsample(x)
#         x = self.conv2(x)
#         return x




class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 1),
             nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class locconv(nn.Module):
    def __init__(self, in_ch):
        super(locconv, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_ch, in_ch, 3, bias=False, padding=1),
             nn.Tanh())
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_ch, 4, 1, bias=False),
             nn.ReLU6()
        )

    def forward(self, x):
        x = self.conv1(x)
        box = self.conv2(x)
        return box



class confconv(nn.Module):
    def __init__(self, in_ch):
        super(confconv, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(in_ch, in_ch, 3, bias=False, padding=1),
             nn.Tanh(),
             nn.Conv2d(in_ch, 1, 1, bias=False),
             nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x