import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 定义基本残差块
class SL_Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, downsample=None):
        super(SL_Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = F.relu(out)
        return out

# 定义完整ResNet18（仅用于建模+切块）
class ResNet18_Full(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet18_Full, self).__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()
        self.blocks += self._make_layer(block, 64, layers[0], stride=1)
        self.blocks += self._make_layer(block, 128, layers[1], stride=2)
        self.blocks += self._make_layer(block, 256, layers[2], stride=2)
        self.blocks += self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet18_Client_Cut(nn.Module):
    def __init__(self, cut_layer, block, layers):
        super(ResNet18_Client_Cut, self).__init__()
        self.inplanes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        all_blocks = []
        all_blocks += self._make_layer(block, 64, layers[0], stride=1)
        all_blocks += self._make_layer(block, 128, layers[1], stride=2)
        all_blocks += self._make_layer(block, 256, layers[2], stride=2)
        all_blocks += self._make_layer(block, 512, layers[3], stride=2)

        self.blocks = nn.Sequential(*all_blocks[:cut_layer])

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

class ResNet18_Server_Cut(nn.Module):
    def __init__(self, cut_layer, block, layers, num_classes=10):
        super(ResNet18_Server_Cut, self).__init__()
        self.inplanes = 64

        all_blocks = []
        all_blocks += self._make_layer(block, 64, layers[0], stride=1)
        all_blocks += self._make_layer(block, 128, layers[1], stride=2)
        all_blocks += self._make_layer(block, 256, layers[2], stride=2)
        all_blocks += self._make_layer(block, 512, layers[3], stride=2)

        self.blocks = nn.Sequential(*all_blocks[cut_layer:])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 构建函数
def resnet18_full():
    return ResNet18_Full(SL_Baseblock, [2, 2, 2, 2], num_classes=10)

def resnet18_client(cut_layer):
    return ResNet18_Client_Cut(cut_layer, SL_Baseblock, [2, 2, 2, 2])

def resnet18_server(cut_layer):
    return ResNet18_Server_Cut(cut_layer, SL_Baseblock, [2, 2, 2, 2], num_classes=10)

# 提取 state_dict 工具函数
def extract_client_state_dict(full_state_dict, cut_layer):
    new_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            idx = int(parts[1])
            if idx < cut_layer:
                parts[1] = str(idx)
                new_k = ".".join(parts)
                new_state_dict[new_k] = v
        elif k.startswith("stem.") or "stem" in k:
            new_state_dict[k] = v
    return new_state_dict

def extract_server_state_dict(full_state_dict, cut_layer):
    new_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("blocks."):
            parts = k.split(".")
            idx = int(parts[1])
            if idx >= cut_layer:
                parts[1] = str(idx - cut_layer)
                new_k = ".".join(parts)
                new_state_dict[new_k] = v
        elif k.startswith("avgpool") or k.startswith("fc"):
            new_state_dict[k] = v
    return new_state_dict


# 根据完整模型和切割层，生成边缘端（Client）和服务器端（Server）的模型及其state_dict
def generate_split_models(full_model, cut_layer):
    client_model = resnet18_client(cut_layer)
    server_model = resnet18_server(cut_layer)

    full_state_dict = full_model.state_dict()

    client_state_dict = extract_client_state_dict(full_state_dict, cut_layer)
    server_state_dict = extract_server_state_dict(full_state_dict, cut_layer)

    client_model.load_state_dict(client_state_dict, strict=False)
    server_model.load_state_dict(server_state_dict, strict=False)

    return client_model, server_model
