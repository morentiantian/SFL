from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_num_groups(num_channels, default_groups=8):
    """
    一个辅助函数，为GroupNorm找到一个有效的组数。
    它会尝试用默认组数，如果不行，则寻找一个能被通道数整除的最大可能组数。
    """
    if num_channels == 0: return 1
    if num_channels % default_groups == 0:
        return default_groups
    for i in range(min(num_channels, default_groups), 0, -1):
        if num_channels % i == 0:
            return i
    return 1

# ======= Basic & Bottleneck Blocks (FINAL FIX APPLIED) =======
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 最终修复: 将BatchNorm替换为线程安全的GroupNorm
        self.bn1 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(get_num_groups(planes), planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_num_groups(planes), planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 正确: 使用非原地加法进行残差连接
        out = out + self.shortcut(x) 
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(get_num_groups(planes * self.expansion), planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_num_groups(planes * self.expansion), planes * self.expansion)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)

# ======= Main ResNet Class (FINAL FIX APPLIED) =======
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channels, output_channels):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(get_num_groups(64), 64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, output_channels)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ======= Helper Conv Block (FINAL FIX APPLIED) =======
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(get_num_groups(out_channels), out_channels), nn.ReLU(inplace=False)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ======= ResNet9 (CORRECTED & FINAL) =======
class ResNet9(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block1 = conv_block(input_channels, 64)
        self.block2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.block3 = conv_block(128, 256, pool=True)
        self.block4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, output_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        # 正确: 使用显式的残差计算来避免歧义
        residual = x
        x = self.res1(x)
        x = x + residual

        x = self.block3(x)
        x = self.block4(x)
        
        residual = x
        x = self.res2(x)
        x = x + residual
        
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# ========== Full Model Constructor (No change) ==========
def get_model(model_name, input_channels, output_channels):
    model_name = model_name.lower()
    if model_name == "resnet9":
        return ResNet9(input_channels, output_channels)
    elif model_name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], input_channels, output_channels)
    elif model_name == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], input_channels, output_channels)
    elif model_name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], input_channels, output_channels)
    elif model_name == "resnet101":
        return ResNet(Bottleneck, [3, 4, 23, 3], input_channels, output_channels)
    elif model_name == "resnet152":
        return ResNet(Bottleneck, [3, 8, 36, 3], input_channels, output_channels)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


# ========== Split Model Constructor (No change) ==========
def _recursive_generate_maps(module, key_map, prefix=""):
    if not list(module.children()):
        for name, _ in module.named_parameters(recurse=False):
            key_map[prefix + name] = prefix + name
        for name, _ in module.named_buffers(recurse=False):
            key_map[prefix + name] = prefix + name
        return

    for name, child in module.named_children():
        _recursive_generate_maps(child, key_map, prefix=f"{prefix}{name}.")

def split_model(model, model_name, cut_layer):
    client_key_map = OrderedDict()
    server_key_map = OrderedDict()
    model_name = model_name.lower()
    
    top_level_modules = list(model.named_children())

    if not (0 < cut_layer <= len(top_level_modules)):
        raise ValueError(f"cut_layer must be in (0, {len(top_level_modules)}], but got {cut_layer}")

    client_list = top_level_modules[:cut_layer]
    server_list = top_level_modules[cut_layer:]
    
    client_model = nn.Sequential(OrderedDict(client_list))
    server_model = nn.Sequential(OrderedDict(server_list))
    
    for name, module in client_list:
        _recursive_generate_maps(module, client_key_map, prefix=f"{name}.")
        
    for name, module in server_list:
        _recursive_generate_maps(module, server_key_map, prefix=f"{name}.")

    return client_model, server_model, client_key_map, server_key_map
