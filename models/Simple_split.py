# filename: Simple_split.py
# description: 一个完整的文件，包含了对VGG16, CifarCNN, 以及新增的ResNet18的支持和分割逻辑。

from collections import OrderedDict
import logging
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================
#                 1. 辅助函数 (所有模型共用)
# =================================================================

def get_num_groups(num_channels: int, default_groups: int = 8) -> int:
    """
    一个辅助函数，为GroupNorm找到一个有效的组数。
    """
    if num_channels == 0: return 1
    if num_channels % default_groups == 0:
        return default_groups
    for i in range(min(num_channels, default_groups), 0, -1):
        if num_channels % i == 0:
            return i
    return 1

# =================================================================
#                 2. 模块化的模型定义
# =================================================================

class VGG16(nn.Module):
    def __init__(self, input_channels=3, output_channels=100):
        super(VGG16, self).__init__()
        vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.block0 = self._make_block(vgg16_config[0:3], input_channels)
        self.block1 = self._make_block(vgg16_config[3:6], 64)
        self.block2 = self._make_block(vgg16_config[6:10], 128)
        self.block3 = self._make_block(vgg16_config[10:14], 256)
        self.block4 = self._make_block(vgg16_config[14:18], 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, output_channels),
        )

    def forward(self, x):
        x = self.block0(x); x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x); x = self.avgpool(x)
        x = self.flatten(x); x = self.classifier(x)
        return x

    def _make_block(self, cfg_block, in_channels):
        layers = []
        for x in cfg_block:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(get_num_groups(x), x),
                    nn.ReLU(inplace=True)
                ])
                in_channels = x
        return nn.Sequential(*layers)


class CifarCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CifarCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.GroupNorm(get_num_groups(6), 6),
            nn.ReLU(inplace=False)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.GroupNorm(get_num_groups(16), 16),
            nn.ReLU(inplace=False)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc_block1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.GroupNorm(get_num_groups(120), 120),
            nn.ReLU(inplace=False)
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.GroupNorm(get_num_groups(84), 84),
            nn.ReLU(inplace=False)
        )
        self.fc_block3 = nn.Linear(84, output_channels)

    def forward(self, x):
        x = self.conv_block1(x); x = self.pool1(x); x = self.conv_block2(x)
        x = self.pool2(x); x = self.flatten(x); x = self.fc_block1(x)
        x = self.fc_block2(x); x = self.fc_block3(x)
        return x

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                ResNet-18 完整实现
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BasicBlock(nn.Module):
    """ResNet-18/34 的基础残差块"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(get_num_groups(planes), planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_num_groups(self.expansion * planes), self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 100):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Block 0: 初始卷积层 (适配CIFAR-100的小尺寸图像)
        self.block0_conv_in = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(64), 64),
            nn.ReLU(inplace=True)
        )
        # Block 1-4: 四个主要的残差层
        self.block1_layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.block2_layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.block3_layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.block4_layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        # Block 5-7: 最终的池化和分类层
        self.block5_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.block6_flatten = nn.Flatten()
        self.block7_fc = nn.Linear(512 * BasicBlock.expansion, output_channels)

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block0_conv_in(x)
        x = self.block1_layer1(x)
        x = self.block2_layer2(x)
        x = self.block3_layer3(x)
        x = self.block4_layer4(x)
        x = self.block5_pool(x)
        x = self.block6_flatten(x)
        x = self.block7_fc(x)
        return x

# =================================================================
#                 3. 统一的接口函数 (已更新以包含ResNet18)
# =================================================================
# 根据模型名称，智能选择并返回正确的模型实例
def get_unified_model_constructor(model_name: str, input_channels: int, output_channels: int) -> nn.Module:
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'cifarcnn':
        return CifarCNN(input_channels, output_channels)
    elif model_name_lower == 'vgg16':
        return VGG16(input_channels, output_channels)
    elif model_name_lower == 'resnet18':
        return ResNet18(input_channels, output_channels)
        
    raise ValueError(f"不支持的模型: '{model_name}'.")

# 根据模型名称，返回其有效分割点的数量
def get_unified_num_split_options(model_name: str, logger: logging.Logger = None) -> int:
    model_name_lower = model_name.lower()
    
    if model_name_lower in ['cifarcnn', 'vgg16']:
        # CifarCNN 和 VGG16 都有8个顶层模块, 7个分割点
        return 7
    elif model_name_lower == 'resnet18':
        # ResNet18 有8个顶层模块 (block0 to block7), 7个分割点
        return 7
        
    raise ValueError(f"未知模型，无法确定分割点数量: {model_name}")

# 使用统一的、稳健的逻辑来分割任何已定义的模块化模型
def get_unified_split_model_function(full_model: nn.Module, model_name: str, split_point: int) -> Tuple[nn.Module, nn.Module, Dict, Dict]:
    top_level_modules = list(full_model.named_children())
    num_modules = len(top_level_modules)
    
    if not (0 < split_point <= num_modules):
        raise ValueError(f"对于 {model_name} 的分割点必须在 1 到 {num_modules} 之间, 但收到的是 {split_point}")
    
    client_module_list = top_level_modules[:split_point]
    server_module_list = top_level_modules[split_point:]
    
    client_model = nn.Sequential(OrderedDict(client_module_list))
    server_model = nn.Sequential(OrderedDict(server_module_list))
    
    client_key_map, server_key_map = {}, {}
    for key in full_model.state_dict().keys():
        top_level_name = key.split('.')[0]
        if top_level_name in dict(client_module_list):
            client_key_map[key] = key
        elif top_level_name in dict(server_module_list):
            server_key_map[key] = key
            
    return client_model, server_model, client_key_map, server_key_map

# 为所有可能的分割点计算归一化后的结构特征
def get_model_structural_features_for_all_splits(model_name: str, example_input_tensor: torch.Tensor, output_channels: int, logger: logging.Logger) -> Dict:
    num_splits = get_unified_num_split_options(model_name, logger)
    input_channels = example_input_tensor.shape[1]
    all_features = []

    try:
        full_model_template = get_unified_model_constructor(model_name, input_channels, output_channels)
        total_params = sum(p.numel() for p in full_model_template.parameters() if p.requires_grad)
    except Exception as e:
        logger.error(f"无法创建或分析模型 '{model_name}': {e}", exc_info=True)
        return {"features_vector": np.zeros(num_splits * 2, dtype=np.float32)}

    smashed_data_sizes = []
    param_ratios = []

    for sp in range(1, num_splits + 1):
        try:
            client_part, _, _, _ = get_unified_split_model_function(full_model_template, model_name, sp)
            
            with torch.no_grad():
                smashed_data = client_part(example_input_tensor)
            
            num_params_client = sum(p.numel() for p in client_part.parameters() if p.requires_grad)
            
            smashed_data_sizes.append(smashed_data.numel())
            param_ratios.append(num_params_client / total_params if total_params > 0 else 0)

        except Exception as e:
            logger.error(f"获取 {model_name} 在分割点 {sp} 的特征失败: {e}", exc_info=True)
            smashed_data_sizes.append(0)
            param_ratios.append(0)
    
    # 对 smashed_data_size 进行 z-score 标准化
    smashed_sizes_arr = np.array(smashed_data_sizes, dtype=np.float32)
    mean_size = np.mean(smashed_sizes_arr)
    std_size = np.std(smashed_sizes_arr)
    normalized_smashed_sizes = (smashed_sizes_arr - mean_size) / (std_size + 1e-9)
    
    # 将参数比率和标准化的数据大小交错合并
    features_vector = np.empty(num_splits * 2, dtype=np.float32)
    features_vector[0::2] = param_ratios
    features_vector[1::2] = normalized_smashed_sizes

    return {"features_vector": features_vector}