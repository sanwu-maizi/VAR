import math
import torch
import torch.nn as nn

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

class IdentityScaleLayer(nn.Module):
    def __init__(self, channels, scales):
        super().__init__()
        self.channels = channels
        self.scales = nn.Parameter(scales.data)  # 直接存储 scales

    def forward(self, x):
        return x / self.scales.view(1, -1).to(x.device)  # 前层除以 scales

class ScaledDynamicFC(nn.Module):
    def __init__(self, fc_layer, scales):
        super().__init__()
        self.pre_scale = IdentityScaleLayer(fc_layer.in_features, scales)  # 传入 scales
        self.fc = fc_layer  # 原始 fc1
        self.scales = nn.Parameter(scales.data)  # 存储 scales

    def forward(self, x):
        x = self.pre_scale(x)  # 前层除以 scales
        x = self.fc(x)  # 后层乘以 scales
        return x