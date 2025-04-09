import math
import torch
import torch.nn.functional as F
import torch.nn as nn

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, levels = [2, 4, 8], pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.levels = levels
        self.pool_type = pool_type

    def forward(self, x):
        b, c, h, w = x.size()
        for level in self.levels:

            kernel_size = (level, level)
            stride = (level, level)
            padding = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式 
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(b, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(b, -1)

            # 展开、拼接
            if (level == self.levels[0] ):
                x_flatten = tensor.view(b, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(b, -1)), 1)
        return x_flatten
