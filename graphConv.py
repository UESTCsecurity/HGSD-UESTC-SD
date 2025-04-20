from __future__ import print_function, absolute_import      # 导入__future__模块的print_function和absolute_import特性，使得代码兼容Python 2.x和3.x版本
import os                                                   # 导入os模块，该模块提供了对操作系统功能调用的接口
import sys                                                  # 导入sys模块，该模块提供对Python解释器使用或维护的一些变量的访问，以及与解释器强烈交互的函数
import time                                                 # 导入time模块，该模块提供了时间相关的函数
import datetime                                             # 导入datetime模块，该模块提供了处理日期和时间的类
import argparse                                             # 导入argparse模块，该模块用于处理命令行参数
import os.path as osp                                       # 导入os.path模块，该模块提供了文件路径操作的相关函数，as osp是为了让后续的路径操作更加简洁
import numpy as np                                          # 导入numpy模块，该模块提供了大量的数学函数和工具来处理大量的维度数组和矩阵计算，支持大量的维度数组和矩阵计算，矩阵运算和数学函数操作等
import torch                                                # 导入torch模块，该模块提供了PyTorch的核心功能，如张量计算、神经网络、自动梯度等
import torch.nn as nn                                       # 导入torch.nn模块，该模块提供了构建神经网络所需的各种类和函数
import torch.backends.cudnn as cudnn                        # 导入torch.backends.cudnn模块，该模块提供了与CUDA相关的后端操作，让PyTorch可以利用GPU进行加速计算
from torch.utils.data import DataLoader                     # 从torch.utils.data中导入DataLoader类，该类用于加载数据集并对其进行批量处理
from torch.autograd import Variable                         # 从torch中导入autograd下的Variable类，该类用于包装Tensor以提供自动求导功能
from torch.optim import lr_scheduler                        # 从torch.optim中导入lr_scheduler类，该类用于调整学习率
import data_manager                                         # 导入自定义的数据管理模块data_manager，该模块可能包含一些数据预处理或数据加载的方法
from video_loader import VideoDataset                       # 导入视频加载器VideoDataset，这个可能是自定义的数据加载类，用于加载视频相关的数据集
import transforms as T                                      # 导入transforms模块，这个可能是自定义的数据预处理模块，包含各种图像或视频的预处理方法
import models                                               # 导入模型相关的模块models，这个可能是自定义的模型模块，包含各种类型的模型定义和实现
from models import resnet3d                                 # 从models中导入resnet3d类，这个可能是3D版本的ResNet模型的定义和实现
from losses import CrossEntropyLabelSmooth, TripletLoss     # 从losses中导入CrossEntropyLabelSmooth和TripletLoss类，这两个可能是自定义的损失函数，分别对应交叉熵平滑和三元组损失
from utils import AverageMeter, Logger, save_checkpoint     # 从utils中导入AverageMeter、Logger、save_checkpoint等工具类或函数，这些可能是用于处理训练过程中的各种任务，如计算平均值、记录日志、保存模型等
from eval_metrics import evaluate                           # 从eval_metrics中导入evaluate函数，这个可能是用于评估模型性能的函数，通常用于在训练结束后对模型进行测试并计算相关的评估指标
from samplers import RandomIdentitySampler                  # 从samplers中导入RandomIdentitySampler类，这个可能是用于数据采样的自定义类，通常用于处理有标签的数据时进行随机身份采样以防止过拟合等问题


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv