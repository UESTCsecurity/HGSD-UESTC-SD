# import numpy as np
# import matplotlib.pyplot as plt
# from __future__ import print_function, absolute_import      # 导入__future__模块的print_function和absolute_import特性，使得代码兼容Python 2.x和3.x版本
# import os                                                   # 导入os模块，该模块提供了对操作系统功能调用的接口
# import sys                                                  # 导入sys模块，该模块提供对Python解释器使用或维护的一些变量的访问，以及与解释器强烈交互的函数
# import time                                                 # 导入time模块，该模块提供了时间相关的函数
# import datetime                                             # 导入datetime模块，该模块提供了处理日期和时间的类
# import argparse                                             # 导入argparse模块，该模块用于处理命令行参数
# import os.path as osp                                       # 导入os.path模块，该模块提供了文件路径操作的相关函数，as osp是为了让后续的路径操作更加简洁
# import numpy as np                                          # 导入numpy模块，该模块提供了大量的数学函数和工具来处理大量的维度数组和矩阵计算，支持大量的维度数组和矩阵计算，矩阵运算和数学函数操作等
# import torch                                                # 导入torch模块，该模块提供了PyTorch的核心功能，如张量计算、神经网络、自动梯度等
# import torch.nn as nn                                       # 导入torch.nn模块，该模块提供了构建神经网络所需的各种类和函数
# import torch.backends.cudnn as cudnn                        # 导入torch.backends.cudnn模块，该模块提供了与CUDA相关的后端操作，让PyTorch可以利用GPU进行加速计算
# from torch.utils.data import DataLoader                     # 从torch.utils.data中导入DataLoader类，该类用于加载数据集并对其进行批量处理
# from torch.autograd import Variable                         # 从torch中导入autograd下的Variable类，该类用于包装Tensor以提供自动求导功能
# from torch.optim import lr_scheduler                        # 从torch.optim中导入lr_scheduler类，该类用于调整学习率
# import data_manager                                         # 导入自定义的数据管理模块data_manager，该模块可能包含一些数据预处理或数据加载的方法
# from video_loader import VideoDataset                       # 导入视频加载器VideoDataset，这个可能是自定义的数据加载类，用于加载视频相关的数据集
# import transforms as T                                      # 导入transforms模块，这个可能是自定义的数据预处理模块，包含各种图像或视频的预处理方法
# import models                                               # 导入模型相关的模块models，这个可能是自定义的模型模块，包含各种类型的模型定义和实现
# from models import resnet3d                                 # 从models中导入resnet3d类，这个可能是3D版本的ResNet模型的定义和实现
# from losses import CrossEntropyLabelSmooth, TripletLoss     # 从losses中导入CrossEntropyLabelSmooth和TripletLoss类，这两个可能是自定义的损失函数，分别对应交叉熵平滑和三元组损失
# from utils import AverageMeter, Logger, save_checkpoint     # 从utils中导入AverageMeter、Logger、save_checkpoint等工具类或函数，这些可能是用于处理训练过程中的各种任务，如计算平均值、记录日志、保存模型等
# from eval_metrics import evaluate                           # 从eval_metrics中导入evaluate函数，这个可能是用于评估模型性能的函数，通常用于在训练结束后对模型进行测试并计算相关的评估指标
# from samplers import RandomIdentitySampler                  # 从samplers中导入RandomIdentitySampler类，这个可能是用于数据采样的自定义类，通常用于处理有标签的数据时进行随机身份采样以防止过拟合等问题
#
#
# # 加载 CIFAR - 10 数据集函数
# def load_cifar10_data():
#     """
#     该函数加载 CIFAR - 10 数据集，并进行预处理
#     :return: 训练数据、测试数据及其对应的标签
#     """
#     (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#     # 数据归一化
#     X_train = X_train.astype('float32') / 255.0
#     X_test = X_test.astype('float32') / 255.0
#     # 标签进行 one - hot 编码
#     y_train = to_categorical(y_train, 10)
#     y_test = to_categorical(y_test, 10)
#     return X_train, y_train, X_test, y_test
#
#
# # 构建 CNN 模型函数
# def build_cnn_model():
#     """
#     该函数构建一个简单的 CNN 模型
#     :return: 构建好的 CNN 模型
#     """
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
#
# # 训练模型函数
# def train_model(model, X_train, y_train, epochs=100, batch_size=64, validation_split=0.2):
#     """
#     该函数用于训练 CNN 模型，并使用早停策略防止过拟合
#     :param model: 待训练的 CNN 模型
#     :param X_train: 训练数据
#     :param y_train: 训练数据的标签
#     :param epochs: 训练的轮数
#     :param batch_size: 每次训练使用的样本数量
#     :param validation_split: 用于验证集的比例
#     :return: 训练好的模型和训练历史记录
#     """
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                         validation_split=validation_split, callbacks=[early_stopping])
#     return model, history
#
#
# # 评估模型函数
# def evaluate_model(model, X_test, y_test):
#     """
#     该函数使用测试数据评估模型的性能
#     :param model: 训练好的 CNN 模型
#     :param X_test: 测试数据
#     :param y_test: 测试数据的标签
#     :return: 测试集的损失和准确率
#     """
#     loss, accuracy = model.evaluate(X_test, y_test)
#     print(f"测试集损失: {loss}")
#     print(f"测试集准确率: {accuracy}")
#     return loss, accuracy
#
#
# # 可视化训练过程函数
# def visualize_training(history):
#     """
#     该函数用于可视化模型的训练过程，包括训练损失和验证损失、训练准确率和验证准确率
#     :param history: 训练历史记录
#     """
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='训练损失')
#     plt.plot(history.history['val_loss'], label='验证损失')
#     plt.title('训练和验证损失')
#     plt.xlabel('轮数')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='训练准确率')
#     plt.plot(history.history['val_accuracy'], label='验证准确率')
#     plt.title('训练和验证准确率')
#     plt.xlabel('轮数')
#     plt.ylabel('准确率')
#     plt.legend()
#     plt.show()
#
#
# # 主函数
# def main():
#     X_train, y_train, X_test, y_test = load_cifar10_data()
#     model = build_cnn_model()
#     model, history = train_model(model, X_train, y_train)
#     loss, accuracy = evaluate_model(model, X_test, y_test)
#     visualize_training(history)
#
#
# if __name__ == "__main__":
#     main()
#
