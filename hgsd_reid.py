import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from timm.models import pvt_v2_b2
import logging
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

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


# 1. 加载数据集，包括 mars iLIDSVID Occluded-DukeMTMC-ReID uestc
# 2. 数据集图像获取加载，然后进行一系列的初始化工作
# 3. 进行图像数据的特征提取，使用PVTv2-B2作为特征提取模型
# 4. 将图像以PCB-RPP的方式进行局部细粒度的划分，然后获取特征向量
# 5. 通过consin以及皮尔逊相似度的模式，获取多个局部特征之间的相似性
# 6. 通过这些局部特征直接的相似性的阈值0.89，视为这些局部特征能否作为同一语义的节点
# 7. 将这些同一语义的节点进行时空特征的聚合，并且作为reid任务的核心判别性对象
# 8. 进行800个epoch的训练，训练过程参数的明细日志保存


# 配置日志记录
logging.basicConfig(filename='reid_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 自定义行人图像数据集类
class PedestrianDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        """
        初始化行人图像数据集类
        :param root_dir: 数据集根目录
        :param transform: 图像预处理变换
        :param augment: 是否进行数据增强
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.image_paths = []
        self.labels = []
        for person_id in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_id)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(person_id))

        if self.augment:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
            ])

    def __len__(self):
        """
        返回数据集的长度
        :return: 数据集的长度
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的图像和标签
        :param idx: 索引
        :return: 图像和对应的标签
        """
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None, self.labels[idx]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        if self.augment:
            image = self.augmentation_transform(image)
        return image, self.labels[idx]


# PCB - RPP 模块
class PCB_RPP(nn.Module):
    def __init__(self, num_parts=6):
        """
        初始化 PCB - RPP 模块
        :param num_parts: 划分的局部部分数量
        """
        super(PCB_RPP, self).__init__()
        self.num_parts = num_parts
        self.avgpool = nn.AdaptiveAvgPool2d((num_parts, 1))
        self.conv_list = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
            for _ in range(num_parts)
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(256)
            for _ in range(num_parts)
        ])
        self.relu_list = nn.ModuleList([
            nn.ReLU(inplace=True)
            for _ in range(num_parts)
        ])

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的特征图
        :return: 局部特征列表
        """
        x = self.avgpool(x)
        part_features = []
        for i in range(self.num_parts):
            part = x[:, :, i, :].unsqueeze(2)
            part = self.conv_list[i](part)
            part = self.bn_list[i](part)
            part = self.relu_list[i](part)
            part = part.view(part.size(0), -1)
            part_features.append(part)
        return part_features


# 特征提取器，使用 PVTv2 - B2
class FeatureExtractor(nn.Module):
    def __init__(self, num_parts=6):
        """
        初始化特征提取器，使用 PVTv2 - B2 并融合 PCB - RPP
        :param num_parts: PCB - RPP 划分的局部部分数量
        """
        super(FeatureExtractor, self).__init__()
        self.pvt = pvt_v2_b2(pretrained=True)
        self.pcb_rpp = PCB_RPP(num_parts=num_parts)
        self.num_parts = num_parts

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的图像张量
        :return: 提取的特征（包含局部特征）
        """
        x = self.pvt.forward_features(x)
        part_features = self.pcb_rpp(x)
        all_part_features = []
        for part in part_features:
            part = part.view(part.size(0), -1)
            all_part_features.append(part)
        all_part_features = torch.stack(all_part_features, dim=1)
        all_part_features = all_part_features.view(all_part_features.size(0), -1)
        return all_part_features


# 计算局部特征之间的相似性
def calculate_similarities(features):
    """
    计算局部特征之间的余弦相似度和皮尔逊相似度
    :param features: 局部特征矩阵
    :return: 余弦相似度矩阵和皮尔逊相似度矩阵
    """
    cosine_sim = cosine_similarity(features)
    num_samples = features.shape[0]
    pearson_sim = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i, num_samples):
            corr, _ = pearsonr(features[i], features[j])
            pearson_sim[i, j] = corr
            pearson_sim[j, i] = corr
    return cosine_sim, pearson_sim


# 节点聚合
def node_aggregation(cosine_sim, pearson_sim, threshold=0.89):
    """
    根据相似度阈值进行节点聚合
    :param cosine_sim: 余弦相似度矩阵
    :param pearson_sim: 皮尔逊相似度矩阵
    :param threshold: 相似度阈值
    :return: 聚合后的节点列表
    """
    num_samples = cosine_sim.shape[0]
    graph = nx.Graph()
    for i in range(num_samples):
        graph.add_node(i)
        for j in range(i + 1, num_samples):
            if cosine_sim[i, j] > threshold and pearson_sim[i, j] > threshold:
                graph.add_edge(i, j)
    connected_components = list(nx.connected_components(graph))
    aggregated_nodes = []
    # for component in connected_components:
        # component_features = np.mean([features[idx] for idx in component], axis=0)
        # aggregated_nodes.append(component_features)
    return np.array(aggregated_nodes)


# 训练分类器
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        初始化分类器
        :param input_size: 输入特征的维度
        :param num_classes: 分类的类别数
        """
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的特征张量
        :return: 分类结果
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# 训练函数
def train(model, dataloader, criterion, optimizer, device, epochs=800, scheduler=None):
    """
    训练模型
    :param model: 训练的模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 设备（CPU 或 GPU）
    :param epochs: 训练的轮数
    :param scheduler: 学习率调度器
    """
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

    return train_losses


# 评估函数
def evaluate(model, dataloader, device):
    """
    评估模型
    :param model: 评估的模型
    :param dataloader: 数据加载器
    :param device: 设备（CPU 或 GPU）
    :return: 准确率、精确率、召回率和 F1 分数
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1


# 交叉验证函数
def cross_validate(features, labels, num_classes, device, epochs=800, lr=0.001, batch_size=32, k=5):
    """
    进行交叉验证
    :param features: 特征矩阵
    :param labels: 标签列表
    :param num_classes: 分类的类别数
    :param device: 设备（CPU 或 GPU）
    :param epochs: 训练的轮数
    :param lr: 学习率
    :param batch_size: 批量大小
    :param k: 交叉验证的折数
    :return: 平均准确率、平均精确率、平均召回率和平均 F1 分数
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = X_train.shape[1]
        classifier = Classifier(input_size, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        train(classifier, train_dataloader, criterion, optimizer, device, epochs=epochs, scheduler=scheduler)
        accuracy, precision, recall, f1 = evaluate(classifier, test_dataloader, device)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    logging.info(f'Cross Validation - Average Accuracy: {avg_accuracy:.4f}, Average Precision: {avg_precision:.4f}, '
                 f'Average Recall: {avg_recall:.4f}, Average F1: {avg_f1:.4f}')
    print(f'Cross Validation - Average Accuracy: {avg_accuracy:.4f}, Average Precision: {avg_precision:.4f}, '
          f'Average Recall: {avg_recall:.4f}, Average F1: {avg_f1:.4f}')

    return avg_accuracy, avg_precision, avg_recall, avg_f1


# 可视化训练损失
def visualize_train_loss(train_losses):
    """
    可视化训练损失
    :param train_losses: 训练损失列表
    """
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()


# 主函数
def main():
    # 数据集路径
    # 使用加载函数加载，不指定路径
    # mars_dir = ''
    # ilidsvid_dir = ''
    # occluded_duke_dir = ''

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    mars_dataset = PedestrianDataset(mars_dir, transform=transform, augment=True)
    ilidsvid_dataset = PedestrianDataset(ilidsvid_dir, transform=transform, augment=True)
    occluded_duke_dataset = PedestrianDataset(occluded_duke_dir, transform=transform, augment=True)

    all_datasets = torch.utils.data.ConcatDataset([mars_dataset, ilidsvid_dataset, occluded_duke_dataset])
    dataloader = DataLoader(all_datasets, batch_size=16, shuffle=True)

    # 初始化特征提取器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = feature_extractor(images)
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_features = np.array(all_features)

    # 计算局部特征之间的相似性
    cosine_sim, pearson_sim = calculate_similarities(all_features)

    # 节点聚合
    # 这里感觉写的有点问题
    aggregated_nodes = node_aggregation(cosine_sim, pearson_sim, threshold=0.89)

    # 准备训练数据
    num_classes = len(set(all_labels))
    input_size = aggregated_nodes.shape[1]

    # 交叉验证
    avg_accuracy, avg_precision, avg_recall, avg_f1 = cross_validate(aggregated_nodes, np.array(all_labels),
                                                                     num_classes, device, epochs=800, lr=0.001,
                                                                     batch_size=32, k=5)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(aggregated_nodes, all_labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    classifier = Classifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # 训练分类器
    train_losses = train(classifier, train_dataloader, criterion, optimizer, device, epochs=800, scheduler=scheduler)

    # 可视化训练损失
    visualize_train_loss(train_losses)

    # 评估模型
    accuracy, precision, recall, f1 = evaluate(classifier, test_dataloader, device)
    logging.info(f'Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print(f'Test - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')


#  需要进行的额外操作：
    # 数据增强：在 PedestrianDataset 类中添加了数据增强选项，通过随机水平翻转、随机旋转和颜色抖动等操作增加数据的多样性。
    # 模型复杂度提升：在 Classifier 类中增加了一层全连接层，提高模型的表达能力。
    # 学习率调度器：在训练函数中引入了 StepLR 学习率调度器，每 100 个 epoch 降低一次学习率。
    # 交叉验证：添加了 cross_validate 函数，使用 K 折交叉验证评估模型的稳定性和泛化能力。
    # 评估指标：在 evaluate 函数中计算了准确率、精确率、召回率，更全面地评估模型性能。

    # 设计模型评估方案：
        # 跑不出来： 先跑个10次，每次将阈值改动一下，看结果趋势！！！！   边分支！！！
        # 节点分支，改动consin函数或者皮尔逊函数 用来重新模拟建模图结构
if __name__ == "__main__":
    main()

# 1. 加载数据集，包括 mars iLIDSVID Occluded-DukeMTMC-ReID uestc
# 2. 数据集图像获取加载，然后进行一系列的初始化工作
# 3. 进行图像数据的特征提取，使用PVTv2-B2作为特征提取模型
# 4. 将图像以PCB-RPP的方式进行局部细粒度的划分，然后获取特征向量
# 5. 通过consin以及皮尔逊相似度的模式，获取多个局部特征之间的相似性
# 6. 通过这些局部特征直接的相似性的阈值0.89，视为这些局部特征能否作为同一语义的节点
# 7. 将这些同一语义的节点进行时空特征的聚合，并且作为reid任务的核心判别性对象
# 8. 先进行800个epoch的训练，训练过程参数的明细日志保存