import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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



# 自定义图片数据集类
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image


# 预训练的特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x


# 边特征提取函数
def extract_edge_features(image_paths, similarity_metric='cosine'):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    all_features = []
    valid_indices = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(dataloader)):
            if images is None:
                continue
            images = images.to(device)
            features = feature_extractor(images)
            all_features.extend(features.cpu().numpy())
            valid_indices.extend(range(i * 16, i * 16 + len(images)))

    all_features = np.array(all_features)

    if similarity_metric == 'cosine':
        edge_features = cosine_similarity(all_features)
    elif similarity_metric == 'euclidean':
        edge_features = euclidean_distances(all_features)
    #  这个做个参考，看效果是不是比consin更好一点，可以使用，因为公式写的比较复杂，可以试试
    elif similarity_metric == 'pearson':
        num_samples = all_features.shape[0]
        edge_features = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i, num_samples):
                corr, _ = pearsonr(all_features[i], all_features[j])
                edge_features[i, j] = corr
                edge_features[j, i] = corr

    return edge_features, valid_indices


# 可视化图结构
# def visualize_graph(edge_features, threshold=0.5, node_labels=None):
#     num_nodes = edge_features.shape[0]
#     G = nx.Graph()
#     for i in range(num_nodes):
#         node_label = node_labels[i] if node_labels is not None else i
#         G.add_node(i, label=node_label)
#         for j in range(i + 1, num_nodes):
#             if edge_features[i, j] > threshold:
#                 G.add_edge(i, j, weight=edge_features[i, j])
#
#     pos = nx.spring_layout(G)
#     node_labels = nx.get_node_attributes(G, 'label')
#     nx.draw_networkx_nodes(G, pos)
#     nx.draw_networkx_edges(G, pos)
#     nx.draw_networkx_labels(G, pos, node_labels)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels)
#     plt.show()


# 边特征分类器
# 自定义分类，参数需要尝试出来
class EdgeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# 训练边特征分类器
def train_edge_classifier(edge_features, labels, num_classes=2, epochs=200, lr=0.001, batch_size=32):
    input_size = edge_features.shape[1]
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(edge_features, labels, test_size=0.2, random_state=42)

    classifier = EdgeClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    num_batches = len(X_train) // batch_size
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_features = X_train[start_idx:end_idx]
            batch_labels = y_train[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / num_batches:.4f}')

    # 在测试集上评估
    # ai为什么要这样写？？？？
    with torch.no_grad():
        test_outputs = classifier(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_correct = (test_predicted == y_test).sum().item()
        test_accuracy = test_correct / len(y_test)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    return classifier


# 评估边特征分类器
#  后面会单独再给一个模型评估的方案，但是感觉不好写模型评估
def evaluate_edge_classifier(classifier, edge_features, labels):
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        outputs = classifier(edge_features)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')


# 可视化边特征矩阵
#  heatmap可以问 研一的
# def visualize_edge_features(edge_features):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(edge_features, annot=False, cmap='coolwarm')
#     plt.title('Edge Features Heatmap')
#     plt.xlabel('Nodes')
#     plt.ylabel('Nodes')
#     plt.show()


# 保存边特征到文件
def save_edge_features(edge_features, file_path):
    np.save(file_path, edge_features)


# 加载边特征从文件
def load_edge_features(file_path):
    return np.load(file_path)


# 主函数
def main():
    image_folder = ''
    # image_folder = 'H:\HeterogeneousGraphSequenceSemanticDifferentiationNetworkforOccluded PersonRe-Identification\code\HGSD-Person-ReID\datasets\coco128\images\train2017'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    # 提取边特征
    edge_features, valid_indices = extract_edge_features(image_paths, similarity_metric='cosine')
    print(f'Edge features shape: {edge_features.shape}')

    # 可视化边特征矩阵
    # visualize_edge_features(edge_features)

    # 可视化图结构
    node_labels = [os.path.basename(image_paths[i]) for i in valid_indices]
    # visualize_graph(edge_features, threshold=0.5, node_labels=node_labels)

    # 标签 设计
    labels = ""
    # labels = np.random.randint(0, 2, edge_features.shape[0])

    # 训练边特征分类器
    classifier = train_edge_classifier(edge_features, labels)

    # 评估边特征分类器
    evaluate_edge_classifier(classifier, edge_features, labels)

    # 保存边特征到文件
    save_edge_features(edge_features, 'edge_features.npy')

    # 加载边特征从文件
    loaded_edge_features = load_edge_features('edge_features.npy')
    print(f'Loaded edge features shape: {loaded_edge_features.shape}')


if __name__ == "__main__":
    main()

