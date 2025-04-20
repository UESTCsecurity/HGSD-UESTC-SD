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
import time

# TemporalFeatureExtractor 类在原有的 ResNet18 特征提取基础上，加入了 PCB_RPP 模块，提取行人视频帧的局部特征，并对每个视频序列的局部特征进行平均池化。
# 并对每个视频序列的局部特征进行平均池化边特征提取：extract_temporal_edge_features 函数在提取时间边特征时，考虑了 PCB - RPP 提取的局部特征。
# 训练和评估：train_temporal_edge_classifier 和 evaluate_temporal_edge_classifier 函数用于训练和评估基于融合 PCB - RPP 特征的时间边特征分类器。

# 自定义行人视频数据集类
class PedestrianVideoDataset(Dataset):
    def __init__(self, video_paths, labels, seq_len=4, transform=None):
        """
        初始化行人视频数据集类
        :param video_paths: 视频文件路径列表
        :param labels: 对应的行人标签列表
        :param seq_len: 每个视频序列的帧数
        :param transform: 图像预处理变换
        """
        self.video_paths = video_paths
        self.labels = labels
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        :return: 数据集的长度
        """
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        获取指定索引的视频序列和标签
        :param idx: 索引
        :return: 视频序列和对应的标签
        """
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                if len(frames) == self.seq_len:
                    break
            else:
                break
        cap.release()

        if len(frames) < self.seq_len:
            # 填充缺失的帧
            last_frame = frames[-1] if frames else torch.zeros_like(self.transform(np.zeros((224, 224, 3))))
            while len(frames) < self.seq_len:
                frames.append(last_frame)

        frames = torch.stack(frames)
        return frames, self.labels[idx]


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


# 预训练的特征提取器，用于提取行人视频帧的特征，融合 PCB - RPP
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, num_parts=6):
        """
        初始化时间特征提取器，融合 PCB - RPP
        使用预训练的 ResNet18 模型，去掉最后一层全连接层
        :param num_parts: PCB - RPP 划分的局部部分数量
        """
        super(TemporalFeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.pcb_rpp = PCB_RPP(num_parts=num_parts)
        self.num_parts = num_parts

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入的视频帧张量
        :return: 提取的时间特征（包含局部特征）
        """
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.model(x)
        part_features = self.pcb_rpp(x)
        all_part_features = []
        for part in part_features:
            part = part.view(batch_size, seq_len, -1)
            # 对每个视频序列的局部特征进行平均池化
            part = torch.mean(part, dim=1)
            all_part_features.append(part)
        all_part_features = torch.stack(all_part_features, dim=1)
        all_part_features = all_part_features.view(batch_size, -1)
        return all_part_features


# 时间边特征提取函数，考虑 PCB - RPP 局部特征
def extract_temporal_edge_features(video_paths, labels, similarity_metric='cosine', seq_len=4, num_parts=6):
    """
    提取行人视频的时间边特征，考虑 PCB - RPP 局部特征
    :param video_paths: 视频文件路径列表
    :param labels: 对应的行人标签列表
    :param similarity_metric: 相似度度量方法，可选 'cosine', 'euclidean', 'pearson'
    :param seq_len: 每个视频序列的帧数
    :param num_parts: PCB - RPP 划分的局部部分数量
    :return: 边特征矩阵和有效的索引列表
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PedestrianVideoDataset(video_paths, labels, seq_len=seq_len, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = TemporalFeatureExtractor(num_parts=num_parts).to(device)
    feature_extractor.eval()

    all_features = []
    valid_indices = []
    valid_labels = []
    with torch.no_grad():
        for i, (frames, labels_batch) in enumerate(tqdm(dataloader)):
            if frames is None:
                continue
            frames = frames.to(device)
            features = feature_extractor(frames)
            all_features.extend(features.cpu().numpy())
            valid_indices.extend(range(i * 4, i * 4 + len(frames)))
            valid_labels.extend(labels_batch.numpy())

    all_features = np.array(all_features)

    if similarity_metric == 'cosine':
        edge_features = cosine_similarity(all_features)
    elif similarity_metric == 'euclidean':
        edge_features = euclidean_distances(all_features)
    elif similarity_metric == 'pearson':
        num_samples = all_features.shape[0]
        edge_features = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(i, num_samples):
                corr, _ = pearsonr(all_features[i], all_features[j])
                edge_features[i, j] = corr
                edge_features[j, i] = corr

    return edge_features, valid_indices, valid_labels


# 可视化时间图结构
def visualize_temporal_graph(edge_features, threshold=0.5, node_labels=None):
    """
    可视化行人视频的时间图结构
    :param edge_features: 边特征矩阵
    :param threshold: 边的权重阈值
    :param node_labels: 节点标签列表
    """
    num_nodes = edge_features.shape[0]
    G = nx.Graph()
    for i in range(num_nodes):
        node_label = node_labels[i] if node_labels is not None else i
        G.add_node(i, label=node_label)
        for j in range(i + 1, num_nodes):
            if edge_features[i, j] > threshold:
                G.add_edge(i, j, weight=edge_features[i, j])

    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, node_labels)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()


# 时间边特征分类器
class TemporalEdgeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        初始化时间边特征分类器
        :param input_size: 输入特征的维度
        :param num_classes: 分类的类别数
        """
        super(TemporalEdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)

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
        return x


# 训练时间边特征分类器
def train_temporal_edge_classifier(edge_features, labels, num_classes=2, epochs=200, lr=0.001, batch_size=32):
    """
    训练时间边特征分类器
    :param edge_features: 边特征矩阵
    :param labels: 对应的标签列表
    :param num_classes: 分类的类别数
    :param epochs: 训练的轮数
    :param lr: 学习率
    :param batch_size: 批量大小
    :return: 训练好的分类器
    """
    input_size = edge_features.shape[1]
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(edge_features, labels, test_size=0.2, random_state=42)

    classifier = TemporalEdgeClassifier(input_size, num_classes)
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
    with torch.no_grad():
        test_outputs = classifier(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_correct = (test_predicted == y_test).sum().item()
        test_accuracy = test_correct / len(y_test)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    return classifier


# 评估时间边特征分类器
def evaluate_temporal_edge_classifier(classifier, edge_features, labels):
    """
    评估时间边特征分类器
    :param classifier: 训练好的分类器
    :param edge_features: 边特征矩阵
    :param labels: 对应的标签列表
    """
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        outputs = classifier(edge_features)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')


# 可视化时间边特征矩阵
def visualize_temporal_edge_features(edge_features):
    """
    可视化时间边特征矩阵
    :param edge_features: 边特征矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(edge_features, annot=False, cmap='coolwarm')
    plt.title('Temporal Edge Features Heatmap')
    plt.xlabel('Nodes')
    plt.ylabel('Nodes')
    plt.show()


# 保存时间边特征到文件
def save_temporal_edge_features(edge_features, file_path):
    """
    保存时间边特征矩阵到文件
    :param edge_features: 边特征矩阵
    :param file_path: 文件路径
    """
    np.save(file_path, edge_features)


# 加载时间边特征从文件
def load_temporal_edge_features(file_path):
    """
    从文件中加载时间边特征矩阵
    :param file_path: 文件路径
    :return: 加载的边特征矩阵
    """
    return np.load(file_path)


# 主函数
def main():
    # 假设行人视频文件夹路径
    video_folder = 'H:\HeterogeneousGraphSequenceSemanticDifferentiationNetworkforOccluded PersonRe-Identification\code\HGSD-Person-ReID\datasets\coco128\images\train2017'
    video_paths = []
    labels = []
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(root, file))
                # 假设标签是文件夹名
                label = int(os.path.basename(root))
                labels.append(label)

    # 提取时间边特征，考虑 PCB - RPP
    num_parts = 6
    edge_features, valid_indices, valid_labels = extract_temporal_edge_features(video_paths, labels,
                                                                                similarity_metric='cosine',
                                                                                seq_len=4, num_parts=num_parts)
    print(f'Temporal edge features shape: {edge_features.shape}')

    # 可视化时间边特征矩阵
    visualize_temporal_edge_features(edge_features)

    # 可视化时间图结构
    node_labels = [os.path.basename(video_paths[i]) for i in valid_indices]
    visualize_temporal_graph(edge_features, threshold=0.5, node_labels=node_labels)

    # 训练时间边特征分类器
    classifier = train_temporal_edge_classifier(edge_features, valid_labels)

    # 评估时间边特征分类器
    evaluate_temporal_edge_classifier(classifier, edge_features, valid_labels)

    # 保存时间边特征到文件
    save_temporal_edge_features(edge_features, 'temporal_edge_features_with_pcb_rpp.npy')

    # 加载时间边特征从文件
    loaded_edge_features = load_temporal_edge_features('temporal_edge_features_with_pcb_rpp.npy')
    print(f'Loaded temporal edge features shape: {loaded_edge_features.shape}')


if __name__ == "__main__":
    main()

