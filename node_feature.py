import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# 定义图片数据集类
# 建议沿用dataset中的处理方法
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


# 节点特征提取函数
def extract_node_features(image_paths, num_clusters=10, clustering_method='kmeans', dim_reduction=None):
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

    if dim_reduction == 'pca':
        pca = PCA(n_components=0.95)
        all_features = pca.fit_transform(all_features)
    elif dim_reduction == 'tsne':
        tsne = TSNE(n_components=2)
        all_features = tsne.fit_transform(all_features)

    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(all_features)
        node_features = kmeans.cluster_centers_
        labels = kmeans.labels_
    elif clustering_method == 'hierarchical':
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
        labels = hierarchical.fit_predict(all_features)
        node_features = []
        for i in range(num_clusters):
            cluster_features = all_features[labels == i]
            if len(cluster_features) > 0:
                node_features.append(np.mean(cluster_features, axis=0))
        node_features = np.array(node_features)

    return node_features, labels, valid_indices


# 可视化节点特征
def visualize_node_features(node_features, labels, valid_indices, image_paths, num_clusters=10):
    if node_features.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        for i in range(num_clusters):
            cluster_features = node_features[labels == i]
            plt.scatter(cluster_features[:, 0], cluster_features[:, 1], label=f'Cluster {i}')
        plt.legend()
        plt.title('Node Features Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
    elif node_features.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # for i in range(num_clusters):
        #     cluster_features = node_features[labels == i]
        #     ax.scatter(cluster_features[:, 0], cluster_features[:, 1], cluster_features[:, 2], label=f'Cluster {i}')
        # ax.set_xlabel('Dimension 1')
        # ax.set_ylabel('Dimension 2')
        # ax.set_zlabel('Dimension 3')
        # ax.set_title('Node Features Visualization')
        ax.legend()
        plt.show()
    else:
        print("Cannot visualize node features with more than 3 dimensions.")

    # 可视化每个聚类中的示例图像
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            sample_index = cluster_indices[0]
            image_index = valid_indices[sample_index]
            image_path = image_paths[image_index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(f'Example Image from Cluster {i}')
            plt.show()


# 节点特征分类器
class NodeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NodeClassifier, self).__init__()
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


# 训练节点特征分类器
def train_node_classifier(node_features, labels, num_classes=10, epochs=200, lr=0.001, batch_size=32):
    input_size = node_features.shape[1]
    node_features = torch.tensor(node_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    classifier = NodeClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    num_batches = len(node_features) // batch_size
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_features = node_features[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / num_batches:.4f}')

    return classifier


# 评估节点特征分类器
def evaluate_node_classifier(classifier, node_features, labels):
    node_features = torch.tensor(node_features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        outputs = classifier(node_features)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')


# 计算节点特征的熵
# 可解释性问题入口函数
def calculate_node_entropy(node_features):
    entropy_values = []
    for feature in node_features:
        feature = np.abs(feature)
        feature = feature / np.sum(feature)
        entropy_values.append(entropy(feature))
    return entropy_values


# 计算节点特征的轮廓系数
def calculate_node_silhouette_score(node_features, labels):
    return silhouette_score(node_features, labels)


# 保存节点特征到文件
def save_node_features(node_features, file_path):
    np.save(file_path, node_features)

# 加载节点特征从文件
def load_node_features(file_path):
    return np.load(file_path)


# 主函数
def main():
    image_folder = 'H:\HeterogeneousGraphSequenceSemanticDifferentiationNetworkforOccluded PersonRe-Identification\code\HGSD-Person-ReID\datasets\coco128\images\train2017'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    # 提取节点特征
    node_features, labels, valid_indices = extract_node_features(image_paths, num_clusters=10,
                                                                 clustering_method='kmeans', dim_reduction='pca')
    print(f'Node features shape: {node_features.shape}')

    # 可视化节点特征
    visualize_node_features(node_features, labels, valid_indices, image_paths, num_clusters=10)

    # 假设标签
    labels = np.random.randint(0, 10, node_features.shape[0])

    # 训练节点特征分类器
    classifier = train_node_classifier(node_features, labels)

    # 评估节点特征分类器
    evaluate_node_classifier(classifier, node_features, labels)

    # 计算节点特征的熵
    entropy_values = calculate_node_entropy(node_features)
    print(f'Node feature entropy: {entropy_values}')

    # 计算节点特征的轮廓系数
    silhouette_score = calculate_node_silhouette_score(node_features, labels)
    print(f'Node feature silhouette score: {silhouette_score}')

    # 保存节点特征到文件
    save_node_features(node_features, 'node_features.npy')

    # 加载节点特征从文件
    loaded_node_features = load_node_features('node_features.npy')
    print(f'Loaded node features shape: {loaded_node_features.shape}')


if __name__ == "__main__":
    main()

