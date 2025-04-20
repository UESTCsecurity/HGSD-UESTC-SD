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
from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

# 配置日志记录
logging.basicConfig(filename='reid_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    # root = './data/mars'
    root = r'H:\REIDDate\mars\MARS-v160809'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        #

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root = './data/ilids-vid'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    data_dir = osp.join(root, 'i-LIDS-VID')
    split_dir = osp.join(root, 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

    def __init__(self, split_id=0):
        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']

            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids / 2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))

                train_idxs = [int(i) - 1 for i in train_idxs]
                test_idxs = [int(i) - 1 for i in test_idxs]

                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]

                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.

    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
            self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


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
def node_aggregation(cosine_sim, pearson_sim, threshold=0.89, features=None):
    """
    根据相似度阈值进行节点聚合
    :param cosine_sim: 余弦相似度矩阵
    :param pearson_sim: 皮尔逊相似度矩阵
    :param threshold: 相似度阈值
    :param features: 局部特征矩阵
    :return: 聚合后的节点列表
    """
    num_samples = cosine_sim.shape[0]
    graph = nx.Graph()
    print(f"开始构建图，总节点数: {num_samples}")
    for i in range(num_samples):
        graph.add_node(i)
        for j in range(i + 1, num_samples):
            if cosine_sim[i, j] > threshold and pearson_sim[i, j] > threshold:
                graph.add_edge(i, j)
                print(f"添加边: ({i}, {j}), 余弦相似度: {cosine_sim[i, j]:.4f}, 皮尔逊相似度: {pearson_sim[i, j]:.4f}")
    print("图构建完成")

    connected_components = list(nx.connected_components(graph))
    print(f"找到 {len(connected_components)} 个连通分量")
    aggregated_nodes = []
    for idx, component in enumerate(connected_components):
        print(f"处理第 {idx + 1} 个连通分量，包含节点: {component}")
        component_features = np.mean([features[k] for k in component], axis=0)
        aggregated_nodes.append(component_features)
        print(f"第 {idx + 1} 个连通分量聚合完成")
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

    # 节点聚合 阈值
    #  --------------------------------------------------0.89，0.72,0.6,0.5，0.4,0.9,0.8，0.7-------------------------------------------------------------
    aggregated_nodes = node_aggregation(cosine_sim, pearson_sim, threshold=0.89, features=all_features)

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


__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    # test
    #dataset = Market1501()
    dataset = Mars()
    # dataset = iLIDSVID()
    # dataset = PRID()


#  需要进行的额外操作：
    # 数据增强：在 PedestrianDataset 类中添加了数据增强选项，通过随机水平翻转、随机旋转和颜色抖动等操作增加数据的多样性。
    # 模型复杂度提升：在 Classifier 类中增加了一层全连接层，提高模型的表达能力。
    # 学习率调度器：在训练函数中引入了 StepLR 学习率调度器，每 100 个 epoch 降低一次学习率。
    # 交叉验证：添加了 cross_validate 函数，使用 K 折交叉验证评估模型的稳定性和泛化能力。
    # 评估指标：在 evaluate 函数中计算了准确率、精确率、召回率，更全面地评估模型性能。

    # 设计模型评估方案：
        # 跑不出来： 先跑个10次，每次将阈值改动一下，看结果趋势！！！！   边分支！！！
        # 节点分支，改动consin函数或者皮尔逊函数 用来重新模拟建模图结构




# 1. 加载数据集，包括 mars iLIDSVID Occluded-DukeMTMC-ReID uestc
# 2. 数据集图像获取加载，然后进行一系列的初始化工作
# 3. 进行图像数据的特征提取，使用PVTv2-B2作为特征提取模型
# 4. 将图像以PCB-RPP的方式进行局部细粒度的划分，然后获取特征向量
# 5. 通过  consin  以及  皮尔逊相似度  的模式，获取多个局部特征之间的相似性
# 6. 通过这些局部特征直接的相似性的阈值0.89，视为这些局部特征能否作为同一语义的节点
# 7. 将这些同一语义的节点进行时空特征的聚合，并且作为reid任务的核心判别性对象
# 8. 先进行800个epoch的训练，训练过程参数的明细日志保存


