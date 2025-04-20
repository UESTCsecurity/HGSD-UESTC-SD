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


# 创建一个ArgumentParser对象，description参数为这个程序的简短描述，这里用于训练视频模型使用交叉熵损失
parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')

# Datasets
# 添加 '-d' 或 '--dataset' 参数，默认值为 'mars'，choices参数限制可选值为data_manager.get_names()返回的值
parser.add_argument('-d', '--dataset', type=str, default='mars', choices=data_manager.get_names())
# 添加 '-j' 或 '--workers' 参数，默认值为 4，表示数据加载的工作进程数
parser.add_argument('-j', '--workers', default=4, type=int, help="number of data loading workers (default: 4)")
# 添加 '--height' 参数，默认值为 224，表示图像的高度
# 图像的宽度为112
# 一个tracklet中的一个seq-len的长度为4
parser.add_argument('--height', type=int, default=224, help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112, help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")


# Optimization options 图像的优化过程
# 默认值为 800，表示最大的训练轮数
parser.add_argument('--max-epoch', default=800, type=int, help="maximum epochs to run")
# 默认值为 0，表示开始的训练轮数
parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
# 默认值为 32，表示训练批次大小
parser.add_argument('--train-batch', default=32, type=int, help="train batch size")
# 默认值为 1，表示测试批次大小
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
# 默认值为 0.0003，表示学习率
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
# 默认值为 200，表示学习率衰减的步长（如果大于0）
parser.add_argument('--stepsize', default=200, type=int, help="stepsize to decay learning rate (>0 means this is enabled)")
# 默认值为 0.1，表示学习率衰减的因子
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
# 默认值为 5e-04，表示权重的衰减因子（正则化项）
parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")
# 默认值为 0.3，表示三元组损失的margin值
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
# 默认值为 4，表示每个身份实例的数量
parser.add_argument('--num-instances', type=int, default=4,  help="number of instances per identity")
# 如果这个参数为True，则在训练中只使用htri损失
parser.add_argument('--htri-only', action='store_true', default=False, help="if this is True, only htri loss is used in training")

# Architecture
# 默认值为 'resnet50tp'，表示使用的模型架构，可选的值为 'resnet503d', 'resnet50tp', 'resnet50ta', 'resnetrnn'
# parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
# 默认值为 'avg'，表示池化方式，可选值为 'avg' 和 'max'
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
# 默认值为80，表示打印训练信息的频率
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
# 默认值为1，表示设置随机种子
parser.add_argument('--seed', type=int, default=1, help="manual seed")
# 对于resnet3d模型需要设置预训练模型路径
# 如果这个参数为True，则只进行评估
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
# 默认值为50，表示每隔N个epoch进行评估（设置为-1表示训练后进行测试）
parser.add_argument('--eval-step', type=int, default=50, help="run evaluation for every N epochs (set to -1 to test after training)")
# 默认值为 'log'，表示保存日志和模型的目录
parser.add_argument('--save-dir', type=str, default='log')
# 如果这个参数为True，则使用CPU进行计算
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
# 默认值为 '0'，表示使用哪些GPU设备进行计算（例如：CUDA_VISIBLE_DEVICES=0,1,2 python train.py）
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

# 使用 argparse 解析命令行参数并赋值给变量 args
args = parser.parse_args()

def main():
    # 设置PyTorch的随机种子，保证实验的可重复性
    torch.manual_seed(args.seed)
    # 设置环境变量，使得CUDA能够识别可用的GPU设备
    # 相当于在黑窗口中输入python文件，查看CUDA_VISIBLE-DEVICES是否为true
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    # 如果用户指定使用CPU，则不使用GPU
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    # 如果不是进行评估操作，则将标准输出重定向到日志文件
    # 如果是进行评估操作，则将标准输出重定向到测试日志文件
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    # 打印当前的参数设置
    # Args:Namespace(arch='resnet50tp', dataset='mars', eval_step=50, evaluate=False, gamma=0.1, gpu_devices='0',
    # height=224, htri_only=False, lr=0.0003, margin=0.3, max_epoch=800, num_instances=4, pool='avg',
    # print_freq=80, save_dir='log', seed=1, seq_len=4, start_epoch=0, stepsize=200, test_batch=1, train_batch=32,
    # use_cpu=False, weight_decay=0.0005, width=112, workers=4)
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        # 设置cudnn的优化选项为True，以提高GPU计算效率
        # 在GPU上设置随机种子，保证实验的可重复性
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # 初始化数据集对象，参数通过args传递
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    # 定义训练数据集的转换操作，包括随机的2D平移、水平翻转、转换为Tensor并进行标准化处理
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 定义测试数据集的转换操作，包括调整图像大小、转换为Tensor并进行标准化处理
    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 如果使用GPU设备，设置pin_memory为True，否则为False。pin_memory用于在数据传输过程中锁定显存，提高数据传输效率。
    pin_memory = True if use_gpu else False

    '''Initializing dataset mars = > MARS
    loaded Dataset statistics:
    ------------------------------
    subset |  # ids | # tracklets
    ------------------------------
    train   | 625 | 8298
    query   | 626 | 1980
    gallery | 622 | 9330
    ------------------------------
    total | 1251 | 19608
    number of images per tracklet: 2 ~ 920, average 59.5'''

    # 定义训练数据加载器，数据来源于train数据集，每个batch采样不同的随机实例，并对数据进行转换处理。batch_size和num_workers参数通过args传递。pin_memory和drop_last参数根据是否使用GPU进行设置。
    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    # 定义查询数据加载器，数据来源于query数据集，每个batch采样密集的实例，并对数据进行转换处理。batch_size和num_workers参数通过args传递。pin_memory和drop_last参数根据是否使用GPU进行设置。shuffle参数设为False表示不进行洗牌操作。
    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    # Initializing model: resnet50tp
    # ！！！！！！！！！！！！！！！！！！！！！！！！！替换！！！！！！！！！！！！！！！！！！！！！！！！！
    print("Initializing model: {}".format(args.arch))
    if args.arch=='resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()
    best_rank1 = -np.inf
    if args.arch=='resnet503d':
        torch.backends.cudnn.benchmark = False
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        
        if args.stepsize > 0: scheduler.step()
        
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features = model(imgs)
        if args.htri_only:
            # only use hard triplet loss to train the network
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # losses.update(loss.data[0], pids.size(0))
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        # b=1, n=number of clips, s=16
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        features = model(imgs)
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features = model(imgs)
        features = features.view(n, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("---------Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------------------------")

    return cmc[0]

if __name__ == '__main__':
    main()
