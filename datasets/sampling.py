import logging
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets, transforms

# 确保日志记录器存在
logger = logging.getLogger(__name__)

def get_dataset_cached(name, root='./data'):
    """从磁盘加载指定的数据集。"""
    logger.info(f"Loading dataset '{name}' from disk...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standard normalization for CIFAR
    ])
    
    if name == 'cifar100':
        trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    elif name == 'cifar10':
        trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    logger.info(f"Dataset '{name}' loaded. Train size: {len(trainset)}, Test size: {len(testset)}")
    return trainset, testset


def get_static_datasets_indices(full_dataset: Dataset, num_clients: int, iid: bool, alpha: float = 0.5):
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    client_indices_list = [[] for _ in range(num_clients)]

    if iid:
        logger.info(f"Performing STATIC IID split for {num_clients} clients.")
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        client_indices_list = [idx.tolist() for idx in split_indices]
    else:
        logger.info(f"Performing STATIC Non-IID (Dirichlet) split with alpha={alpha} for {num_clients} clients.")
        try:
            labels = np.array(full_dataset.targets)
            num_classes = len(np.unique(labels))
        except AttributeError:
            logger.error("Dataset for Non-IID split must have a .targets attribute. Cannot perform Dirichlet split.")
            raise

        # (num_clients, num_classes)的矩阵, 每一行是该客户端对所有类别的样本比例
        label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)

        # (class, [indices])的字典
        class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

        for k in range(num_classes):
            # 打乱某个类别的所有样本索引
            np.random.shuffle(class_indices[k])
            
            # 为该类别分配样本
            proportions = label_distribution[:, k]
            proportions = (proportions / proportions.sum() * len(class_indices[k])).astype(int)
            
            # 确保所有样本都被分配
            diff = len(class_indices[k]) - proportions.sum()
            proportions[:diff] += 1
            
            start = 0
            for i in range(num_clients):
                end = start + proportions[i]
                client_indices_list[i].extend(class_indices[k][start:end])
                start = end

    logger.info(f"Data indices partitioned for {num_clients} clients.")
    return client_indices_list

def prepare_validation_loader(args, rank, logger_ref=None) -> DataLoader:
    """为每个边缘服务器准备一个独立的、不重叠的本地验证集。"""
    try:
        trainset_full, _ = get_dataset_cached(args.dataset, getattr(args, 'data_dir', './data'))
        
        num_total_samples = len(trainset_full)
        num_edges = getattr(args, 'num_edges', 1)
        val_ratio_total = 0.1 # 假设总验证集占10%
        num_val_samples_per_edge = int((val_ratio_total * num_total_samples) / num_edges)
        total_val_samples = num_val_samples_per_edge * num_edges
        num_train_samples = num_total_samples - total_val_samples

        if num_train_samples < 0:
            raise ValueError("Validation set ratio is too large, no data left for training.")

        generator = torch.Generator().manual_seed(getattr(args, 'seed', 42))
        split_lengths = [num_train_samples] + [num_val_samples_per_edge] * num_edges
        all_subsets = torch.utils.data.random_split(trainset_full, split_lengths, generator=generator)

        validation_subset = all_subsets[1 + rank]
        
        if logger_ref:
            logger_ref.info(f"Created a local validation set with {len(validation_subset)} samples for reward calculation.")
            
        return DataLoader(validation_subset, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        if logger_ref:
            logger_ref.error(f"Failed to prepare validation loader: {e}", exc_info=True)
        raise