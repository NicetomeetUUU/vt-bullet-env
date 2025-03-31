#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import shutil
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

# 定义数据集路径
DATASET_DIR = "/home/iccd-simulator/code/vt-bullet-env/dataset"
OUTPUT_DIR = "/home/iccd-simulator/code/vt-bullet-env/processed_dataset"

# 定义点云采样数量
TACTILE_POINTS = 512  # 每个触觉点云的点数
OBJECT_POINTS = 1024  # 物体点云的点数
LOCAL_SURFACE_POINTS = 2048  # 局部表面点云的点数
GLOBAL_SURFACE_POINTS = 4096  # 全局表面点云的点数

# 定义数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def load_point_cloud(file_path):
    """
    加载点云文件
    
    参数:
        file_path: 点云文件路径
    
    返回:
        点云的numpy数组表示 (N, 3)
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"警告: 点云为空 {file_path}")
            return None
        return np.asarray(pcd.points)
    except Exception as e:
        print(f"错误: 无法加载点云 {file_path}: {str(e)}")
        return None

def normalize_point_cloud(points):
    """
    标准化点云：将点云居中并缩放到[-1, 1]范围内
    
    参数:
        points: 点云数组 (N, 3)
        
    返回:
        标准化后的点云数组 (N, 3)
    """
    if points is None or len(points) == 0:
        return None
    
    # 计算质心
    centroid = np.mean(points, axis=0)
    
    # 居中
    points_centered = points - centroid
    
    # 计算最大距离
    max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    
    # 缩放到[-1, 1]
    points_normalized = points_centered / max_dist
    
    return points_normalized

def sample_point_cloud(points, n_samples, method='fps'):
    """
    采样点云到固定数量的点
    
    参数:
        points: 点云数组 (N, 3)
        n_samples: 采样后的点数
        method: 采样方法，'fps'表示最远点采样，'random'表示随机采样
        
    返回:
        采样后的点云数组 (n_samples, 3)
    """
    if points is None:
        return None
    
    n_points = len(points)
    
    # 如果点数已经小于等于目标采样数，则通过复制点来达到目标数量
    if n_points <= n_samples:
        # 计算需要复制的次数
        repeat_times = int(np.ceil(n_samples / n_points))
        # 复制点云
        repeated_points = np.tile(points, (repeat_times, 1))
        # 截取需要的点数
        return repeated_points[:n_samples]
    
    if method == 'random':
        # 随机采样
        indices = np.random.choice(n_points, n_samples, replace=False)
        return points[indices]
    
    elif method == 'fps':
        # 最远点采样
        fps_indices = farthest_point_sampling(points, n_samples)
        return points[fps_indices]
    
    else:
        raise ValueError(f"未知的采样方法: {method}")

def farthest_point_sampling(points, n_samples):
    """
    实现最远点采样算法
    
    参数:
        points: 点云数组 (N, 3)
        n_samples: 采样后的点数
        
    返回:
        采样点的索引数组 (n_samples,)
    """
    n_points = len(points)
    selected_indices = np.zeros(n_samples, dtype=np.int32)
    distances = np.ones(n_points) * 1e10
    
    # 随机选择第一个点
    farthest = np.random.randint(0, n_points)
    
    # 迭代选择剩余的点
    for i in range(n_samples):
        selected_indices[i] = farthest
        centroid = points[farthest]
        
        # 计算所有点到当前点的距离
        dist = np.sum((points - centroid) ** 2, axis=1)
        
        # 更新最小距离
        distances = np.minimum(distances, dist)
        
        # 选择具有最大最小距离的点作为下一个点
        farthest = np.argmax(distances)
    
    return selected_indices

def process_grasp_data(grasp_dir, obj_id, metadata):
    """
    处理单个抓取数据，包括加载、标准化和采样点云
    
    参数:
        grasp_dir: 抓取数据目录
        obj_id: 物体ID
        metadata: 元数据字典，用于记录处理信息
        
    返回:
        处理后的数据字典和更新的元数据
    """
    # 加载点云数据
    tactile_left_file = os.path.join(grasp_dir, f"{obj_id}_tactile_left.pcd")
    tactile_right_file = os.path.join(grasp_dir, f"{obj_id}_tactile_right.pcd")
    cropped_object_file = os.path.join(grasp_dir, f"{obj_id}_cropped_object.pcd")
    cropped_surface_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface.pcd")
    cropped_surface_context_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface_context.pcd")
    
    # 加载抓取位姿信息
    grasp_pose_file = os.path.join(grasp_dir, f"{obj_id}_tactile_grasp_pose.json")
    grasp_pose = None
    if os.path.exists(grasp_pose_file):
        with open(grasp_pose_file, 'r') as f:
            grasp_pose = json.load(f)
    
    # 加载点云
    tactile_left = load_point_cloud(tactile_left_file)
    tactile_right = load_point_cloud(tactile_right_file)
    cropped_object = load_point_cloud(cropped_object_file)
    cropped_surface = load_point_cloud(cropped_surface_file)
    cropped_surface_context = load_point_cloud(cropped_surface_context_file)
    
    # 检查是否所有必要的点云都存在
    if tactile_left is None or tactile_right is None or cropped_object is None or \
       cropped_surface is None or cropped_surface_context is None:
        print(f"警告: 缺少必要的点云数据 {grasp_dir}")
        return None, metadata
    
    # 标准化点云
    tactile_left_norm = normalize_point_cloud(tactile_left)
    tactile_right_norm = normalize_point_cloud(tactile_right)
    cropped_object_norm = normalize_point_cloud(cropped_object)
    cropped_surface_norm = normalize_point_cloud(cropped_surface)
    cropped_surface_context_norm = normalize_point_cloud(cropped_surface_context)
    
    # 采样点云
    tactile_left_sampled = sample_point_cloud(tactile_left_norm, TACTILE_POINTS)
    tactile_right_sampled = sample_point_cloud(tactile_right_norm, TACTILE_POINTS)
    cropped_object_sampled = sample_point_cloud(cropped_object_norm, OBJECT_POINTS)
    cropped_surface_sampled = sample_point_cloud(cropped_surface_norm, LOCAL_SURFACE_POINTS)
    cropped_surface_context_sampled = sample_point_cloud(cropped_surface_context_norm, GLOBAL_SURFACE_POINTS)
    
    # 合并触觉点云和物体点云作为输入
    input_points = np.concatenate([tactile_left_sampled, tactile_right_sampled, cropped_object_sampled], axis=0)
    
    # 创建数据字典
    data = {
        'input': input_points,
        'gt_local': cropped_surface_sampled,
        'gt_global': cropped_surface_context_sampled,
        'grasp_pose': grasp_pose
    }
    
    # 更新元数据
    metadata_entry = {
        'object_id': obj_id,
        'pose_id': os.path.basename(os.path.dirname(grasp_dir)),
        'grasp_id': os.path.basename(grasp_dir),
        'original_path': grasp_dir,
        'num_points': {
            'tactile_left': len(tactile_left),
            'tactile_right': len(tactile_right),
            'object': len(cropped_object),
            'surface_local': len(cropped_surface),
            'surface_global': len(cropped_surface_context)
        },
        'sampled_points': {
            'tactile_left': TACTILE_POINTS,
            'tactile_right': TACTILE_POINTS,
            'object': OBJECT_POINTS,
            'surface_local': LOCAL_SURFACE_POINTS,
            'surface_global': GLOBAL_SURFACE_POINTS
        }
    }
    
    return data, metadata_entry

def split_dataset_by_object(obj_dirs, output_dir):
    """
    按物体ID划分数据集
    
    参数:
        obj_dirs: 物体目录列表
        output_dir: 输出目录
    
    返回:
        训练集、验证集和测试集的物体目录列表
    """
    np.random.shuffle(obj_dirs)  # 随机打乱物体顺序
    
    n_objects = len(obj_dirs)
    n_train = int(n_objects * TRAIN_RATIO)
    n_val = int(n_objects * VAL_RATIO)
    
    train_dirs = obj_dirs[:n_train]
    val_dirs = obj_dirs[n_train:n_train+n_val]
    test_dirs = obj_dirs[n_train+n_val:]
    
    print(f"数据集划分: 训练集 {len(train_dirs)} 物体, 验证集 {len(val_dirs)} 物体, 测试集 {len(test_dirs)} 物体")
    
    return train_dirs, val_dirs, test_dirs

def process_dataset(dataset_dir, output_dir, split_method='by_object'):
    """
    处理整个数据集
    
    参数:
        dataset_dir: 数据集根目录
        output_dir: 输出目录
        split_method: 数据集划分方法，'by_object'表示按物体划分
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'gt_local'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'gt_global'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'gt_local'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'gt_global'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'gt_local'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'gt_global'), exist_ok=True)
    
    # 获取所有物体目录
    obj_dirs = []
    for i in range(0, 88):
        obj_dir = os.path.join(dataset_dir, f"{i:03d}")
        if os.path.exists(obj_dir):
            obj_dirs.append(obj_dir)
    
    if not obj_dirs:
        print(f"错误: 未找到物体目录 {dataset_dir}")
        return
    
    # 划分数据集
    if split_method == 'by_object':
        train_dirs, val_dirs, test_dirs = split_dataset_by_object(obj_dirs, output_dir)
    else:
        raise ValueError(f"未知的数据集划分方法: {split_method}")
    
    # 初始化元数据
    train_metadata = {}
    val_metadata = {}
    test_metadata = {}
    
    # 处理训练集
    train_index = 0
    for obj_dir in tqdm(train_dirs, desc="处理训练集"):
        obj_id = os.path.basename(obj_dir)
        train_index = process_object_dir(obj_dir, obj_id, 'train', train_index, train_metadata, output_dir)
    
    # 处理验证集
    val_index = 0
    for obj_dir in tqdm(val_dirs, desc="处理验证集"):
        obj_id = os.path.basename(obj_dir)
        val_index = process_object_dir(obj_dir, obj_id, 'val', val_index, val_metadata, output_dir)
    
    # 处理测试集
    test_index = 0
    for obj_dir in tqdm(test_dirs, desc="处理测试集"):
        obj_id = os.path.basename(obj_dir)
        test_index = process_object_dir(obj_dir, obj_id, 'test', test_index, test_metadata, output_dir)
    
    # 保存元数据
    with open(os.path.join(output_dir, 'train_metadata.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(os.path.join(output_dir, 'val_metadata.json'), 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    with open(os.path.join(output_dir, 'test_metadata.json'), 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"数据集处理完成: 训练集 {train_index} 样本, 验证集 {val_index} 样本, 测试集 {test_index} 样本")

def process_object_dir(obj_dir, obj_id, split, index, metadata, output_dir):
    """
    处理单个物体目录下的所有位姿和抓取数据
    
    参数:
        obj_dir: 物体目录
        obj_id: 物体ID
        split: 数据集划分，'train', 'val'或'test'
        index: 当前索引
        metadata: 元数据字典
        output_dir: 输出目录
        
    返回:
        更新后的索引
    """
    # 获取所有位姿目录
    pose_dirs = []
    for i in range(0, 24):
        pose_dir = os.path.join(obj_dir, f"pose_{i:03d}")
        if os.path.exists(pose_dir):
            pose_dirs.append(pose_dir)
    
    # 处理每个位姿目录
    for pose_dir in pose_dirs:
        # 查找所有抓取目录
        grasp_pattern = os.path.join(pose_dir, f"*_grasp_*")
        grasp_dirs = sorted(glob.glob(grasp_pattern))
        
        # 处理每个抓取目录
        for grasp_dir in grasp_dirs:
            # 处理抓取数据
            data, metadata_entry = process_grasp_data(grasp_dir, obj_id, metadata)
            
            if data is not None:
                # 保存数据
                np.save(os.path.join(output_dir, split, 'input', f"{index:06d}.npy"), data['input'])
                np.save(os.path.join(output_dir, split, 'gt_local', f"{index:06d}.npy"), data['gt_local'])
                np.save(os.path.join(output_dir, split, 'gt_global', f"{index:06d}.npy"), data['gt_global'])
                
                # 更新元数据
                metadata[f"{index:06d}"] = metadata_entry
                
                # 更新索引
                index += 1
    
    return index

def create_data_loader_example():
    """
    创建PyTorch数据加载器的示例代码
    """
    example_code = """
# PyTorch数据加载器示例
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class PointCloudCompletionDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加载元数据
        with open(os.path.join(data_dir, f'{split}_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.sample_ids = list(self.metadata.keys())
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 加载输入和GT点云
        input_path = os.path.join(self.data_dir, self.split, 'input', f'{sample_id}.npy')
        gt_local_path = os.path.join(self.data_dir, self.split, 'gt_local', f'{sample_id}.npy')
        gt_global_path = os.path.join(self.data_dir, self.split, 'gt_global', f'{sample_id}.npy')
        
        input_pc = np.load(input_path)
        gt_local = np.load(gt_local_path)
        gt_global = np.load(gt_global_path)
        
        # 应用变换
        if self.transform:
            input_pc = self.transform(input_pc)
            gt_local = self.transform(gt_local)
            gt_global = self.transform(gt_global)
        
        # 转换为PyTorch张量
        input_pc = torch.from_numpy(input_pc).float()
        gt_local = torch.from_numpy(gt_local).float()
        gt_global = torch.from_numpy(gt_global).float()
        
        return {
            'input': input_pc,
            'gt_local': gt_local,
            'gt_global': gt_global,
            'metadata': self.metadata[sample_id]
        }

# 数据增强
class PointCloudTransform:
    def __init__(self, rotate_z=True, jitter=0.01, scale=0.1):
        self.rotate_z = rotate_z
        self.jitter = jitter
        self.scale = scale
    
    def __call__(self, points):
        # 随机旋转（绕Z轴）
        if self.rotate_z:
            angle = np.random.uniform() * 2 * np.pi
            cos_theta, sin_theta = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            points = np.matmul(points, rotation_matrix)
        
        # 随机抖动
        if self.jitter > 0:
            noise = np.random.normal(0, self.jitter, points.shape)
            points += noise
        
        # 随机缩放
        if self.scale > 0:
            scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale)
            points *= scale_factor
        
        return points

# 创建数据加载器
def create_dataloader(data_dir, batch_size=32, num_workers=4):
    train_transform = PointCloudTransform(rotate_z=True, jitter=0.01, scale=0.1)
    test_transform = None  # 测试时不使用数据增强
    
    train_dataset = PointCloudCompletionDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = PointCloudCompletionDataset(
        data_dir=data_dir,
        split='val',
        transform=test_transform
    )
    
    test_dataset = PointCloudCompletionDataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
"""
    
    print("PyTorch数据加载器示例代码:")
    print(example_code)

def main():
    parser = argparse.ArgumentParser(description="点云数据处理工具")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--split_method", type=str, default="by_object", choices=["by_object"], help="数据集划分方法")
    parser.add_argument("--example", action="store_true", help="显示数据加载器示例代码")
    
    args = parser.parse_args()
    
    if args.example:
        create_data_loader_example()
        return
    
    print(f"处理数据集: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据集划分方法: {args.split_method}")
    
    # 处理数据集
    process_dataset(args.dataset_dir, args.output_dir, args.split_method)

if __name__ == "__main__":
    main()