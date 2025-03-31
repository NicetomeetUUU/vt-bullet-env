#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云数据集查看工具

用于可视化和分析处理后的点云数据集。
支持查看不同采样策略和存储模式下的数据集结构和点云特征。
"""

import os
import sys
import numpy as np
import open3d as o3d
import json
import h5py
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_viewer.log')
    ]
)
logger = logging.getLogger('dataset_viewer')

class DatasetViewer:
    """点云数据集查看器"""
    
    def __init__(self, dataset_dir):
        """初始化数据集查看器
        
        参数:
            dataset_dir: 处理后的数据集目录
        """
        self.dataset_dir = dataset_dir
        
        # 检查数据集目录是否存在
        if not os.path.exists(dataset_dir):
            logger.error(f"数据集目录不存在: {dataset_dir}")
            raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
        
        # 检查数据集结构
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.val_dir = os.path.join(dataset_dir, 'val')
        self.test_dir = os.path.join(dataset_dir, 'test')
        
        # 确定存储模式
        self.storage_mode = self._determine_storage_mode()
        logger.info(f"检测到存储模式: {self.storage_mode}")
    
    def _determine_storage_mode(self):
        """确定数据集的存储模式"""
        # 检查是否为单一数组存储
        if (os.path.exists(os.path.join(self.train_dir, 'dataset.h5')) or
            os.path.exists(os.path.join(self.train_dir, 'dataset.npz'))):
            return 'single_array'
        
        # 检查是否为分层存储
        if os.path.exists(os.path.join(self.train_dir, 'metadata.json')):
            return 'hierarchical'
        
        # 默认为分散存储
        return 'scattered'
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'storage_mode': self.storage_mode,
            'train_samples': self._count_samples('train'),
            'val_samples': self._count_samples('val'),
            'test_samples': self._count_samples('test'),
            'total_samples': 0,
            'point_cloud_types': self._get_point_cloud_types(),
            'point_cloud_stats': self._get_point_cloud_stats()
        }
        
        info['total_samples'] = info['train_samples'] + info['val_samples'] + info['test_samples']
        
        return info
    
    def _count_samples(self, split):
        """计算指定划分中的样本数量"""
        split_dir = getattr(self, f"{split}_dir")
        
        if self.storage_mode == 'single_array':
            # 对于单一数组存储，检查h5或npz文件
            h5_path = os.path.join(split_dir, 'dataset.h5')
            npz_path = os.path.join(split_dir, 'dataset.npz')
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    return len(f['metadata'])
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                return len(data['metadata'])
            else:
                return 0
        
        elif self.storage_mode == 'hierarchical':
            # 对于分层存储，读取metadata.json
            metadata_path = os.path.join(split_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return len(metadata)
            else:
                return 0
        
        else:  # scattered
            # 对于分散存储，计算样本文件夹数量
            sample_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            return len(sample_dirs)
    
    def _get_point_cloud_types(self):
        """获取数据集中的点云类型"""
        if self.storage_mode == 'single_array':
            # 对于单一数组存储，检查h5或npz文件
            h5_path = os.path.join(self.train_dir, 'dataset.h5')
            npz_path = os.path.join(self.train_dir, 'dataset.npz')
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    return list(f.keys())[:-1]  # 排除metadata
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                return [k for k in data.keys() if k != 'metadata']
            else:
                return []
        
        elif self.storage_mode == 'hierarchical':
            # 对于分层存储，读取metadata.json
            metadata_path = os.path.join(self.train_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if len(metadata) > 0:
                    return list(metadata[0]['point_clouds'].keys())
                else:
                    return []
            else:
                return []
        
        else:  # scattered
            # 对于分散存储，查找第一个样本文件夹
            sample_dirs = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
            if len(sample_dirs) > 0:
                first_sample = os.path.join(self.train_dir, sample_dirs[0])
                return [f.split('.')[0] for f in os.listdir(first_sample) if f.endswith('.npy')]
            else:
                return []
    
    def _get_point_cloud_stats(self):
        """获取点云统计信息"""
        point_cloud_types = self._get_point_cloud_types()
        stats = {pc_type: {'min_points': float('inf'), 'max_points': 0, 'avg_points': 0} for pc_type in point_cloud_types}
        
        # 随机采样10个样本进行统计
        samples = self._get_random_samples(10)
        
        for sample in samples:
            for pc_type in point_cloud_types:
                if pc_type in sample:
                    points = sample[pc_type]
                    num_points = len(points)
                    
                    stats[pc_type]['min_points'] = min(stats[pc_type]['min_points'], num_points)
                    stats[pc_type]['max_points'] = max(stats[pc_type]['max_points'], num_points)
                    stats[pc_type]['avg_points'] += num_points / len(samples)
        
        return stats
    
    def _get_random_samples(self, num_samples):
        """获取随机样本"""
        samples = []
        
        if self.storage_mode == 'single_array':
            # 从单一数组存储中获取随机样本
            h5_path = os.path.join(self.train_dir, 'dataset.h5')
            npz_path = os.path.join(self.train_dir, 'dataset.npz')
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    metadata = f['metadata'][:]
                    indices = np.random.choice(len(metadata), min(num_samples, len(metadata)), replace=False)
                    
                    for idx in indices:
                        sample = {}
                        for pc_type in self._get_point_cloud_types():
                            sample[pc_type] = f[pc_type][idx]
                        samples.append(sample)
            
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                metadata = data['metadata']
                indices = np.random.choice(len(metadata), min(num_samples, len(metadata)), replace=False)
                
                for idx in indices:
                    sample = {}
                    for pc_type in self._get_point_cloud_types():
                        sample[pc_type] = data[pc_type][idx]
                    samples.append(sample)
        
        elif self.storage_mode == 'hierarchical':
            # 从分层存储中获取随机样本
            metadata_path = os.path.join(self.train_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                indices = np.random.choice(len(metadata), min(num_samples, len(metadata)), replace=False)
                
                for idx in indices:
                    sample_info = metadata[idx]
                    sample = {}
                    
                    for pc_type, pc_path in sample_info['point_clouds'].items():
                        full_path = os.path.join(self.train_dir, pc_path)
                        if os.path.exists(full_path):
                            sample[pc_type] = np.load(full_path)
                    
                    samples.append(sample)
        
        else:  # scattered
            # 从分散存储中获取随机样本
            sample_dirs = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
            if len(sample_dirs) > 0:
                selected_dirs = np.random.choice(sample_dirs, min(num_samples, len(sample_dirs)), replace=False)
                
                for dir_name in selected_dirs:
                    sample_dir = os.path.join(self.train_dir, dir_name)
                    sample = {}
                    
                    for pc_type in self._get_point_cloud_types():
                        pc_path = os.path.join(sample_dir, f"{pc_type}.npy")
                        if os.path.exists(pc_path):
                            sample[pc_type] = np.load(pc_path)
                    
                    samples.append(sample)
        
        return samples
    
    def visualize_sample(self, sample_idx=None, split='train'):
        """可视化样本点云
        
        参数:
            sample_idx: 样本索引，如果为None则随机选择
            split: 数据集划分，'train', 'val' 或 'test'
        """
        split_dir = getattr(self, f"{split}_dir")
        
        # 获取样本
        if sample_idx is None:
            # 随机选择一个样本
            samples = self._get_random_samples(1)
            if len(samples) == 0:
                logger.error("无法获取随机样本")
                return
            sample = samples[0]
        else:
            # 获取指定索引的样本
            sample = self._get_sample_by_idx(sample_idx, split)
            if sample is None:
                logger.error(f"无法获取样本 {sample_idx}")
                return
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"点云样本 - {split} - {sample_idx if sample_idx is not None else '随机'}")
        
        # 为每种点云类型创建不同颜色的点云
        colors = {
            'tactile_left': [1, 0, 0],  # 红色
            'tactile_right': [0, 1, 0],  # 绿色
            'cropped_object': [0, 0, 1],  # 蓝色
            'cropped_surface': [1, 1, 0],  # 黄色
            'cropped_surface_context': [1, 0, 1]  # 紫色
        }
        
        # 添加点云到可视化器
        for pc_type, points in sample.items():
            if pc_type in colors and len(points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # 设置颜色
                color = colors.get(pc_type, [0.5, 0.5, 0.5])
                pcd.paint_uniform_color(color)
                
                # 添加到可视化器
                vis.add_geometry(pcd)
        
        # 设置视角
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]  # 深灰色背景
        
        # 添加坐标系
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        # 运行可视化器
        vis.run()
        vis.destroy_window()
    
    def _get_sample_by_idx(self, sample_idx, split='train'):
        """获取指定索引的样本"""
        split_dir = getattr(self, f"{split}_dir")
        
        if self.storage_mode == 'single_array':
            # 从单一数组存储中获取样本
            h5_path = os.path.join(split_dir, 'dataset.h5')
            npz_path = os.path.join(split_dir, 'dataset.npz')
            
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    if sample_idx >= len(f['metadata']):
                        return None
                    
                    sample = {}
                    for pc_type in self._get_point_cloud_types():
                        sample[pc_type] = f[pc_type][sample_idx]
                    return sample
            
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                metadata = data['metadata']
                
                if sample_idx >= len(metadata):
                    return None
                
                sample = {}
                for pc_type in self._get_point_cloud_types():
                    sample[pc_type] = data[pc_type][sample_idx]
                return sample
        
        elif self.storage_mode == 'hierarchical':
            # 从分层存储中获取样本
            metadata_path = os.path.join(split_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if sample_idx >= len(metadata):
                    return None
                
                sample_info = metadata[sample_idx]
                sample = {}
                
                for pc_type, pc_path in sample_info['point_clouds'].items():
                    full_path = os.path.join(split_dir, pc_path)
                    if os.path.exists(full_path):
                        sample[pc_type] = np.load(full_path)
                
                return sample
        
        else:  # scattered
            # 从分散存储中获取样本
            sample_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
            
            if sample_idx >= len(sample_dirs):
                return None
            
            sample_dir = os.path.join(split_dir, sample_dirs[sample_idx])
            sample = {}
            
            for pc_type in self._get_point_cloud_types():
                pc_path = os.path.join(sample_dir, f"{pc_type}.npy")
                if os.path.exists(pc_path):
                    sample[pc_type] = np.load(pc_path)
            
            return sample
    
    def plot_point_cloud_distribution(self):
        """绘制点云分布统计图"""
        # 获取点云类型
        point_cloud_types = self._get_point_cloud_types()
        
        # 随机采样30个样本进行统计
        samples = self._get_random_samples(30)
        
        # 收集每种点云类型的点数
        point_counts = {pc_type: [] for pc_type in point_cloud_types}
        
        for sample in samples:
            for pc_type in point_cloud_types:
                if pc_type in sample:
                    point_counts[pc_type].append(len(sample[pc_type]))
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        plt.boxplot([point_counts[pc_type] for pc_type in point_cloud_types], labels=point_cloud_types)
        plt.title('点云点数分布')
        plt.ylabel('点数')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.savefig('point_cloud_distribution.png')
        logger.info("已保存点云分布图表到 point_cloud_distribution.png")
        
        # 显示图表
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="点云数据集查看工具")
    parser.add_argument('--dataset_dir', type=str, default='../processed_dataset', help='处理后的数据集目录')
    parser.add_argument('--visualize', action='store_true', help='可视化随机样本')
    parser.add_argument('--sample_idx', type=int, default=None, help='要可视化的样本索引')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='数据集划分')
    parser.add_argument('--plot_distribution', action='store_true', help='绘制点云分布统计图')
    
    args = parser.parse_args()
    
    # 创建数据集查看器
    viewer = DatasetViewer(args.dataset_dir)
    
    # 获取并显示数据集信息
    info = viewer.get_dataset_info()
    logger.info("数据集信息:")
    logger.info(f"  存储模式: {info['storage_mode']}")
    logger.info(f"  总样本数: {info['total_samples']}")
    logger.info(f"  训练集: {info['train_samples']} 样本")
    logger.info(f"  验证集: {info['val_samples']} 样本")
    logger.info(f"  测试集: {info['test_samples']} 样本")
    logger.info(f"  点云类型: {', '.join(info['point_cloud_types'])}")
    
    # 显示点云统计信息
    logger.info("点云统计信息:")
    for pc_type, stats in info['point_cloud_stats'].items():
        logger.info(f"  {pc_type}:")
        logger.info(f"    最小点数: {stats['min_points']}")
        logger.info(f"    最大点数: {stats['max_points']}")
        logger.info(f"    平均点数: {stats['avg_points']:.2f}")
    
    # 可视化样本
    if args.visualize:
        viewer.visualize_sample(args.sample_idx, args.split)
    
    # 绘制点云分布统计图
    if args.plot_distribution:
        viewer.plot_point_cloud_distribution()

if __name__ == '__main__':
    main()
