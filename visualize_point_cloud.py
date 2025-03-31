#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云数据集可视化工具

专门用于可视化处理后的点云数据集，支持.h5格式。
"""

import os
import sys
import numpy as np
import open3d as o3d
import h5py
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('visualize_point_cloud.log')
    ]
)
logger = logging.getLogger('visualize_point_cloud')

class PointCloudVisualizer:
    """点云可视化器"""
    
    def __init__(self, dataset_path):
        """初始化点云可视化器
        
        参数:
            dataset_path: 数据集文件路径（.h5文件）
        """
        self.dataset_path = dataset_path
        
        # 检查数据集文件是否存在
        if not os.path.exists(dataset_path):
            logger.error(f"数据集文件不存在: {dataset_path}")
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 加载数据集
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        """加载数据集"""
        logger.info(f"加载数据集: {self.dataset_path}")
        
        try:
            # 打开H5文件
            h5_file = h5py.File(self.dataset_path, 'r')
            
            # 获取数据集信息
            keys = list(h5_file.keys())
            logger.info(f"数据集包含以下键: {keys}")
            
            # 检查数据集结构
            for key in keys:
                if key in h5_file:
                    item = h5_file[key]
                    if isinstance(item, h5py.Dataset):
                        logger.info(f"  {key}: 形状 {item.shape}, 类型 {item.dtype}")
                    else:
                        logger.info(f"  {key}: 组")
            
            # 返回打开的H5文件对象
            return h5_file
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'file_path': self.dataset_path,
            'keys': list(self.dataset.keys()),
            'sample_count': 0,
            'point_cloud_types': []
        }
        
        # 获取样本数量
        if 'metadata' in self.dataset:
            info['sample_count'] = len(self.dataset['metadata'])
        
        # 获取点云类型
        for key in info['keys']:
            if key != 'metadata' and isinstance(self.dataset[key], h5py.Dataset):
                info['point_cloud_types'].append(key)
        
        # 获取点云统计信息
        info['point_cloud_stats'] = {}
        for pc_type in info['point_cloud_types']:
            if pc_type in self.dataset:
                points_shape = self.dataset[pc_type].shape
                if len(points_shape) >= 2:
                    # 检查第一个样本的点云数据
                    try:
                        sample_points = self.dataset[pc_type][0]
                        info['point_cloud_stats'][pc_type] = {
                            'shape': points_shape,
                            'min_points': len(sample_points),
                            'max_points': len(sample_points),
                            'avg_points': len(sample_points),
                            'sample_shape': sample_points.shape
                        }
                        logger.info(f"点云类型 {pc_type}: 样本形状 {sample_points.shape}")
                    except Exception as e:
                        logger.error(f"获取 {pc_type} 点云统计信息失败: {e}")
        
        return info
    
    def visualize_sample(self, sample_idx=0):
        """可视化样本点云
        
        参数:
            sample_idx: 样本索引
        """
        # 获取数据集信息
        info = self.get_dataset_info()
        
        # 检查样本索引是否有效
        if sample_idx >= info['sample_count']:
            logger.error(f"样本索引 {sample_idx} 超出范围 (0-{info['sample_count']-1})")
            return
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"点云样本 - 索引 {sample_idx}")
        
        # 为每种点云类型创建不同颜色的点云
        colors = {
            'tactile_left': [1, 0, 0],  # 红色
            'tactile_right': [0, 1, 0],  # 绿色
            'cropped_object': [0, 0, 1],  # 蓝色
            'cropped_surface': [1, 1, 0],  # 黄色
            'cropped_surface_context': [1, 0, 1],  # 紫色
            'camera': [0, 1, 1],  # 青色
            'object': [0.5, 0.5, 0.5]  # 灰色
        }
        
        # 获取元数据
        if 'metadata' in self.dataset:
            try:
                metadata = self.dataset['metadata'][sample_idx]
                if isinstance(metadata, np.ndarray):
                    if metadata.dtype.names:
                        metadata_dict = {name: metadata[name] for name in metadata.dtype.names}
                        logger.info(f"样本 {sample_idx} 元数据: {metadata_dict}")
                    else:
                        logger.info(f"样本 {sample_idx} 元数据: {metadata}")
            except Exception as e:
                logger.error(f"获取元数据失败: {e}")
        
        # 添加点云到可视化器
        for pc_type in info['point_cloud_types']:
            try:
                points = self.dataset[pc_type][sample_idx]
                
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 设置颜色
                    color = colors.get(pc_type, [0.5, 0.5, 0.5])
                    pcd.paint_uniform_color(color)
                    
                    # 添加到可视化器
                    vis.add_geometry(pcd)
                    
                    logger.info(f"添加 {pc_type} 点云，点数: {len(points)}")
            except Exception as e:
                logger.error(f"可视化 {pc_type} 点云失败: {e}")
        
        # 设置视角
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]  # 深灰色背景
        
        # 添加坐标系
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        
        # 设置初始视角
        opt = vis.get_view_control()
        opt.set_zoom(0.8)
        
        # 运行可视化器
        vis.run()
        vis.destroy_window()
    
    def visualize_all_types(self, sample_idx=0):
        """分别可视化每种点云类型
        
        参数:
            sample_idx: 样本索引
        """
        # 获取数据集信息
        info = self.get_dataset_info()
        
        # 检查样本索引是否有效
        if sample_idx >= info['sample_count']:
            logger.error(f"样本索引 {sample_idx} 超出范围 (0-{info['sample_count']-1})")
            return
        
        # 为每种点云类型创建单独的可视化窗口
        for pc_type in info['point_cloud_types']:
            try:
                points = self.dataset[pc_type][sample_idx]
                
                if len(points) > 0:
                    # 创建点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 设置随机颜色
                    pcd.paint_uniform_color([np.random.random(), np.random.random(), np.random.random()])
                    
                    # 可视化
                    logger.info(f"可视化 {pc_type} 点云，点数: {len(points)}")
                    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)], 
                                                     window_name=f"{pc_type} - 样本 {sample_idx}")
            except Exception as e:
                logger.error(f"可视化 {pc_type} 点云失败: {e}")
    
    def plot_point_distribution(self, sample_idx=0):
        """绘制点云空间分布图
        
        参数:
            sample_idx: 样本索引
        """
        # 获取数据集信息
        info = self.get_dataset_info()
        
        # 检查样本索引是否有效
        if sample_idx >= info['sample_count']:
            logger.error(f"样本索引 {sample_idx} 超出范围 (0-{info['sample_count']-1})")
            return
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        
        # 为每种点云类型创建子图
        n_types = len(info['point_cloud_types'])
        rows = (n_types + 2) // 3  # 向上取整
        cols = min(3, n_types)
        
        for i, pc_type in enumerate(info['point_cloud_types']):
            try:
                points = self.dataset[pc_type][sample_idx]
                
                if len(points) > 0:
                    # 创建3D子图
                    ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                    
                    # 绘制散点图
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)
                    
                    # 设置标题和坐标轴标签
                    ax.set_title(f"{pc_type} ({len(points)} 点)")
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # 设置坐标轴范围
                    max_range = np.max(np.abs(points))
                    ax.set_xlim(-max_range, max_range)
                    ax.set_ylim(-max_range, max_range)
                    ax.set_zlim(-max_range, max_range)
            except Exception as e:
                logger.error(f"绘制 {pc_type} 点云分布失败: {e}")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'point_distribution_sample_{sample_idx}.png')
        logger.info(f"已保存点云分布图表到 point_distribution_sample_{sample_idx}.png")
        
        # 显示图表
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="点云数据集可视化工具")
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集文件路径（.h5文件）')
    parser.add_argument('--sample_idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--visualize', action='store_true', help='可视化所有点云类型')
    parser.add_argument('--visualize_separate', action='store_true', help='分别可视化每种点云类型')
    parser.add_argument('--plot', action='store_true', help='绘制点云空间分布图')
    
    args = parser.parse_args()
    
    # 创建点云可视化器
    visualizer = PointCloudVisualizer(args.dataset_path)
    
    # 获取并显示数据集信息
    info = visualizer.get_dataset_info()
    logger.info("数据集信息:")
    logger.info(f"  文件路径: {info['file_path']}")
    logger.info(f"  样本数量: {info['sample_count']}")
    logger.info(f"  点云类型: {', '.join(info['point_cloud_types'])}")
    
    # 显示点云统计信息
    logger.info("点云统计信息:")
    for pc_type, stats in info['point_cloud_stats'].items():
        logger.info(f"  {pc_type}:")
        logger.info(f"    形状: {stats['shape']}")
        logger.info(f"    点数: {stats['avg_points']}")
    
    # 可视化样本
    if args.visualize:
        visualizer.visualize_sample(args.sample_idx)
    
    # 分别可视化每种点云类型
    if args.visualize_separate:
        visualizer.visualize_all_types(args.sample_idx)
    
    # 绘制点云空间分布图
    if args.plot:
        visualizer.plot_point_distribution(args.sample_idx)

if __name__ == '__main__':
    main()
