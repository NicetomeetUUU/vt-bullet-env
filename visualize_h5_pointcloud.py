#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
H5点云数据集可视化工具
专门用于可视化处理后的H5格式点云数据集
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
        logging.FileHandler('visualize_h5_pointcloud.log')
    ]
)
logger = logging.getLogger('visualize_h5_pointcloud')

class H5PointCloudVisualizer:
    """H5点云可视化器"""
    
    def __init__(self, h5_path):
        """初始化H5点云可视化器
        
        参数:
            h5_path: H5文件路径
        """
        self.h5_path = h5_path
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            logger.error(f"H5文件不存在: {h5_path}")
            raise FileNotFoundError(f"H5文件不存在: {h5_path}")
        
        # 加载H5文件
        self.h5_file = h5py.File(h5_path, 'r')
        
        # 分析H5文件结构
        self.analyze_structure()
    
    def analyze_structure(self):
        """分析H5文件结构"""
        logger.info(f"分析H5文件结构: {self.h5_path}")
        
        # 获取顶层键
        self.top_keys = list(self.h5_file.keys())
        logger.info(f"顶层键: {self.top_keys}")
        
        # 检查元数据
        if 'metadata' in self.h5_file:
            self.sample_count = len(self.h5_file['metadata'])
            logger.info(f"样本数量: {self.sample_count}")
            
            # 获取第一个样本的元数据
            if self.sample_count > 0:
                metadata = self.h5_file['metadata'][0]
                if metadata.dtype.names:
                    metadata_dict = {name: metadata[name] for name in metadata.dtype.names}
                    logger.info(f"元数据示例: {metadata_dict}")
        else:
            self.sample_count = 0
            logger.warning("未找到元数据")
        
        # 识别点云类型
        self.point_cloud_types = []
        self.point_cloud_info = {}
        
        for key in self.top_keys:
            if key != 'metadata':
                item = self.h5_file[key]
                if isinstance(item, h5py.Group):
                    # 检查组内是否有points数据集
                    if 'points' in item:
                        self.point_cloud_types.append(key)
                        self.point_cloud_info[key] = {
                            'type': 'uniform',
                            'shape': item['points'].shape
                        }
                        logger.info(f"点云类型: {key}, 形状: {item['points'].shape}")
                    # 检查是否是样本集合
                    elif any('sample_' in subkey for subkey in item.keys()):
                        self.point_cloud_types.append(key)
                        self.point_cloud_info[key] = {
                            'type': 'variable',
                            'samples': [s for s in item.keys() if 'sample_' in s]
                        }
                        logger.info(f"点云类型: {key}, 样本数: {len(self.point_cloud_info[key]['samples'])}")
    
    def get_metadata(self, sample_idx):
        """获取样本元数据
        
        参数:
            sample_idx: 样本索引
        
        返回:
            元数据字典
        """
        if 'metadata' in self.h5_file and sample_idx < self.sample_count:
            metadata = self.h5_file['metadata'][sample_idx]
            if metadata.dtype.names:
                return {name: metadata[name].decode('utf-8') if isinstance(metadata[name], bytes) else metadata[name] 
                        for name in metadata.dtype.names}
        return {}
    
    def get_point_cloud(self, pc_type, sample_idx):
        """获取点云数据
        
        参数:
            pc_type: 点云类型
            sample_idx: 样本索引
        
        返回:
            点云数据 (n, 3)
        """
        if pc_type not in self.point_cloud_types:
            logger.error(f"未知点云类型: {pc_type}")
            return np.empty((0, 3))
        
        try:
            group = self.h5_file[pc_type]
            
            # 根据点云类型获取数据
            if self.point_cloud_info[pc_type]['type'] == 'uniform':
                # 统一形状的点云
                return group['points'][sample_idx]
            elif self.point_cloud_info[pc_type]['type'] == 'variable':
                # 可变形状的点云
                sample_key = f'sample_{sample_idx}'
                if sample_key in group:
                    return group[sample_key]['points'][:]
                else:
                    logger.warning(f"未找到样本: {pc_type}/{sample_key}")
                    return np.empty((0, 3))
        except Exception as e:
            logger.error(f"获取点云数据失败: {pc_type}, 样本 {sample_idx}, 错误: {e}")
            return np.empty((0, 3))
    
    def get_point_cloud_color(self, pc_type, sample_idx):
        """获取点云颜色数据
        
        参数:
            pc_type: 点云类型
            sample_idx: 样本索引
        
        返回:
            颜色数据 (n, 3) 或 None
        """
        if pc_type not in self.point_cloud_types:
            return None
        
        try:
            group = self.h5_file[pc_type]
            
            # 根据点云类型获取数据
            if self.point_cloud_info[pc_type]['type'] == 'uniform':
                # 统一形状的点云
                if 'colors' in group:
                    return group['colors'][sample_idx]
            elif self.point_cloud_info[pc_type]['type'] == 'variable':
                # 可变形状的点云
                sample_key = f'sample_{sample_idx}'
                if sample_key in group and 'colors' in group[sample_key]:
                    return group[sample_key]['colors'][:]
        except Exception as e:
            logger.error(f"获取点云颜色数据失败: {pc_type}, 样本 {sample_idx}, 错误: {e}")
        
        return None
    
    def visualize_sample(self, sample_idx=0):
        """可视化样本点云
        
        参数:
            sample_idx: 样本索引
        """
        if sample_idx >= self.sample_count:
            logger.error(f"样本索引超出范围: {sample_idx} >= {self.sample_count}")
            return
        
        # 获取元数据
        metadata = self.get_metadata(sample_idx)
        logger.info(f"样本 {sample_idx} 元数据: {metadata}")
        
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
        
        # 添加点云到可视化器
        for pc_type in self.point_cloud_types:
            try:
                points = self.get_point_cloud(pc_type, sample_idx)
                
                if len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 尝试获取颜色
                    colors_data = self.get_point_cloud_color(pc_type, sample_idx)
                    if colors_data is not None and len(colors_data) == len(points):
                        pcd.colors = o3d.utility.Vector3dVector(colors_data)
                    else:
                        # 设置统一颜色
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
    
    def visualize_separate(self, sample_idx=0):
        """分别可视化每种点云类型
        
        参数:
            sample_idx: 样本索引
        """
        if sample_idx >= self.sample_count:
            logger.error(f"样本索引超出范围: {sample_idx} >= {self.sample_count}")
            return
        
        # 获取元数据
        metadata = self.get_metadata(sample_idx)
        logger.info(f"样本 {sample_idx} 元数据: {metadata}")
        
        # 为每种点云类型创建单独的可视化窗口
        for pc_type in self.point_cloud_types:
            try:
                points = self.get_point_cloud(pc_type, sample_idx)
                
                if len(points) > 0:
                    # 创建点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 尝试获取颜色
                    colors_data = self.get_point_cloud_color(pc_type, sample_idx)
                    if colors_data is not None and len(colors_data) == len(points):
                        pcd.colors = o3d.utility.Vector3dVector(colors_data)
                    else:
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
        if sample_idx >= self.sample_count:
            logger.error(f"样本索引超出范围: {sample_idx} >= {self.sample_count}")
            return
        
        # 获取元数据
        metadata = self.get_metadata(sample_idx)
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        
        # 为每种点云类型创建子图
        n_types = len(self.point_cloud_types)
        rows = (n_types + 2) // 3  # 向上取整
        cols = min(3, n_types)
        
        for i, pc_type in enumerate(self.point_cloud_types):
            try:
                points = self.get_point_cloud(pc_type, sample_idx)
                
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
    
    def close(self):
        """关闭H5文件"""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="H5点云数据集可视化工具")
    parser.add_argument('--h5_path', type=str, required=True, help='H5文件路径')
    parser.add_argument('--sample_idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--visualize', action='store_true', help='可视化所有点云类型')
    parser.add_argument('--visualize_separate', action='store_true', help='分别可视化每种点云类型')
    parser.add_argument('--plot', action='store_true', help='绘制点云空间分布图')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = H5PointCloudVisualizer(args.h5_path)
    
    try:
        # 可视化样本
        if args.visualize:
            visualizer.visualize_sample(args.sample_idx)
        
        # 分别可视化每种点云类型
        if args.visualize_separate:
            visualizer.visualize_separate(args.sample_idx)
        
        # 绘制点云空间分布图
        if args.plot:
            visualizer.plot_point_distribution(args.sample_idx)
    finally:
        # 关闭H5文件
        visualizer.close()

if __name__ == '__main__':
    main()
