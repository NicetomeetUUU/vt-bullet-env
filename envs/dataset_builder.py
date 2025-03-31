#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云数据集构建工具

此脚本实现了多种点云采样策略，用于构建触觉-视觉点云数据集。
支持多尺度混合采样策略、基于密度的自适应采样和基于曲率的自适应采样。
"""

import os
import numpy as np
import open3d as o3d
import json
import h5py
from tqdm import tqdm
import argparse
import random
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatasetBuilder")

class PointCloudSampler:
    """点云采样器，实现多种采样策略"""
    
    def __init__(self):
        """初始化采样器"""
        pass
    
    def uniform_sampling(self, pcd, voxel_size):
        """均匀采样
        
        参数:
            pcd: Open3D点云对象
            voxel_size: 体素大小
            
        返回:
            采样后的点云
        """
        if len(pcd.points) == 0:
            return pcd
            
        return pcd.voxel_down_sample(voxel_size)
    
    def adaptive_density_sampling(self, pcd, pcd_type):
        """基于密度的自适应采样
        
        参数:
            pcd: Open3D点云对象
            pcd_type: 点云类型，可选值为'tactile_left', 'tactile_right', 
                     'cropped_object', 'cropped_surface', 'cropped_surface_context'
                     
        返回:
            采样后的点云
        """
        if len(pcd.points) == 0:
            return pcd
            
        # 根据点云类型设置不同的采样参数
        voxel_sizes = {
            'tactile_left': 0.0005,
            'tactile_right': 0.0005,
            'cropped_object': 0.001,
            'cropped_surface': 0.002,
            'cropped_surface_context': 0.0015
        }
        
        voxel_size = voxel_sizes.get(pcd_type, 0.001)
        return pcd.voxel_down_sample(voxel_size)
    
    def curvature_based_sampling(self, pcd, high_curv_voxel_size=0.0008, low_curv_voxel_size=0.002, curv_threshold=0.005):
        """基于曲率的自适应采样
        
        参数:
            pcd: Open3D点云对象
            high_curv_voxel_size: 高曲率区域的体素大小
            low_curv_voxel_size: 低曲率区域的体素大小
            curv_threshold: 曲率阈值
            
        返回:
            采样后的点云
        """
        if len(pcd.points) < 10:  # 点数太少无法计算曲率
            return pcd
            
        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        
        # 计算曲率
        pcd_points = np.asarray(pcd.points)
        pcd_normals = np.asarray(pcd.normals)
        curvatures = []
        
        # 使用PCA方法估计曲率
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(len(pcd_points)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_points[i], 10)
            if k < 3:
                curvatures.append(0)
                continue
                
            # 获取邻近点
            neighbors = pcd_points[idx, :]
            
            # 计算协方差矩阵
            mean = np.mean(neighbors, axis=0)
            neighbors = neighbors - mean
            cov = np.matmul(neighbors.T, neighbors) / (k - 1)
            
            # 计算特征值
            eigenvalues, _ = np.linalg.eig(cov)
            eigenvalues = np.sort(eigenvalues)
            
            # 使用最小特征值与总和的比值作为曲率估计
            if np.sum(eigenvalues) != 0:
                curvature = eigenvalues[0] / np.sum(eigenvalues)
            else:
                curvature = 0
                
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # 分离高曲率和低曲率点
        high_curv_indices = np.where(curvatures > curv_threshold)[0]
        low_curv_indices = np.where(curvatures <= curv_threshold)[0]
        
        # 创建高曲率和低曲率点云
        high_curv_pcd = o3d.geometry.PointCloud()
        low_curv_pcd = o3d.geometry.PointCloud()
        
        if len(high_curv_indices) > 0:
            high_curv_pcd.points = o3d.utility.Vector3dVector(pcd_points[high_curv_indices])
            if pcd.has_colors():
                high_curv_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[high_curv_indices])
        
        if len(low_curv_indices) > 0:
            low_curv_pcd.points = o3d.utility.Vector3dVector(pcd_points[low_curv_indices])
            if pcd.has_colors():
                low_curv_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[low_curv_indices])
        
        # 对高曲率和低曲率点云分别采样
        high_curv_sampled = high_curv_pcd.voxel_down_sample(high_curv_voxel_size) if len(high_curv_indices) > 0 else high_curv_pcd
        low_curv_sampled = low_curv_pcd.voxel_down_sample(low_curv_voxel_size) if len(low_curv_indices) > 0 else low_curv_pcd
        
        # 合并采样后的点云
        combined_pcd = o3d.geometry.PointCloud()
        combined_points = []
        combined_colors = []
        
        if len(high_curv_sampled.points) > 0:
            combined_points.append(np.asarray(high_curv_sampled.points))
            if high_curv_sampled.has_colors():
                combined_colors.append(np.asarray(high_curv_sampled.colors))
        
        if len(low_curv_sampled.points) > 0:
            combined_points.append(np.asarray(low_curv_sampled.points))
            if low_curv_sampled.has_colors():
                combined_colors.append(np.asarray(low_curv_sampled.colors))
        
        if combined_points:
            combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
            if combined_colors and all(c.shape[0] > 0 for c in combined_colors):
                combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
        
        return combined_pcd
    
    def multiscale_hybrid_sampling(self, pcd_dict, contact_point=None, gt_density=None):
        """多尺度混合采样策略
        
        参数:
            pcd_dict: 包含不同类型点云的字典
            contact_point: 触觉接触点坐标，用于基于空间位置的重要性权重
            gt_density: GT点云的密度，用于统一采样密度
            
        返回:
            采样后的点云字典
        """
        result = {}
        
        # 第一级：基于点云类型的基础采样
        target_points = {
            'tactile_left': None,  # 保持原样
            'tactile_right': None,  # 保持原样
            'cropped_object': 2500,  # 目标约2000-3000点
            'cropped_surface': 25000,  # 目标约20000-30000点
            'cropped_surface_context': 10000  # 目标约10000点
        }
        
        base_voxel_sizes = {
            'tactile_left': 0.0005,  # 触觉点云保持高密度
            'tactile_right': 0.0005,
            'cropped_object': 0.001,
            'cropped_surface': 0.003,
            'cropped_surface_context': 0.002
        }
        
        # 对每种点云类型应用采样
        for pcd_type, pcd in pcd_dict.items():
            if pcd is None or len(pcd.points) == 0:
                result[pcd_type] = pcd
                continue
            
            # 基础采样
            voxel_size = base_voxel_sizes.get(pcd_type, 0.001)
            
            # 触觉点云保持原样
            if pcd_type in ['tactile_left', 'tactile_right']:
                result[pcd_type] = pcd
                continue
            
            # 第二级：基于局部特征的自适应调整
            # 计算曲率
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            pcd_points = np.asarray(pcd.points)
            pcd_normals = np.asarray(pcd.normals)
            
            # 使用PCA方法估计曲率
            curvatures = []
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            for i in range(len(pcd_points)):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_points[i], 10)
                if k < 3:
                    curvatures.append(0)
                    continue
                    
                neighbors = pcd_points[idx, :]
                mean = np.mean(neighbors, axis=0)
                neighbors = neighbors - mean
                cov = np.matmul(neighbors.T, neighbors) / (k - 1)
                eigenvalues, _ = np.linalg.eig(cov)
                eigenvalues = np.sort(eigenvalues)
                
                if np.sum(eigenvalues) != 0:
                    curvature = eigenvalues[0] / np.sum(eigenvalues)
                else:
                    curvature = 0
                    
                curvatures.append(curvature)
            
            curvatures = np.array(curvatures)
            
            # 根据曲率调整体素大小
            high_curv_indices = np.where(curvatures > 0.01)[0]
            low_curv_indices = np.where(curvatures < 0.001)[0]
            mid_curv_indices = np.setdiff1d(np.arange(len(pcd_points)), np.concatenate([high_curv_indices, low_curv_indices]))
            
            # 创建不同曲率区域的点云
            high_curv_pcd = o3d.geometry.PointCloud()
            low_curv_pcd = o3d.geometry.PointCloud()
            mid_curv_pcd = o3d.geometry.PointCloud()
            
            if len(high_curv_indices) > 0:
                high_curv_pcd.points = o3d.utility.Vector3dVector(pcd_points[high_curv_indices])
                if pcd.has_colors():
                    high_curv_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[high_curv_indices])
            
            if len(low_curv_indices) > 0:
                low_curv_pcd.points = o3d.utility.Vector3dVector(pcd_points[low_curv_indices])
                if pcd.has_colors():
                    low_curv_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[low_curv_indices])
            
            if len(mid_curv_indices) > 0:
                mid_curv_pcd.points = o3d.utility.Vector3dVector(pcd_points[mid_curv_indices])
                if pcd.has_colors():
                    mid_curv_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mid_curv_indices])
            
            # 对不同曲率区域应用不同的采样密度
            high_curv_sampled = high_curv_pcd.voxel_down_sample(voxel_size * 0.5) if len(high_curv_indices) > 0 else high_curv_pcd
            low_curv_sampled = low_curv_pcd.voxel_down_sample(voxel_size * 1.5) if len(low_curv_indices) > 0 else low_curv_pcd
            mid_curv_sampled = mid_curv_pcd.voxel_down_sample(voxel_size) if len(mid_curv_indices) > 0 else mid_curv_pcd
            
            # 第三级：基于空间位置的重要性权重
            # 如果提供了触觉接触点，根据距离调整采样
            if contact_point is not None and pcd_type in ['cropped_surface', 'cropped_surface_context']:
                # 计算每个点到接触点的距离
                distances = np.linalg.norm(np.asarray(pcd.points) - contact_point, axis=1)
                
                # 根据距离划分区域
                near_indices = np.where(distances < 0.02)[0]  # 2cm以内
                far_indices = np.where(distances >= 0.02)[0]
                
                near_pcd = o3d.geometry.PointCloud()
                far_pcd = o3d.geometry.PointCloud()
                
                if len(near_indices) > 0:
                    near_pcd.points = o3d.utility.Vector3dVector(pcd_points[near_indices])
                    if pcd.has_colors():
                        near_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[near_indices])
                
                if len(far_indices) > 0:
                    far_pcd.points = o3d.utility.Vector3dVector(pcd_points[far_indices])
                    if pcd.has_colors():
                        far_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[far_indices])
                
                # 对近距离区域使用更高的采样密度
                near_sampled = near_pcd.voxel_down_sample(voxel_size * 0.7) if len(near_indices) > 0 else near_pcd
                far_sampled = far_pcd.voxel_down_sample(voxel_size * 1.2) if len(far_indices) > 0 else far_pcd
                
                # 合并采样后的点云
                combined_pcd = o3d.geometry.PointCloud()
                combined_points = []
                combined_colors = []
                
                if len(near_sampled.points) > 0:
                    combined_points.append(np.asarray(near_sampled.points))
                    if near_sampled.has_colors():
                        combined_colors.append(np.asarray(near_sampled.colors))
                
                if len(far_sampled.points) > 0:
                    combined_points.append(np.asarray(far_sampled.points))
                    if far_sampled.has_colors():
                        combined_colors.append(np.asarray(far_sampled.colors))
                
                if combined_points:
                    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
                    if combined_colors and all(c.shape[0] > 0 for c in combined_colors):
                        combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
                
                result[pcd_type] = combined_pcd
            else:
                # 合并不同曲率区域的采样结果
                combined_pcd = o3d.geometry.PointCloud()
                combined_points = []
                combined_colors = []
                
                if len(high_curv_sampled.points) > 0:
                    combined_points.append(np.asarray(high_curv_sampled.points))
                    if high_curv_sampled.has_colors():
                        combined_colors.append(np.asarray(high_curv_sampled.colors))
                
                if len(mid_curv_sampled.points) > 0:
                    combined_points.append(np.asarray(mid_curv_sampled.points))
                    if mid_curv_sampled.has_colors():
                        combined_colors.append(np.asarray(mid_curv_sampled.colors))
                
                if len(low_curv_sampled.points) > 0:
                    combined_points.append(np.asarray(low_curv_sampled.points))
                    if low_curv_sampled.has_colors():
                        combined_colors.append(np.asarray(low_curv_sampled.colors))
                
                if combined_points:
                    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
                    if combined_colors and all(c.shape[0] > 0 for c in combined_colors):
                        combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
                
                result[pcd_type] = combined_pcd
            
            # 检查是否需要进一步采样以达到目标点数
            target = target_points.get(pcd_type)
            if target is not None and len(result[pcd_type].points) > target:
                # 计算需要的体素大小以达到目标点数
                current_points = len(result[pcd_type].points)
                ratio = (current_points / target) ** (1/3)  # 三维空间中的比例
                adjusted_voxel_size = voxel_size * ratio
                result[pcd_type] = result[pcd_type].voxel_down_sample(adjusted_voxel_size)
        
        return result


class DatasetBuilder:
    """数据集构建器，用于处理和存储点云数据集"""
    
    def __init__(self, dataset_dir, output_dir, sampling_strategy='multiscale_hybrid', 
                 max_points=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 storage_mode='hierarchical', excluded_objects=None):
        """初始化数据集构建器
        
        参数:
            dataset_dir: 原始数据集目录
            output_dir: 输出目录
            sampling_strategy: 采样策略，可选值为'uniform', 'adaptive_density', 'curvature_based', 'multiscale_hybrid'
            max_points: 每种点云类型的最大点数，字典格式
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            storage_mode: 存储模式，可选值为'single_array', 'scattered', 'hierarchical'
            excluded_objects: 要排除的物体ID列表
        """
        # 添加dataset_structure属性，用于存储筛选后的数据集结构
        self.dataset_structure = None
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.sampling_strategy = sampling_strategy
        self.max_points = max_points or {
            'tactile_left': None,  # 保持原样
            'tactile_right': None,  # 保持原样
            'cropped_object': 2500,
            'cropped_surface': 25000,
            'cropped_surface_context': 10000
        }
        
        # 设置默认排除的问题物体ID（71-87）
        self.excluded_objects = excluded_objects or [f"{i:03d}" for i in range(71, 88)]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.storage_mode = storage_mode
        
        # 确保比例之和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"数据集比例之和不为1（{total_ratio}），已自动调整")
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio
        
        # 创建采样器
        self.sampler = PointCloudSampler()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    def analyze_dataset_structure(self):
        """分析数据集的目录结构"""
        logger.info(f"分析数据集结构: {self.dataset_dir}")
        
        # 获取所有物体目录，排除问题物体
        obj_dirs = []
        for i in range(0, 88):
            obj_id = f"{i:03d}"
            if obj_id in self.excluded_objects:
                logger.info(f"排除问题物体: {obj_id}")
                continue
                
            obj_dir = os.path.join(self.dataset_dir, obj_id)
            if os.path.exists(obj_dir):
                obj_dirs.append(obj_dir)
        
        logger.info(f"找到 {len(obj_dirs)} 个有效物体")
        
        # 分析每个物体的位姿和抓取
        dataset_structure = {}
        for obj_dir in obj_dirs:
            obj_id = os.path.basename(obj_dir)
            pose_dirs = [d for d in os.listdir(obj_dir) if d.startswith('pose_')]
            
            dataset_structure[obj_id] = {
                'poses': {}
            }
            
            for pose_dir in pose_dirs:
                pose_id = pose_dir.split('_')[1]
                pose_path = os.path.join(obj_dir, pose_dir)
                
                # 查找抓取目录（支持多种命名格式）
                grasp_dirs = []
                for item in os.listdir(pose_path):
                    if os.path.isdir(os.path.join(pose_path, item)) and ('_grasp_' in item):
                        grasp_dirs.append(item)
                
                # 查找抓取姿态文件
                grasp_pose_file = os.path.join(pose_path, f"{pose_id}_grasp_poses.json")
                has_grasp_poses = os.path.exists(grasp_pose_file)
                
                dataset_structure[obj_id]['poses'][pose_id] = {
                    'path': pose_path,
                    'grasps': grasp_dirs,
                    'grasp_pose_file': grasp_pose_file if has_grasp_poses else None
                }
                
                logger.info(f"物体 {obj_id} 位姿 {pose_id}: 找到 {len(grasp_dirs)} 个抓取目录, 抓取姿态文件: {has_grasp_poses}")
        
        self.dataset_structure = dataset_structure
        return dataset_structure
    
    def load_point_clouds(self, obj_id, pose_id, grasp_id):
        """加载指定物体、位姿和抓取的点云数据
        
        参数:
            obj_id: 物体ID，如'000'
            pose_id: 位姿ID，如'000'
            grasp_id: 抓取ID，如'000_grasp_000'
            
        返回:
            包含不同类型点云的字典
        """
        pose_path = os.path.join(self.dataset_dir, obj_id, f"pose_{pose_id}")
        grasp_path = os.path.join(pose_path, grasp_id)
        
        if not os.path.exists(grasp_path):
            logger.warning(f"抓取路径不存在: {grasp_path}")
            return None
        
        # 加载各类点云
        pcd_dict = {}
        pcd_files = {
            'tactile_left': f"{obj_id}_tactile_left.pcd",
            'tactile_right': f"{obj_id}_tactile_right.pcd",
            'cropped_object': f"{obj_id}_cropped_object.pcd",
            'cropped_surface': f"{obj_id}_cropped_surface.pcd",
            'cropped_surface_context': f"{obj_id}_cropped_surface_context.pcd"
        }
        
        # 加载相机点云和物体点云（位于位姿目录下）
        camera_pcd_path = os.path.join(pose_path, f"{pose_id}_camera.pcd")
        object_pcd_path = os.path.join(pose_path, f"{pose_id}_object.pcd")
        
        if os.path.exists(camera_pcd_path):
            try:
                pcd = o3d.io.read_point_cloud(camera_pcd_path)
                if len(pcd.points) > 0:
                    pcd_dict['camera'] = pcd
                    logger.info(f"加载相机点云: {camera_pcd_path}, 点数: {len(pcd.points)}")
            except Exception as e:
                logger.error(f"加载相机点云失败: {camera_pcd_path}, 错误: {e}")
        
        if os.path.exists(object_pcd_path):
            try:
                pcd = o3d.io.read_point_cloud(object_pcd_path)
                if len(pcd.points) > 0:
                    pcd_dict['object'] = pcd
                    logger.info(f"加载物体点云: {object_pcd_path}, 点数: {len(pcd.points)}")
            except Exception as e:
                logger.error(f"加载物体点云失败: {object_pcd_path}, 错误: {e}")
        
        # 加载抓取目录下的点云
        missing_files = []
        for pcd_type, pcd_file in pcd_files.items():
            pcd_path = os.path.join(grasp_path, pcd_file)
            if os.path.exists(pcd_path):
                try:
                    pcd = o3d.io.read_point_cloud(pcd_path)
                    if len(pcd.points) > 0:
                        pcd_dict[pcd_type] = pcd
                        logger.info(f"加载点云: {pcd_path}, 点数: {len(pcd.points)}")
                    else:
                        logger.warning(f"点云为空: {pcd_path}")
                        missing_files.append(f"{pcd_type} (空点云)")
                except Exception as e:
                    logger.error(f"加载点云失败: {pcd_path}, 错误: {e}")
                    missing_files.append(f"{pcd_type} (加载错误)")
            else:
                logger.warning(f"点云文件不存在: {pcd_path}")
                missing_files.append(pcd_type)
        
        if missing_files:
            logger.warning(f"样本 {obj_id}/{pose_id}/{grasp_id} 缺失文件: {', '.join(missing_files)}")
        
        if not pcd_dict:
            logger.warning(f"样本 {obj_id}/{pose_id}/{grasp_id} 没有有效点云")
        
        return pcd_dict
    
    def sample_point_clouds(self, pcd_dict):
        """对点云进行采样
        
        参数:
            pcd_dict: 包含不同类型点云的字典
            
        返回:
            采样后的点云字典
        """
        if not pcd_dict:
            return None
        
        # 获取触觉点云的中心点，用于空间位置重要性权重
        contact_point = None
        if 'tactile_left' in pcd_dict and len(pcd_dict['tactile_left'].points) > 0:
            contact_point = np.mean(np.asarray(pcd_dict['tactile_left'].points), axis=0)
        
        # 根据采样策略进行采样
        if self.sampling_strategy == 'uniform':
            # 均匀采样
            result = {}
            
            # 首先处理GT点云（tactile_left和tactile_right）和物体点云，确保它们采用相同密度
            gt_density = self._calculate_gt_density(pcd_dict)
            
            for pcd_type, pcd in pcd_dict.items():
                max_points = self.max_points.get(pcd_type)
                
                # 对于GT和物体点云，使用相同的采样密度
                if pcd_type in ['tactile_left', 'tactile_right', 'cropped_object']:
                    if max_points is None or len(pcd.points) <= max_points:
                        result[pcd_type] = pcd
                    else:
                        result[pcd_type] = self._sample_with_density(pcd, gt_density, max_points)
                # 对于上下文点云，基于GT密度进行降采样
                elif pcd_type in ['cropped_surface', 'cropped_surface_context']:
                    if max_points is None or len(pcd.points) <= max_points:
                        result[pcd_type] = pcd
                    else:
                        # 上下文点云使用较低的采样密度（GT密度的2-3倍）
                        context_density = gt_density * 2.5
                        result[pcd_type] = self._sample_with_density(pcd, context_density, max_points)
                else:
                    # 其他类型点云使用默认处理
                    if max_points is None or len(pcd.points) <= max_points:
                        result[pcd_type] = pcd
                    else:
                        current_points = len(pcd.points)
                        ratio = (current_points / max_points) ** (1/3)  # 三维空间中的比例
                        voxel_size = 0.001 * ratio  # 基础体素大小 * 比例
                        result[pcd_type] = self.sampler.uniform_sampling(pcd, voxel_size)
            
            return result
        
        elif self.sampling_strategy == 'adaptive_density':
            # 基于密度的自适应采样
            result = {}
            
            # 首先处理GT点云和物体点云，确保它们采用相同密度
            gt_density = self._calculate_gt_density(pcd_dict)
            
            for pcd_type, pcd in pcd_dict.items():
                if pcd_type in ['tactile_left', 'tactile_right', 'cropped_object']:
                    max_points = self.max_points.get(pcd_type)
                    if max_points is None or len(pcd.points) <= max_points:
                        result[pcd_type] = pcd
                    else:
                        result[pcd_type] = self._sample_with_density(pcd, gt_density, max_points)
                elif pcd_type in ['cropped_surface', 'cropped_surface_context']:
                    # 上下文点云使用基于密度的自适应采样，但基于GT密度降采样
                    result[pcd_type] = self.sampler.adaptive_density_sampling(pcd, pcd_type, density_factor=2.5)
                else:
                    result[pcd_type] = self.sampler.adaptive_density_sampling(pcd, pcd_type)
            
            return result
        
        elif self.sampling_strategy == 'curvature_based':
            # 基于曲率的自适应采样
            result = {}
            
            # 首先处理GT点云和物体点云，确保它们采用相同密度
            gt_density = self._calculate_gt_density(pcd_dict)
            
            for pcd_type, pcd in pcd_dict.items():
                if pcd_type in ['tactile_left', 'tactile_right', 'cropped_object']:
                    max_points = self.max_points.get(pcd_type)
                    if max_points is None or len(pcd.points) <= max_points:
                        result[pcd_type] = pcd
                    else:
                        result[pcd_type] = self._sample_with_density(pcd, gt_density, max_points)
                else:
                    result[pcd_type] = self.sampler.curvature_based_sampling(pcd)
            
            return result
        
        elif self.sampling_strategy == 'multiscale_hybrid':
            # 多尺度混合采样策略，但确保GT和物体点云采用相同密度
            gt_density = self._calculate_gt_density(pcd_dict)
            
            # 首先处理GT和物体点云
            for pcd_type in ['tactile_left', 'tactile_right', 'cropped_object']:
                if pcd_type in pcd_dict:
                    max_points = self.max_points.get(pcd_type)
                    if max_points is not None and len(pcd_dict[pcd_type].points) > max_points:
                        pcd_dict[pcd_type] = self._sample_with_density(pcd_dict[pcd_type], gt_density, max_points)
            
            # 然后使用多尺度混合采样处理其他点云
            return self.sampler.multiscale_hybrid_sampling(pcd_dict, contact_point, gt_density=gt_density)
        
        else:
            logger.error(f"未知的采样策略: {self.sampling_strategy}")
            return pcd_dict
    
    def _calculate_gt_density(self, pcd_dict):
        """计算GT点云的密度，用于统一采样密度
        
        参数:
            pcd_dict: 包含不同类型点云的字典
            
        返回:
            估计的GT点云密度（体素大小）
        """
        # 默认密度（体素大小）
        default_density = 0.001
        
        # 尝试从触觉点云计算密度
        gt_pcds = []
        if 'tactile_left' in pcd_dict and len(pcd_dict['tactile_left'].points) > 0:
            gt_pcds.append(pcd_dict['tactile_left'])
        if 'tactile_right' in pcd_dict and len(pcd_dict['tactile_right'].points) > 0:
            gt_pcds.append(pcd_dict['tactile_right'])
        
        if not gt_pcds:
            return default_density
        
        # 计算平均点云密度
        densities = []
        for pcd in gt_pcds:
            points = np.asarray(pcd.points)
            if len(points) < 10:  # 点数太少，跳过
                continue
                
            # 计算点云的边界框体积
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            volume = np.prod(max_bound - min_bound)
            
            if volume > 0:
                # 估计点密度（点数/体积）
                point_density = len(points) / volume
                # 转换为体素大小（体素大小与点密度成反比）
                voxel_size = 0.1 / (point_density ** (1/3))
                densities.append(voxel_size)
        
        if densities:
            # 使用中位数避免异常值影响
            return np.median(densities)
        else:
            return default_density
    
    def _sample_with_density(self, pcd, density, max_points=None):
        """使用指定密度对点云进行采样
        
        参数:
            pcd: 点云对象
            density: 采样密度（体素大小）
            max_points: 最大点数限制
            
        返回:
            采样后的点云
        """
        # 首先使用指定密度进行下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=density)
        
        # 如果指定了最大点数且下采样后点数仍然过多
        if max_points is not None and len(downsampled_pcd.points) > max_points:
            # 计算需要的体素大小以达到目标点数
            current_points = len(downsampled_pcd.points)
            ratio = (current_points / max_points) ** (1/3)  # 三维空间中的比例
            new_density = density * ratio
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=new_density)
        
        return downsampled_pcd
    
    def convert_to_numpy(self, pcd_dict):
        """将点云转换为numpy数组
        
        参数:
            pcd_dict: 包含不同类型点云的字典
            
        返回:
            包含numpy数组的字典
        """
        if not pcd_dict:
            return None
        
        np_dict = {}
        for pcd_type, pcd in pcd_dict.items():
            points = np.asarray(pcd.points).astype(np.float32)
            
            # 如果点云有颜色，也保存颜色信息
            if pcd.has_colors():
                colors = np.asarray(pcd.colors).astype(np.float32)
                np_dict[pcd_type] = {
                    'points': points,
                    'colors': colors
                }
            else:
                np_dict[pcd_type] = {
                    'points': points
                }
        
        return np_dict
    
    def normalize_point_cloud(self, points):
        """归一化点云
        
        参数:
            points: 点云坐标，形状为(N, 3)的numpy数组
            
        返回:
            归一化后的点云
        """
        if len(points) == 0:
            return points
            
        # 计算中心点
        centroid = np.mean(points, axis=0)
        
        # 移到原点
        points_centered = points - centroid
        
        # 计算最大距离
        max_dist = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        
        # 归一化到[-1, 1]范围
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        
        return points_normalized
    
    def pad_or_sample_point_cloud(self, points, target_size):
        """填充或采样点云到指定大小
        
        参数:
            points: 点云坐标，形状为(N, 3)的numpy数组
            target_size: 目标点数
            
        返回:
            处理后的点云，形状为(target_size, 3)
        """
        if len(points) == 0:
            return np.zeros((target_size, 3), dtype=np.float32)
            
        if len(points) == target_size:
            return points
            
        if len(points) > target_size:
            # 随机采样
            indices = np.random.choice(len(points), target_size, replace=False)
            return points[indices]
        else:
            # 填充（重复采样）
            indices = np.random.choice(len(points), target_size - len(points), replace=True)
            return np.vstack([points, points[indices]])
    
    def process_sample(self, obj_id, pose_id, grasp_id):
        """处理单个样本
        
        参数:
            obj_id: 物体ID
            pose_id: 位姿ID
            grasp_id: 抓取ID
            
        返回:
            处理后的样本数据
        """
        # 加载点云
        pcd_dict = self.load_point_clouds(obj_id, pose_id, grasp_id)
        if not pcd_dict:
            return None
        
        # 检查抓取姿态文件
        grasp_id_num = grasp_id.split('_')[-1]  # 提取抓取ID的数字部分
        grasp_pose_file = os.path.join(self.dataset_dir, obj_id, f"pose_{pose_id}", f"{pose_id}_grasp_poses.json")
        grasp_label = 0  # 默认为无效抓取
        
        if os.path.exists(grasp_pose_file):
            try:
                with open(grasp_pose_file, 'r') as f:
                    grasp_poses = json.load(f)
                    
                # 在JSON文件中查找对应的抓取姿态
                for grasp in grasp_poses:
                    if str(grasp.get('id', '')) == grasp_id_num:
                        grasp_label = 1 if grasp.get('label', 0) > 0 else 0
                        logger.info(f"找到抓取 {grasp_id_num} 的标签: {grasp_label}")
                        break
            except Exception as e:
                logger.error(f"读取抓取姿态文件失败: {grasp_pose_file}, 错误: {e}")
        else:
            # 尝试从抓取目录中读取抓取姿态
            grasp_path = os.path.join(self.dataset_dir, obj_id, f"pose_{pose_id}", grasp_id)
            grasp_pose_json = os.path.join(grasp_path, f"{obj_id}_tactile_grasp_pose.json")
            
            if os.path.exists(grasp_pose_json):
                try:
                    with open(grasp_pose_json, 'r') as f:
                        grasp_data = json.load(f)
                        grasp_label = 1 if grasp_data.get('label', 0) > 0 else 0
                        logger.info(f"从抓取目录读取到标签: {grasp_label}")
                except Exception as e:
                    logger.error(f"读取抓取姿态文件失败: {grasp_pose_json}, 错误: {e}")
        
        # 采样点云
        sampled_pcd_dict = self.sample_point_clouds(pcd_dict)
        if not sampled_pcd_dict:
            logger.warning(f"样本 {obj_id}/{pose_id}/{grasp_id} 点云采样失败")
            return None
        
        # 转换为numpy数组
        np_dict = self.convert_to_numpy(sampled_pcd_dict)
        if not np_dict:
            return None
        
        # 处理每种点云
        processed_data = {}
        for pcd_type, data in np_dict.items():
            points = data['points']
            
            # 归一化
            points_normalized = self.normalize_point_cloud(points)
            
            # 填充或采样到固定大小
            target_size = self.max_points.get(pcd_type)
            if target_size:
                points_processed = self.pad_or_sample_point_cloud(points_normalized, target_size)
            else:
                points_processed = points_normalized
            
            processed_data[pcd_type] = {
                'points': points_processed
            }
            
            # 如果有颜色，也处理颜色
            if 'colors' in data:
                colors = data['colors']
                if target_size and len(colors) != len(points_processed):
                    if len(colors) > target_size:
                        indices = np.random.choice(len(colors), target_size, replace=False)
                        colors_processed = colors[indices]
                    else:
                        indices = np.random.choice(len(colors), target_size - len(colors), replace=True)
                        colors_processed = np.vstack([colors, colors[indices]])
                else:
                    colors_processed = colors
                
                processed_data[pcd_type]['colors'] = colors_processed
        
        # 添加元数据
        processed_data['metadata'] = {
            'obj_id': obj_id,
            'pose_id': pose_id,
            'grasp_id': grasp_id,
            'label': grasp_label
        }
        
        logger.info(f"处理样本 {obj_id}/{pose_id}/{grasp_id} 完成，标签: {grasp_label}")
        return processed_data
    
    def split_dataset(self, samples):
        """将数据集划分为训练集、验证集和测试集
        
        参数:
            samples: 样本列表
            
        返回:
            训练集、验证集和测试集
        """
        # 按物体ID分组
        obj_groups = {}
        for sample in samples:
            obj_id = sample['metadata']['obj_id']
            if obj_id not in obj_groups:
                obj_groups[obj_id] = []
            obj_groups[obj_id].append(sample)
        
        # 对每个物体的样本进行随机排序
        for obj_id in obj_groups:
            random.shuffle(obj_groups[obj_id])
        
        # 按比例划分
        train_samples = []
        val_samples = []
        test_samples = []
        
        for obj_id, obj_samples in obj_groups.items():
            n_samples = len(obj_samples)
            n_train = int(n_samples * self.train_ratio)
            n_val = int(n_samples * self.val_ratio)
            
            train_samples.extend(obj_samples[:n_train])
            val_samples.extend(obj_samples[n_train:n_train+n_val])
            test_samples.extend(obj_samples[n_train+n_val:])
        
        return train_samples, val_samples, test_samples
    
    def save_single_array(self, samples, split):
        """将样本保存为单一大型数组
        
        参数:
            samples: 样本列表
            split: 数据集划分，'train', 'val' 或 'test'
        """
        if not samples:
            logger.warning(f"没有{split}样本可保存")
            return
        
        output_path = os.path.join(self.output_dir, split, f"{split}_data.h5")
        logger.info(f"保存{split}数据到: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # 创建点云数据组
            pcd_types = ['tactile_left', 'tactile_right', 'cropped_object', 'cropped_surface', 'cropped_surface_context']
            
            for pcd_type in pcd_types:
                # 收集所有样本的该类型点云
                points_list = []
                colors_list = []
                
                for sample in samples:
                    if pcd_type in sample:
                        points_list.append(sample[pcd_type]['points'])
                        if 'colors' in sample[pcd_type]:
                            colors_list.append(sample[pcd_type]['colors'])
                
                if points_list:
                    # 创建数据集
                    points_array = np.stack(points_list)
                    f.create_dataset(f"{pcd_type}/points", data=points_array, compression="gzip")
                    
                    if colors_list and len(colors_list) == len(points_list):
                        colors_array = np.stack(colors_list)
                        f.create_dataset(f"{pcd_type}/colors", data=colors_array, compression="gzip")
            
            # 保存元数据
            metadata = np.array([
                (sample['metadata']['obj_id'], sample['metadata']['pose_id'], sample['metadata']['grasp_id'])
                for sample in samples
            ], dtype=[('obj_id', 'S10'), ('pose_id', 'S10'), ('grasp_id', 'S20')])
            
            f.create_dataset("metadata", data=metadata)
    
    def save_scattered(self, samples, split):
        """将样本分散保存
        
        参数:
            samples: 样本列表
            split: 数据集划分，'train', 'val' 或 'test'
        """
        if not samples:
            logger.warning(f"没有{split}样本可保存")
            return
        
        output_dir = os.path.join(self.output_dir, split)
        logger.info(f"保存{split}数据到: {output_dir}")
        
        for i, sample in enumerate(tqdm(samples, desc=f"保存{split}数据")):
            obj_id = sample['metadata']['obj_id']
            pose_id = sample['metadata']['pose_id']
            grasp_id = sample['metadata']['grasp_id']
            
            # 创建样本目录
            sample_dir = os.path.join(output_dir, f"{obj_id}_{pose_id}_{grasp_id}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存每种点云
            for pcd_type, data in sample.items():
                if pcd_type == 'metadata':
                    continue
                
                # 保存点云坐标
                points = data['points']
                np.save(os.path.join(sample_dir, f"{pcd_type}_points.npy"), points)
                
                # 如果有颜色，也保存颜色
                if 'colors' in data:
                    colors = data['colors']
                    np.save(os.path.join(sample_dir, f"{pcd_type}_colors.npy"), colors)
            
            # 保存元数据
            with open(os.path.join(sample_dir, "metadata.json"), 'w') as f:
                json.dump(sample['metadata'], f)
    
    def save_hierarchical(self, samples, split):
        """使用分层混合存储方式保存样本
        
        参数:
            samples: 样本列表
            split: 数据集划分，'train', 'val' 或 'test'
        """
        if not samples:
            logger.warning(f"没有{split}样本可保存")
            return
        
        # 按物体ID分组
        obj_groups = {}
        for sample in samples:
            obj_id = sample['metadata']['obj_id']
            if obj_id not in obj_groups:
                obj_groups[obj_id] = []
            obj_groups[obj_id].append(sample)
        
        output_dir = os.path.join(self.output_dir, split)
        logger.info(f"保存{split}数据到: {output_dir}")
        
        # 为每个物体创建一个文件
        for obj_id, obj_samples in tqdm(obj_groups.items(), desc=f"保存{split}数据"):
            output_path = os.path.join(output_dir, f"{obj_id}.h5")
            
            with h5py.File(output_path, 'w') as f:
                # 创建点云数据组
                pcd_types = ['tactile_left', 'tactile_right', 'cropped_object', 'cropped_surface', 'cropped_surface_context']
                
                for pcd_type in pcd_types:
                    # 收集所有样本的该类型点云
                    points_list = []
                    colors_list = []
                    sample_indices = []
                    
                    for i, sample in enumerate(obj_samples):
                        if pcd_type in sample:
                            points_list.append(sample[pcd_type]['points'])
                            if 'colors' in sample[pcd_type]:
                                colors_list.append(sample[pcd_type]['colors'])
                            sample_indices.append(i)
                    
                    if points_list:
                        # 检查所有点云数组的形状
                        shapes = [points.shape for points in points_list]
                        logger.info(f"{pcd_type} 点云形状: {shapes}")
                        
                        # 检查是否所有数组形状相同
                        if len(set(tuple(shape) for shape in shapes)) == 1:
                            # 形状相同，可以直接使用stack
                            points_array = np.stack(points_list)
                            f.create_dataset(f"{pcd_type}/points", data=points_array, compression="gzip")
                            
                            # 保存样本索引
                            f.create_dataset(f"{pcd_type}/sample_indices", data=np.array(sample_indices))
                            
                            if colors_list and len(colors_list) == len(points_list):
                                # 检查颜色数组形状
                                color_shapes = [colors.shape for colors in colors_list]
                                if len(set(tuple(shape) for shape in color_shapes)) == 1:
                                    colors_array = np.stack(colors_list)
                                    f.create_dataset(f"{pcd_type}/colors", data=colors_array, compression="gzip")
                                else:
                                    logger.warning(f"{pcd_type} 颜色数组形状不一致，跳过保存")
                        else:
                            # 形状不同，分别保存每个样本
                            logger.warning(f"{pcd_type} 点云形状不一致，分别保存每个样本")
                            for i, (points, idx) in enumerate(zip(points_list, sample_indices)):
                                f.create_dataset(f"{pcd_type}/sample_{idx}/points", data=points, compression="gzip")
                                if i < len(colors_list):
                                    f.create_dataset(f"{pcd_type}/sample_{idx}/colors", data=colors_list[i], compression="gzip")
                            
                            # 保存样本索引
                            f.create_dataset(f"{pcd_type}/sample_indices", data=np.array(sample_indices))
                
                # 保存元数据
                metadata = np.array([
                    (sample['metadata']['obj_id'], sample['metadata']['pose_id'], sample['metadata']['grasp_id'])
                    for sample in obj_samples
                ], dtype=[('obj_id', 'S10'), ('pose_id', 'S10'), ('grasp_id', 'S20')])
                
                f.create_dataset("metadata", data=metadata)
    
    def process_dataset(self):
        """处理整个数据集
        
        处理流程：
        1. 加载点云数据
        2. 采样点云
        3. 归一化点云
        4. 填充或采样到固定大小
        5. 划分数据集
        6. 保存数据集
        """
        # 如果dataset_structure为None，则先分析数据集结构
        if self.dataset_structure is None:
            self.dataset_structure = self.analyze_dataset_structure()
            logger.info(f"已分析数据集结构，共有 {len(self.dataset_structure)} 个物体")
        else:
            logger.info(f"使用已提供的数据集结构，共有 {len(self.dataset_structure)} 个物体")
        
        # 打印每个物体的位姿和抓取数量统计
        for obj_id, obj_data in self.dataset_structure.items():
            pose_count = len(obj_data['poses'])
            grasp_count = sum(len(pose_data['grasps']) for pose_data in obj_data['poses'].values())
            logger.info(f"物体 {obj_id}: {pose_count} 个位姿, {grasp_count} 个抓取")
        
        # 收集所有样本
        all_samples = []
        processed_count = 0
        skipped_count = 0
        
        for obj_id, obj_data in tqdm(self.dataset_structure.items(), desc="处理物体"):
            logger.info(f"处理物体: {obj_id}")
            obj_processed = 0
            obj_skipped = 0
            
            for pose_id, pose_data in obj_data['poses'].items():
                for grasp_id in tqdm(pose_data['grasps'], desc=f"处理物体{obj_id}位姿{pose_id}", leave=False):
                    # 处理样本
                    sample = self.process_sample(obj_id, pose_id, grasp_id)
                    if sample:
                        all_samples.append(sample)
                        obj_processed += 1
                        processed_count += 1
                    else:
                        obj_skipped += 1
                        skipped_count += 1
            
            logger.info(f"物体 {obj_id} 处理结果: {obj_processed} 个成功, {obj_skipped} 个跳过")
        
        logger.info(f"共处理 {processed_count} 个样本成功, {skipped_count} 个样本跳过, 总计: {processed_count + skipped_count}")
        
        # 如果没有样本，则提前返回
        if not all_samples:
            logger.warning("没有有效样本可处理")
            return
        
        # 划分数据集
        train_samples, val_samples, test_samples = self.split_dataset(all_samples)
        
        logger.info(f"训练集: {len(train_samples)} 个样本")
        logger.info(f"验证集: {len(val_samples)} 个样本")
        logger.info(f"测试集: {len(test_samples)} 个样本")
        
        # 确保输出目录存在
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        
        # 保存数据集
        if self.storage_mode == 'single_array':
            self.save_single_array(train_samples, 'train')
            self.save_single_array(val_samples, 'val')
            self.save_single_array(test_samples, 'test')
        elif self.storage_mode == 'scattered':
            self.save_scattered(train_samples, 'train')
            self.save_scattered(val_samples, 'val')
            self.save_scattered(test_samples, 'test')
        elif self.storage_mode == 'hierarchical':
            self.save_hierarchical(train_samples, 'train')
            self.save_hierarchical(val_samples, 'val')
            self.save_hierarchical(test_samples, 'test')
        else:
            logger.error(f"未知的存储模式: {self.storage_mode}")


def process_sample_parallel(args):
    """并行处理样本的辅助函数"""
    builder, obj_id, pose_id, grasp_id = args
    return builder.process_sample(obj_id, pose_id, grasp_id)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="点云数据集构建工具")
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='../processed_dataset', help='输出目录')
    parser.add_argument('--sampling_strategy', type=str, default='multiscale_hybrid', 
                        choices=['uniform', 'adaptive_density', 'curvature_based', 'multiscale_hybrid'],
                        help='采样策略')
    parser.add_argument('--storage_mode', type=str, default='hierarchical',
                        choices=['single_array', 'scattered', 'hierarchical'],
                        help='存储模式')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--parallel', action='store_true', help='是否使用并行处理')
    parser.add_argument('--num_workers', type=int, default=4, help='并行处理的工作进程数')
    parser.add_argument('--max_tactile_left', type=int, default=None, help='触觉左侧点云的最大点数')
    parser.add_argument('--max_tactile_right', type=int, default=None, help='触觉右侧点云的最大点数')
    parser.add_argument('--max_object', type=int, default=2500, help='物体点云的最大点数')
    parser.add_argument('--max_surface', type=int, default=25000, help='表面点云的最大点数')
    parser.add_argument('--max_context', type=int, default=10000, help='上下文点云的最大点数')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析数据集而不处理')
    parser.add_argument('--include_all_objects', action='store_true', help='包含所有物体，不排除问题物体')
    parser.add_argument('--excluded_objects', type=str, nargs='+', help='要排除的物体ID列表，例如 "071 072 073"')
    parser.add_argument('--specific_object', type=str, help='只处理特定物体ID，例如"000"')
    
    args = parser.parse_args()
    
    # 设置最大点数
    max_points = {
        'tactile_left': args.max_tactile_left,
        'tactile_right': args.max_tactile_right,
        'cropped_object': args.max_object,
        'cropped_surface': args.max_surface,
        'cropped_surface_context': args.max_context
    }
    
    # 处理排除物体参数
    excluded_objects = None
    if args.include_all_objects:
        excluded_objects = []
    elif args.excluded_objects:
        excluded_objects = args.excluded_objects
    
    # 创建数据集构建器
    builder = DatasetBuilder(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sampling_strategy=args.sampling_strategy,
        max_points=max_points,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        storage_mode=args.storage_mode,
        excluded_objects=excluded_objects
    )
    
    # 分析数据集结构
    dataset_structure = builder.analyze_dataset_structure()
    
    # 如果指定了特定物体ID，只保留该物体的数据
    if args.specific_object:
        if args.specific_object in dataset_structure:
            logger.info(f"只处理物体ID: {args.specific_object}")
            dataset_structure = {args.specific_object: dataset_structure[args.specific_object]}
            # 将筛选后的数据结构设置到builder实例
            builder.dataset_structure = dataset_structure
        else:
            logger.error(f"指定的物体ID {args.specific_object} 不存在于数据集中")
            return
    
    if args.analyze_only:
        logger.info("已完成数据集分析")
        return
    
    # 如果使用并行处理
    if args.parallel:
        logger.info(f"使用并行处理，进程数: {args.num_workers}")
        
        # 准备并行处理的参数
        process_args = []
        # 使用builder中保存的dataset_structure
        for obj_id, obj_data in builder.dataset_structure.items():
            for pose_id, pose_data in obj_data['poses'].items():
                for grasp_id in pose_data['grasps']:
                    process_args.append((builder, obj_id, pose_id, grasp_id))
        
        logger.info(f"共有 {len(process_args)} 个样本需要处理")
        
        # 使用多进程池并行处理
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            all_samples = list(tqdm(pool.imap(process_sample_parallel, process_args), total=len(process_args), desc="并行处理样本"))
        
        # 过滤空样本
        all_samples = [sample for sample in all_samples if sample is not None]
        
        logger.info(f"共处理 {len(all_samples)} 个有效样本")
        
        # 划分数据集
        train_samples, val_samples, test_samples = builder.split_dataset(all_samples)
        
        logger.info(f"训练集: {len(train_samples)} 个样本")
        logger.info(f"验证集: {len(val_samples)} 个样本")
        logger.info(f"测试集: {len(test_samples)} 个样本")
        
        # 保存数据集
        if args.storage_mode == 'single_array':
            builder.save_single_array(train_samples, 'train')
            builder.save_single_array(val_samples, 'val')
            builder.save_single_array(test_samples, 'test')
        elif args.storage_mode == 'scattered':
            builder.save_scattered(train_samples, 'train')
            builder.save_scattered(val_samples, 'val')
            builder.save_scattered(test_samples, 'test')
        elif args.storage_mode == 'hierarchical':
            builder.save_hierarchical(train_samples, 'train')
            builder.save_hierarchical(val_samples, 'val')
            builder.save_hierarchical(test_samples, 'test')
        else:
            logger.error(f"未知的存储模式: {args.storage_mode}")
    else:
        # 使用常规处理
        builder.process_dataset()


if __name__ == '__main__':
    main()
