#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云数据集分析工具
用于查看和分析hierarchical模式下保存的数据集结构
"""

import os
import argparse
import h5py
import numpy as np
import logging
import sys
import json
import open3d as o3d
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('dataset_analyzer')

def visualize_point_cloud(points, colors=None, window_name="点云可视化"):
    """可视化点云
    
    参数:
        points: 点云坐标，形状为(N, 3)
        colors: 点云颜色，形状为(N, 3)，可选
        window_name: 窗口名称
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # 确保颜色值在[0,1]范围内
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 默认使用红色
        pcd.paint_uniform_color([1, 0, 0])
    
    # 创建坐标系可视化
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # 显示点云和坐标系
    o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=window_name)

def analyze_h5_file(file_path):
    """分析HDF5文件结构
    
    参数:
        file_path: HDF5文件路径
    
    返回:
        文件结构信息
    """
    logger.info(f"分析文件: {file_path}")
    
    result = {
        'file_path': file_path,
        'groups': {},
        'datasets': {},
        'attributes': {}
    }
    
    def visit_item(name, obj):
        if isinstance(obj, h5py.Group):
            result['groups'][name] = {
                'attrs': dict(obj.attrs.items()),
                'items': list(obj.keys())
            }
        elif isinstance(obj, h5py.Dataset):
            result['datasets'][name] = {
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'attrs': dict(obj.attrs.items())
            }
            # 如果数据集不太大，获取一些示例数据
            if obj.size < 100:
                result['datasets'][name]['data'] = obj[()]
        
        return None
    
    with h5py.File(file_path, 'r') as f:
        # 获取文件属性
        result['attributes'] = dict(f.attrs.items())
        
        # 遍历文件中的所有项
        f.visititems(visit_item)
    
    return result

def print_h5_structure(structure, indent=0):
    """打印HDF5文件结构
    
    参数:
        structure: 文件结构信息
        indent: 缩进级别
    """
    print(f"文件路径: {structure['file_path']}")
    
    if structure['attributes']:
        print("文件属性:")
        for key, value in structure['attributes'].items():
            print(f"  {key}: {value}")
    
    print("\n组:")
    for name, info in structure['groups'].items():
        print(f"{'  ' * (indent + 1)}{name}/")
        if info['attrs']:
            print(f"{'  ' * (indent + 2)}属性:")
            for key, value in info['attrs'].items():
                print(f"{'  ' * (indent + 3)}{key}: {value}")
        print(f"{'  ' * (indent + 2)}子项: {', '.join(info['items'])}")
    
    print("\n数据集:")
    for name, info in structure['datasets'].items():
        print(f"{'  ' * (indent + 1)}{name}: 形状={info['shape']}, 类型={info['dtype']}")
        if 'data' in info:
            print(f"{'  ' * (indent + 2)}数据: {info['data']}")
        if info['attrs']:
            print(f"{'  ' * (indent + 2)}属性:")
            for key, value in info['attrs'].items():
                print(f"{'  ' * (indent + 3)}{key}: {value}")

def visualize_sample(file_path, pcd_type, sample_index=0):
    """可视化特定样本的点云
    
    参数:
        file_path: HDF5文件路径
        pcd_type: 点云类型，如'tactile_left', 'cropped_object'等
        sample_index: 样本索引
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if f"{pcd_type}/points" not in f:
                logger.error(f"文件中不存在 {pcd_type}/points 数据集")
                return
            
            # 获取点云数据
            points = f[f"{pcd_type}/points"][:]
            
            # 检查样本索引是否有效
            if sample_index >= len(points):
                logger.error(f"样本索引 {sample_index} 超出范围，最大索引为 {len(points) - 1}")
                return
            
            # 获取特定样本的点云
            sample_points = points[sample_index]
            
            # 尝试获取颜色数据
            colors = None
            if f"{pcd_type}/colors" in f:
                colors_data = f[f"{pcd_type}/colors"][:]
                if sample_index < len(colors_data):
                    colors = colors_data[sample_index]
            
            # 可视化点云
            logger.info(f"可视化 {pcd_type} 点云，样本索引: {sample_index}")
            visualize_point_cloud(sample_points, colors, f"{pcd_type} - 样本 {sample_index}")
            
    except Exception as e:
        logger.error(f"可视化点云时出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="点云数据集分析工具")
    parser.add_argument('--dataset_dir', type=str, default='./processed_dataset', help='处理后的数据集目录')
    parser.add_argument('--obj_id', type=str, default='000', help='要分析的物体ID')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='数据集划分')
    parser.add_argument('--visualize', action='store_true', help='是否可视化点云')
    parser.add_argument('--pcd_type', type=str, default='cropped_object', 
                        choices=['tactile_left', 'tactile_right', 'cropped_object', 'cropped_surface', 'cropped_surface_context'],
                        help='要可视化的点云类型')
    parser.add_argument('--sample_index', type=int, default=0, help='要可视化的样本索引')
    
    args = parser.parse_args()
    
    # 构建文件路径
    file_path = os.path.join(args.dataset_dir, args.split, f"{args.obj_id}.h5")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return
    
    # 分析文件结构
    structure = analyze_h5_file(file_path)
    print_h5_structure(structure)
    
    # 如果需要可视化点云
    if args.visualize:
        visualize_sample(file_path, args.pcd_type, args.sample_index)

if __name__ == '__main__':
    main()
