#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import time

def load_point_cloud(file_path):
    """
    加载点云文件
    
    参数:
        file_path: 点云文件路径
    
    返回:
        点云对象
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"警告: 点云为空 {file_path}")
            return None
        return pcd
    except Exception as e:
        print(f"错误: 无法加载点云 {file_path}: {str(e)}")
        return None

def create_coordinate_frame(size=0.1):
    """创建坐标系可视化"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def combine_point_clouds(pcds, colors=None):
    """
    将多个点云合并为一个，并为每个点云分配不同颜色
    
    参数:
        pcds: 点云对象列表
        colors: 颜色列表，每个颜色对应一个点云
        
    返回:
        合并后的点云
    """
    if not pcds:
        return None
    
    # 过滤掉None值
    pcds = [p for p in pcds if p is not None]
    if not pcds:
        return None
    
    # 如果没有提供颜色，使用默认颜色
    if colors is None:
        colors = [
            [1, 0, 0],  # 红色
            [0, 1, 0],  # 绿色
            [0, 0, 1],  # 蓝色
            [1, 1, 0],  # 黄色
            [1, 0, 1],  # 紫色
        ]
    
    # 创建合并点云
    combined_pcd = o3d.geometry.PointCloud()
    
    for i, pcd in enumerate(pcds):
        # 为当前点云分配颜色
        color = colors[i % len(colors)]
        pcd_colored = o3d.geometry.PointCloud(pcd)
        pcd_colored.paint_uniform_color(color)
        
        # 合并到结果点云
        combined_pcd += pcd_colored
    
    return combined_pcd

def visualize_dataset(dataset_dir, obj_range=None, pose_range=None, grasp_range=None):
    """
    可视化数据集中的点云数据
    
    参数:
        dataset_dir: 数据集根目录
        obj_range: 物体范围，格式为(start, end)
        pose_range: 位姿范围，格式为(start, end)
        grasp_range: 抓取范围，格式为(start, end)
    """
    # 如果未指定范围，使用默认值
    if obj_range is None:
        obj_range = (0, 88)
    if pose_range is None:
        pose_range = (0, 24)
    if grasp_range is None:
        grasp_range = (0, 100)  # 设置一个较大的值，实际会根据文件夹数量确定
    
    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="点云数据集可视化", width=1600, height=900)
    
    # 添加坐标系
    coordinate_frame = create_coordinate_frame()
    vis.add_geometry(coordinate_frame)
    
    # 获取所有物体目录
    obj_dirs = []
    for i in range(obj_range[0], obj_range[1]):
        obj_dir = os.path.join(dataset_dir, f"{i:03d}")
        if os.path.exists(obj_dir):
            obj_dirs.append(obj_dir)
    
    if not obj_dirs:
        print(f"错误: 未找到物体目录 {dataset_dir}")
        return
    
    print(f"找到 {len(obj_dirs)} 个物体目录")
    
    # 遍历所有物体、位姿和抓取
    current_obj_idx = 0
    current_pose_idx = 0
    current_grasp_idx = 0
    
    # 定义键盘回调函数
    def next_obj(vis):
        nonlocal current_obj_idx, current_pose_idx, current_grasp_idx
        current_obj_idx = (current_obj_idx + 1) % len(obj_dirs)
        current_pose_idx = 0
        current_grasp_idx = 0
        update_visualization()
        return False
    
    def prev_obj(vis):
        nonlocal current_obj_idx, current_pose_idx, current_grasp_idx
        current_obj_idx = (current_obj_idx - 1) % len(obj_dirs)
        current_pose_idx = 0
        current_grasp_idx = 0
        update_visualization()
        return False
    
    def next_pose(vis):
        nonlocal current_pose_idx, current_grasp_idx
        pose_dirs = get_pose_dirs()
        if pose_dirs:
            current_pose_idx = (current_pose_idx + 1) % len(pose_dirs)
            current_grasp_idx = 0
            update_visualization()
        return False
    
    def prev_pose(vis):
        nonlocal current_pose_idx, current_grasp_idx
        pose_dirs = get_pose_dirs()
        if pose_dirs:
            current_pose_idx = (current_pose_idx - 1) % len(pose_dirs)
            current_grasp_idx = 0
            update_visualization()
        return False
    
    def next_grasp(vis):
        nonlocal current_grasp_idx
        grasp_dirs = get_grasp_dirs()
        if grasp_dirs:
            current_grasp_idx = (current_grasp_idx + 1) % len(grasp_dirs)
            update_visualization()
        return False
    
    def prev_grasp(vis):
        nonlocal current_grasp_idx
        grasp_dirs = get_grasp_dirs()
        if grasp_dirs:
            current_grasp_idx = (current_grasp_idx - 1) % len(grasp_dirs)
            update_visualization()
        return False
    
    # 注册键盘回调
    vis.register_key_callback(ord('D'), next_obj)      # 下一个物体
    vis.register_key_callback(ord('A'), prev_obj)      # 上一个物体
    vis.register_key_callback(ord('W'), next_pose)     # 下一个位姿
    vis.register_key_callback(ord('S'), prev_pose)     # 上一个位姿
    vis.register_key_callback(ord('E'), next_grasp)    # 下一个抓取
    vis.register_key_callback(ord('Q'), prev_grasp)    # 上一个抓取
    
    # 辅助函数，获取当前物体的所有位姿目录
    def get_pose_dirs():
        obj_dir = obj_dirs[current_obj_idx]
        pose_dirs = []
        for i in range(pose_range[0], pose_range[1]):
            pose_dir = os.path.join(obj_dir, f"pose_{i:03d}")
            if os.path.exists(pose_dir):
                pose_dirs.append(pose_dir)
        return pose_dirs
    
    # 辅助函数，获取当前位姿的所有抓取目录
    def get_grasp_dirs():
        pose_dirs = get_pose_dirs()
        if not pose_dirs or current_pose_idx >= len(pose_dirs):
            return []
        
        pose_dir = pose_dirs[current_pose_idx]
        grasp_dirs = []
        
        # 查找所有抓取目录（格式为: xxx_grasp_xxx）
        grasp_pattern = os.path.join(pose_dir, f"*_grasp_*")
        grasp_dirs = sorted(glob.glob(grasp_pattern))
        
        return grasp_dirs
    
    # 当前显示的几何体
    current_geometries = []
    
    # 更新可视化
    def update_visualization():
        nonlocal current_geometries
        
        # 清除当前几何体
        for geom in current_geometries:
            vis.remove_geometry(geom, reset_bounding_box=False)
        current_geometries = []
        
        # 获取当前抓取目录
        grasp_dirs = get_grasp_dirs()
        if not grasp_dirs or current_grasp_idx >= len(grasp_dirs):
            print("没有找到抓取数据")
            return
        
        grasp_dir = grasp_dirs[current_grasp_idx]
        obj_id = os.path.basename(obj_dirs[current_obj_idx])
        
        # 加载点云数据
        # 1. 加载物体点云
        cropped_object_file = os.path.join(grasp_dir, f"{obj_id}_cropped_object.pcd")
        cropped_object_pcd = load_point_cloud(cropped_object_file)
        
        # 2. 加载左右触觉点云
        tactile_left_file = os.path.join(grasp_dir, f"{obj_id}_tactile_left.pcd")
        tactile_right_file = os.path.join(grasp_dir, f"{obj_id}_tactile_right.pcd")
        tactile_left_pcd = load_point_cloud(tactile_left_file)
        tactile_right_pcd = load_point_cloud(tactile_right_file)
        
        # 3. 加载表面点云（局部和上下文）
        cropped_surface_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface.pcd")
        cropped_surface_context_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface_context.pcd")
        cropped_surface_pcd = load_point_cloud(cropped_surface_file)
        cropped_surface_context_pcd = load_point_cloud(cropped_surface_context_file)
        
        # 创建组合点云
        # 组合1: 物体点云 + 左右触觉点云
        input_pcds = [cropped_object_pcd, tactile_left_pcd, tactile_right_pcd]
        input_colors = [
            [0.7, 0.7, 0.7],  # 灰色 - 物体点云
            [1.0, 0.0, 0.0],  # 红色 - 左触觉
            [0.0, 0.0, 1.0]   # 蓝色 - 右触觉
        ]
        combined_input = combine_point_clouds(input_pcds, input_colors)
        
        # 组合2: 局部表面点云
        if cropped_surface_pcd is not None:
            cropped_surface_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        
        # 组合3: 上下文表面点云
        if cropped_surface_context_pcd is not None:
            cropped_surface_context_pcd.paint_uniform_color([1.0, 0.5, 0.0])  # 橙色
        
        # 添加到可视化器
        if combined_input is not None:
            vis.add_geometry(combined_input, reset_bounding_box=False)
            current_geometries.append(combined_input)
        
        if cropped_surface_pcd is not None:
            # 将局部表面点云向右移动0.2米
            cropped_surface_pcd.translate([0.2, 0, 0])
            vis.add_geometry(cropped_surface_pcd, reset_bounding_box=False)
            current_geometries.append(cropped_surface_pcd)
        
        if cropped_surface_context_pcd is not None:
            # 将上下文表面点云向右移动0.4米
            cropped_surface_context_pcd.translate([0.4, 0, 0])
            vis.add_geometry(cropped_surface_context_pcd, reset_bounding_box=False)
            current_geometries.append(cropped_surface_context_pcd)
        
        # 更新标题
        pose_dirs = get_pose_dirs()
        pose_id = os.path.basename(pose_dirs[current_pose_idx]) if pose_dirs else "未知"
        grasp_id = os.path.basename(grasp_dir)
        window_title = f"物体: {obj_id}, 位姿: {pose_id}, 抓取: {grasp_id}"
        vis.get_render_option().background_color = [0.1, 0.1, 0.1]  # 深灰色背景
        
        # 打印当前状态
        print(f"\n当前显示: {window_title}")
        print(f"按键指南: A/D - 上一个/下一个物体, W/S - 上一个/下一个位姿, Q/E - 上一个/下一个抓取")
        
        # 更新视图
        vis.update_renderer()
    
    # 初始化可视化
    update_visualization()
    
    # 运行可视化器
    print("\n=== 点云数据集可视化 ===")
    print("按键指南:")
    print("  A/D - 上一个/下一个物体")
    print("  W/S - 上一个/下一个位姿")
    print("  Q/E - 上一个/下一个抓取")
    print("  ESC - 退出")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云数据集可视化工具")
    parser.add_argument("--dataset_dir", type=str, default="/home/iccd-simulator/code/vt-bullet-env/dataset",
                       help="数据集根目录")
    parser.add_argument("--obj_start", type=int, default=0, help="起始物体索引")
    parser.add_argument("--obj_end", type=int, default=88, help="结束物体索引")
    parser.add_argument("--pose_start", type=int, default=0, help="起始位姿索引")
    parser.add_argument("--pose_end", type=int, default=24, help="结束位姿索引")
    
    args = parser.parse_args()
    
    visualize_dataset(
        args.dataset_dir,
        obj_range=(args.obj_start, args.obj_end),
        pose_range=(args.pose_start, args.pose_end)
    )
