#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
调试特定物体（000）的点云样本处理
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

# 导入数据集构建器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.dataset_builder import DatasetBuilder

def check_grasp_poses_json(json_path):
    """检查抓取姿态JSON文件"""
    if not os.path.exists(json_path):
        logging.error(f"抓取姿态JSON文件不存在: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            grasp_poses = json.load(f)
        
        logging.info(f"抓取姿态JSON包含 {len(grasp_poses)} 个抓取")
        
        # 检查有效抓取数量
        valid_grasps = [g for g in grasp_poses if g.get('label', 0) == 1]
        logging.info(f"有效抓取数量: {len(valid_grasps)}/{len(grasp_poses)}")
        
        # 列出所有抓取ID
        grasp_ids = [g.get('id', 'unknown') for g in grasp_poses]
        logging.info(f"抓取IDs: {grasp_ids}")
        
        return grasp_poses
    except Exception as e:
        logging.error(f"解析JSON文件时出错: {e}")
        return None

def check_pose_directories(obj_dir):
    """检查物体目录中的位姿目录"""
    pose_dirs = [d for d in Path(obj_dir).glob("pose_*") if d.is_dir()]
    pose_ids = [d.name.split('_')[1] for d in pose_dirs]
    logging.info(f"找到 {len(pose_dirs)} 个位姿目录, 位姿IDs: {sorted(pose_ids)}")
    
    return sorted(pose_dirs)

def check_grasp_directories(pose_dir):
    """检查位姿目录中的抓取目录"""
    grasp_dirs = [d for d in Path(pose_dir).glob("*_grasp_*") if d.is_dir()]
    pose_id = pose_dir.name.split('_')[1]
    logging.info(f"位姿 {pose_id}: 找到 {len(grasp_dirs)} 个抓取目录")
    
    # 检查抓取姿态JSON文件
    pose_id_prefix = f"{pose_id.zfill(3)}"
    json_path = os.path.join(pose_dir, f"{pose_id_prefix}_grasp_poses.json")
    grasp_poses = check_grasp_poses_json(json_path)
    
    return grasp_dirs, grasp_poses

def load_and_check_point_cloud(pcd_path):
    """加载并检查点云文件"""
    if not os.path.exists(pcd_path):
        logging.error(f"点云文件不存在: {pcd_path}")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        logging.info(f"点云包含 {len(points)} 个点")
        
        if len(points) == 0:
            logging.warning("警告: 空点云!")
        
        return pcd
    except Exception as e:
        logging.error(f"加载点云时出错: {e}")
        return None

def check_grasp_directory_content(grasp_dir):
    """检查抓取目录内容"""
    files = list(Path(grasp_dir).glob("*"))
    file_names = [f.name for f in files]
    logging.info(f"抓取目录 {grasp_dir.name} 包含文件: {file_names}")
    
    # 检查是否有必要的文件
    required_files = ["rgb.png", "depth.png", "label.txt"]
    missing_files = [f for f in required_files if f not in file_names]
    
    if missing_files:
        logging.warning(f"缺少必要文件: {missing_files}")
    
    # 检查标签文件
    label_path = os.path.join(grasp_dir, "label.txt")
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
            logging.info(f"抓取标签: {label}")
        except Exception as e:
            logging.error(f"读取标签文件时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="调试物体000的点云样本")
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='./processed_dataset', help='输出目录')
    parser.add_argument('--check_files', action='store_true', help='检查文件完整性')
    parser.add_argument('--process_samples', action='store_true', help='处理样本')
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug_object_000.log')
        ]
    )
    
    # 物体ID
    obj_id = "000"
    obj_dir = os.path.join(args.dataset_dir, obj_id)
    
    logging.info(f"开始调试物体 {obj_id}")
    logging.info(f"物体目录: {obj_dir}")
    
    if not os.path.exists(obj_dir):
        logging.error(f"物体目录不存在: {obj_dir}")
        return
    
    # 检查文件
    if args.check_files:
        logging.info("检查文件...")
        pose_dirs = check_pose_directories(obj_dir)
        
        # 检查每个位姿目录
        valid_grasp_count = 0
        total_grasp_count = 0
        
        for pose_dir in pose_dirs:
            grasp_dirs, grasp_poses = check_grasp_directories(pose_dir)
            
            # 检查相机点云文件
            pose_id = pose_dir.name.split('_')[1]
            pose_id_prefix = f"{pose_id.zfill(3)}"
            camera_pcd_path = os.path.join(pose_dir, f"{pose_id_prefix}_camera.pcd")
            object_pcd_path = os.path.join(pose_dir, f"{pose_id_prefix}_object.pcd")
            
            logging.info(f"检查相机点云文件: {camera_pcd_path}")
            camera_pcd = load_and_check_point_cloud(camera_pcd_path)
            
            logging.info(f"检查物体点云文件: {object_pcd_path}")
            object_pcd = load_and_check_point_cloud(object_pcd_path)
            
            # 检查每个抓取目录
            for grasp_dir in grasp_dirs:
                check_grasp_directory_content(grasp_dir)
                total_grasp_count += 1
            
            # 统计有效抓取
            if grasp_poses:
                valid_grasps = [g for g in grasp_poses if g.get('label', 0) == 1]
                valid_grasp_count += len(valid_grasps)
        
        logging.info(f"总计: {len(pose_dirs)} 个位姿, {total_grasp_count} 个抓取, {valid_grasp_count} 个有效抓取")
    
    # 处理样本 - 这部分需要修改DatasetBuilder类来适应新的目录结构
    if args.process_samples:
        logging.info("处理样本功能需要修改DatasetBuilder类来适应新的目录结构")

if __name__ == "__main__":
    main()
