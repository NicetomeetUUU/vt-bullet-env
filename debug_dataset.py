#!/usr/bin/env python3
import os
import argparse
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_structure(dataset_dir, check_files=False):
    """分析数据集的目录结构，统计物体、位姿和抓取数量"""
    logger.info(f"分析数据集目录: {dataset_dir}")
    
    # 获取所有物体目录
    obj_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    obj_dirs = sorted([d for d in obj_dirs if d.isdigit() or (len(d) == 3 and d.isdigit())])
    
    logger.info(f"找到 {len(obj_dirs)} 个物体目录")
    
    # 统计每个物体的位姿和抓取
    total_poses = 0
    total_grasps = 0
    total_valid_grasps = 0
    
    # 详细统计每个物体
    obj_stats = {}
    
    for obj_id in tqdm(obj_dirs, desc="分析物体"):
        obj_dir = os.path.join(dataset_dir, obj_id)
        pose_dirs = [d for d in os.listdir(obj_dir) if d.startswith('pose_')]
        
        obj_poses = len(pose_dirs)
        obj_grasps = 0
        obj_valid_grasps = 0
        
        for pose_dir in pose_dirs:
            pose_path = os.path.join(obj_dir, pose_dir)
            grasp_dirs = [d for d in os.listdir(pose_path) if os.path.isdir(os.path.join(pose_path, d)) and d.endswith('_grasp_000')]
            
            obj_grasps += len(grasp_dirs)
            
            if check_files:
                for grasp_dir in grasp_dirs:
                    grasp_path = os.path.join(pose_path, grasp_dir)
                    all_files_exist = True
                    
                    # 检查所有必要的点云文件是否存在
                    for pcd_type in ['tactile_left', 'tactile_right', 'cropped_object', 'cropped_surface', 'cropped_surface_context']:
                        pcd_file = f"{obj_id}_{pcd_type}.pcd"
                        pcd_path = os.path.join(grasp_path, pcd_file)
                        if not os.path.exists(pcd_path):
                            all_files_exist = False
                            break
                    
                    if all_files_exist:
                        obj_valid_grasps += 1
            else:
                obj_valid_grasps = obj_grasps  # 如果不检查文件，假设所有抓取都有效
        
        total_poses += obj_poses
        total_grasps += obj_grasps
        total_valid_grasps += obj_valid_grasps
        
        obj_stats[obj_id] = {
            'poses': obj_poses,
            'grasps': obj_grasps,
            'valid_grasps': obj_valid_grasps
        }
        
        logger.info(f"物体 {obj_id}: {obj_poses} 个位姿, {obj_grasps} 个抓取, {obj_valid_grasps} 个有效抓取")
    
    # 输出总体统计
    logger.info(f"总计: {len(obj_dirs)} 个物体, {total_poses} 个位姿, {total_grasps} 个抓取, {total_valid_grasps} 个有效抓取")
    
    # 特别检查物体 "000"
    if "000" in obj_stats:
        logger.info(f"物体 '000' 详情: {obj_stats['000']['poses']} 个位姿, {obj_stats['000']['grasps']} 个抓取, {obj_stats['000']['valid_grasps']} 个有效抓取")
    
    return obj_stats

def main():
    parser = argparse.ArgumentParser(description="数据集分析工具")
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='原始数据集目录')
    parser.add_argument('--check_files', action='store_true', help='是否检查点云文件是否存在')
    
    args = parser.parse_args()
    
    analyze_dataset_structure(args.dataset_dir, args.check_files)

if __name__ == "__main__":
    main()
