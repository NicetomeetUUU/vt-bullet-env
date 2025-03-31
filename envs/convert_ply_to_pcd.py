#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm

def convert_ply_to_pcd(model_dir):
    """
    将指定目录下的object_surface_points.ply文件转换为object_surface_points.pcd文件
    
    参数:
        model_dir: 模型目录路径
    """
    # 获取所有子目录（000-087）
    subdirs = sorted(glob.glob(os.path.join(model_dir, "[0-9][0-9][0-9]")))
    
    print(f"找到{len(subdirs)}个对象目录")
    
    for subdir in tqdm(subdirs):
        ply_file = os.path.join(subdir, "object_surface_points.ply")
        pcd_file = os.path.join(subdir, "object_surface_points.pcd")
        
        # 检查ply文件是否存在
        if not os.path.exists(ply_file):
            print(f"警告: {ply_file} 不存在，跳过")
            continue
        
        # 检查pcd文件是否已存在
        if os.path.exists(pcd_file):
            print(f"信息: {pcd_file} 已存在，跳过")
            continue
        
        try:
            # 读取PLY文件
            pcd = o3d.io.read_point_cloud(ply_file)
            
            # 保存为PCD文件
            o3d.io.write_point_cloud(pcd_file, pcd)
            
            print(f"成功转换: {ply_file} -> {pcd_file}")
        except Exception as e:
            print(f"错误: 转换 {ply_file} 失败: {str(e)}")

if __name__ == "__main__":
    # 设置模型目录路径
    model_dir = "/home/iccd-simulator/code/vt-bullet-env/models"
    
    # 执行转换
    convert_ply_to_pcd(model_dir)
    
    print("转换完成！")
