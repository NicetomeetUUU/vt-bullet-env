#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 定义数据集路径（可通过命令行参数覆盖）
DATASET_DIR = "/home/iccd-simulator/code/vt-bullet-env/dataset"

def load_point_cloud(file_path):
    """加载点云文件"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"警告: 点云为空 {file_path}")
            return None
        return np.asarray(pcd.points), pcd
    except Exception as e:
        print(f"错误: 无法加载点云 {file_path}: {str(e)}")
        return None, None

def analyze_point_cloud(points):
    """分析点云的基本特性"""
    if points is None or len(points) == 0:
        return None
    
    # 计算点云的基本统计信息
    stats = {
        "num_points": len(points),
        "min_coords": np.min(points, axis=0),
        "max_coords": np.max(points, axis=0),
        "mean_coords": np.mean(points, axis=0),
        "std_coords": np.std(points, axis=0),
        "bbox_size": np.max(points, axis=0) - np.min(points, axis=0),
        "centroid": np.mean(points, axis=0)
    }
    
    # 计算点云密度
    bbox_volume = np.prod(stats["bbox_size"])
    if bbox_volume > 0:
        stats["point_density"] = stats["num_points"] / bbox_volume
    else:
        stats["point_density"] = 0
    
    return stats

def analyze_point_distribution(points):
    """分析点云的分布特性"""
    if points is None or len(points) == 0:
        return None
    
    # 计算点到质心的距离
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # 计算距离统计
    dist_stats = {
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "mean_distance": np.mean(distances),
        "median_distance": np.median(distances),
        "std_distance": np.std(distances)
    }
    
    # 计算距离分布
    hist, bin_edges = np.histogram(distances, bins=10)
    dist_stats["distance_histogram"] = hist
    dist_stats["distance_bins"] = bin_edges
    
    return dist_stats

def estimate_complexity(points, pcd=None):
    """估计点云的几何复杂度"""
    if points is None or len(points) < 10:
        return None
    
    # 如果没有提供pcd对象，创建一个
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
    # 计算法线
    if len(points) > 10:  # 需要足够的点来估计法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        
        # 计算法线变化
        complexity = {
            "normal_variation": np.std(np.linalg.norm(normals, axis=1))
        }
        
        # 尝试计算曲率
        try:
            pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            eigenvalues = []
            for i in range(len(points)):
                cov = pcd.covariances[i]
                if cov is not None and cov.shape == (3, 3):
                    eigvals = np.linalg.eigvalsh(cov)
                    if not np.isnan(eigvals).any():
                        eigenvalues.append(eigvals)
            
            if eigenvalues:
                eigenvalues = np.array(eigenvalues)
                # 曲率估计 (最小特征值 / 特征值之和)
                curvatures = eigenvalues[:, 0] / (eigenvalues.sum(axis=1) + 1e-10)
                complexity["mean_curvature"] = np.mean(curvatures)
                complexity["max_curvature"] = np.max(curvatures)
            else:
                complexity["mean_curvature"] = None
                complexity["max_curvature"] = None
        except Exception as e:
            print(f"计算曲率时出错: {str(e)}")
            complexity["mean_curvature"] = None
            complexity["max_curvature"] = None
    else:
        complexity = {"normal_variation": None, "mean_curvature": None, "max_curvature": None}
    
    return complexity

def analyze_dataset_structure(dataset_dir):
    """分析数据集的目录结构"""
    # 获取所有物体目录
    obj_dirs = []
    for i in range(0, 88):
        obj_dir = os.path.join(dataset_dir, f"{i:03d}")
        if os.path.exists(obj_dir):
            obj_dirs.append(obj_dir)
    
    structure = {
        "num_objects": len(obj_dirs),
        "objects": {}
    }
    
    # 随机选择几个物体进行详细分析
    sample_objs = random.sample(obj_dirs, min(5, len(obj_dirs)))
    
    for obj_dir in sample_objs:
        obj_id = os.path.basename(obj_dir)
        
        # 获取位姿目录
        pose_dirs = []
        for i in range(0, 24):
            pose_dir = os.path.join(obj_dir, f"pose_{i:03d}")
            if os.path.exists(pose_dir):
                pose_dirs.append(pose_dir)
        
        structure["objects"][obj_id] = {
            "num_poses": len(pose_dirs),
            "poses": {}
        }
        
        # 随机选择几个位姿进行详细分析
        sample_poses = random.sample(pose_dirs, min(3, len(pose_dirs)))
        
        for pose_dir in sample_poses:
            pose_id = os.path.basename(pose_dir)
            
            # 查找所有抓取目录
            grasp_pattern = os.path.join(pose_dir, f"*_grasp_*")
            grasp_dirs = sorted(glob.glob(grasp_pattern))
            
            structure["objects"][obj_id]["poses"][pose_id] = {
                "num_grasps": len(grasp_dirs),
                "grasps": {}
            }
            
            # 随机选择几个抓取进行详细分析
            sample_grasps = random.sample(grasp_dirs, min(2, len(grasp_dirs)))
            
            for grasp_dir in sample_grasps:
                grasp_id = os.path.basename(grasp_dir)
                structure["objects"][obj_id]["poses"][pose_id]["grasps"][grasp_id] = {
                    "path": grasp_dir
                }
    
    return structure

def analyze_point_clouds(dataset_structure, dataset_dir):
    """分析数据集中的点云文件"""
    results = {
        "tactile_left": [],
        "tactile_right": [],
        "cropped_object": [],
        "cropped_surface": [],
        "cropped_surface_context": []
    }
    
    # 遍历样本物体
    for obj_id, obj_info in dataset_structure["objects"].items():
        # 遍历样本位姿
        for pose_id, pose_info in obj_info["poses"].items():
            # 遍历样本抓取
            for grasp_id, grasp_info in pose_info["grasps"].items():
                grasp_dir = grasp_info["path"]
                
                # 加载点云数据
                tactile_left_file = os.path.join(grasp_dir, f"{obj_id}_tactile_left.pcd")
                tactile_right_file = os.path.join(grasp_dir, f"{obj_id}_tactile_right.pcd")
                cropped_object_file = os.path.join(grasp_dir, f"{obj_id}_cropped_object.pcd")
                cropped_surface_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface.pcd")
                cropped_surface_context_file = os.path.join(grasp_dir, f"{obj_id}_cropped_surface_context.pcd")
                
                # 分析每种点云
                for name, file_path in [
                    ("tactile_left", tactile_left_file),
                    ("tactile_right", tactile_right_file),
                    ("cropped_object", cropped_object_file),
                    ("cropped_surface", cropped_surface_file),
                    ("cropped_surface_context", cropped_surface_context_file)
                ]:
                    points_data = load_point_cloud(file_path)
                    if points_data is not None:
                        points, pcd = points_data
                        if points is not None:
                            # 基本分析
                            stats = analyze_point_cloud(points)
                            # 分布分析
                            dist_stats = analyze_point_distribution(points)
                            # 复杂度分析
                            complexity = estimate_complexity(points, pcd)
                            
                            # 合并结果
                            result = {
                                "obj_id": obj_id,
                                "pose_id": pose_id,
                                "grasp_id": grasp_id,
                                "stats": stats,
                                "distribution": dist_stats,
                                "complexity": complexity
                            }
                            
                            results[name].append(result)
    
    return results

def plot_point_count_distribution(results):
    """绘制点云数量分布"""
    plt.figure(figsize=(12, 8))
    
    for i, (name, data) in enumerate(results.items()):
        if data:
            point_counts = [item["stats"]["num_points"] for item in data if item["stats"] is not None]
            if point_counts:
                plt.subplot(2, 3, i+1)
                plt.hist(point_counts, bins=20)
                plt.title(f"{name} 点数分布")
                plt.xlabel("点数")
                plt.ylabel("频率")
                
                # 添加统计信息
                plt.axvline(np.mean(point_counts), color='r', linestyle='dashed', linewidth=1)
                plt.text(0.7, 0.9, f"平均: {np.mean(point_counts):.1f}\n中位数: {np.median(point_counts):.1f}\n最小: {np.min(point_counts)}\n最大: {np.max(point_counts)}",
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("point_count_distribution.png")
    plt.close()
    print("点云数量分布图已保存为 point_count_distribution.png")

def plot_density_distribution(results):
    """绘制点云密度分布"""
    plt.figure(figsize=(12, 8))
    
    for i, (name, data) in enumerate(results.items()):
        if data:
            densities = [item["stats"]["point_density"] for item in data if item["stats"] is not None]
            if densities:
                plt.subplot(2, 3, i+1)
                plt.hist(densities, bins=20)
                plt.title(f"{name} 密度分布")
                plt.xlabel("密度 (点/立方米)")
                plt.ylabel("频率")
                
                # 添加统计信息
                plt.axvline(np.mean(densities), color='r', linestyle='dashed', linewidth=1)
                plt.text(0.7, 0.9, f"平均: {np.mean(densities):.1f}\n中位数: {np.median(densities):.1f}\n最小: {np.min(densities):.1f}\n最大: {np.max(densities):.1f}",
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("density_distribution.png")
    plt.close()
    print("点云密度分布图已保存为 density_distribution.png")

def plot_bounding_box_sizes(results):
    """绘制点云包围盒大小分布"""
    plt.figure(figsize=(15, 10))
    
    for i, (name, data) in enumerate(results.items()):
        if data:
            # 提取x, y, z尺寸
            x_sizes = [item["stats"]["bbox_size"][0] for item in data if item["stats"] is not None]
            y_sizes = [item["stats"]["bbox_size"][1] for item in data if item["stats"] is not None]
            z_sizes = [item["stats"]["bbox_size"][2] for item in data if item["stats"] is not None]
            
            if x_sizes and y_sizes and z_sizes:
                plt.subplot(2, 3, i+1)
                plt.boxplot([x_sizes, y_sizes, z_sizes], labels=['X', 'Y', 'Z'])
                plt.title(f"{name} 包围盒尺寸")
                plt.ylabel("尺寸 (米)")
                
                # 添加统计信息
                avg_size = np.mean([x_sizes, y_sizes, z_sizes], axis=1)
                plt.text(0.7, 0.9, f"平均尺寸:\nX: {avg_size[0]:.4f}\nY: {avg_size[1]:.4f}\nZ: {avg_size[2]:.4f}",
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("bounding_box_sizes.png")
    plt.close()
    print("点云包围盒尺寸分布图已保存为 bounding_box_sizes.png")

def plot_complexity_metrics(results):
    """绘制点云复杂度指标"""
    plt.figure(figsize=(15, 10))
    
    for i, (name, data) in enumerate(results.items()):
        if data:
            # 提取曲率信息
            curvatures = [item["complexity"]["mean_curvature"] for item in data 
                         if item["complexity"] is not None and item["complexity"]["mean_curvature"] is not None]
            
            if curvatures:
                plt.subplot(2, 3, i+1)
                plt.hist(curvatures, bins=20)
                plt.title(f"{name} 平均曲率分布")
                plt.xlabel("平均曲率")
                plt.ylabel("频率")
                
                # 添加统计信息
                plt.axvline(np.mean(curvatures), color='r', linestyle='dashed', linewidth=1)
                plt.text(0.7, 0.9, f"平均: {np.mean(curvatures):.6f}\n中位数: {np.median(curvatures):.6f}\n最小: {np.min(curvatures):.6f}\n最大: {np.max(curvatures):.6f}",
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("complexity_metrics.png")
    plt.close()
    print("点云复杂度指标分布图已保存为 complexity_metrics.png")

def analyze_spatial_relationships(results):
    """分析不同点云之间的空间关系"""
    relationships = []
    
    # 获取所有样本的obj_id, pose_id, grasp_id组合
    sample_ids = set()
    for data in results.values():
        for item in data:
            sample_ids.add((item["obj_id"], item["pose_id"], item["grasp_id"]))
    
    # 对每个样本，分析不同点云之间的关系
    for obj_id, pose_id, grasp_id in sample_ids:
        # 获取每种点云的质心
        centroids = {}
        for name, data in results.items():
            for item in data:
                if (item["obj_id"] == obj_id and 
                    item["pose_id"] == pose_id and 
                    item["grasp_id"] == grasp_id and 
                    item["stats"] is not None):
                    centroids[name] = item["stats"]["centroid"]
        
        # 如果至少有两种点云，计算它们之间的距离
        if len(centroids) >= 2:
            rel = {
                "obj_id": obj_id,
                "pose_id": pose_id,
                "grasp_id": grasp_id,
                "centroid_distances": {}
            }
            
            # 计算每对点云之间的质心距离
            for name1 in centroids:
                for name2 in centroids:
                    if name1 < name2:  # 避免重复计算
                        dist = np.linalg.norm(centroids[name1] - centroids[name2])
                        rel["centroid_distances"][f"{name1}_to_{name2}"] = dist
            
            relationships.append(rel)
    
    return relationships

def print_summary_statistics(results, relationships):
    """打印数据集的汇总统计信息"""
    print("\n===== 点云数据集分析汇总 =====\n")
    
    # 1. 点云数量统计
    print("1. 点云数量统计:")
    for name, data in results.items():
        if data:
            point_counts = [item["stats"]["num_points"] for item in data if item["stats"] is not None]
            if point_counts:
                print(f"  {name}:")
                print(f"    - 样本数: {len(point_counts)}")
                print(f"    - 平均点数: {np.mean(point_counts):.1f}")
                print(f"    - 中位数点数: {np.median(point_counts):.1f}")
                print(f"    - 最小点数: {np.min(point_counts)}")
                print(f"    - 最大点数: {np.max(point_counts)}")
                print(f"    - 标准差: {np.std(point_counts):.1f}")
    
    # 2. 点云尺寸统计
    print("\n2. 点云包围盒尺寸统计 (米):")
    for name, data in results.items():
        if data:
            x_sizes = [item["stats"]["bbox_size"][0] for item in data if item["stats"] is not None]
            y_sizes = [item["stats"]["bbox_size"][1] for item in data if item["stats"] is not None]
            z_sizes = [item["stats"]["bbox_size"][2] for item in data if item["stats"] is not None]
            
            if x_sizes and y_sizes and z_sizes:
                print(f"  {name}:")
                print(f"    - X轴平均尺寸: {np.mean(x_sizes):.4f}")
                print(f"    - Y轴平均尺寸: {np.mean(y_sizes):.4f}")
                print(f"    - Z轴平均尺寸: {np.mean(z_sizes):.4f}")
                print(f"    - 平均体积: {np.mean(np.multiply(x_sizes, np.multiply(y_sizes, z_sizes))):.6f}")
    
    # 3. 点云密度统计
    print("\n3. 点云密度统计 (点/立方米):")
    for name, data in results.items():
        if data:
            densities = [item["stats"]["point_density"] for item in data if item["stats"] is not None]
            if densities:
                print(f"  {name}:")
                print(f"    - 平均密度: {np.mean(densities):.1f}")
                print(f"    - 中位数密度: {np.median(densities):.1f}")
                print(f"    - 最小密度: {np.min(densities):.1f}")
                print(f"    - 最大密度: {np.max(densities):.1f}")
    
    # 4. 点云复杂度统计
    print("\n4. 点云复杂度统计:")
    for name, data in results.items():
        if data:
            curvatures = [item["complexity"]["mean_curvature"] for item in data 
                         if item["complexity"] is not None and item["complexity"]["mean_curvature"] is not None]
            
            if curvatures:
                print(f"  {name}:")
                print(f"    - 平均曲率: {np.mean(curvatures):.6f}")
                print(f"    - 中位数曲率: {np.median(curvatures):.6f}")
                print(f"    - 曲率范围: {np.min(curvatures):.6f} - {np.max(curvatures):.6f}")
    
    # 5. 点云间距离关系
    print("\n5. 点云间质心距离统计 (米):")
    if relationships:
        # 收集所有距离对
        distance_pairs = defaultdict(list)
        for rel in relationships:
            for pair, dist in rel["centroid_distances"].items():
                distance_pairs[pair].append(dist)
        
        # 打印每对点云的距离统计
        for pair, distances in distance_pairs.items():
            print(f"  {pair}:")
            print(f"    - 平均距离: {np.mean(distances):.4f}")
            print(f"    - 中位数距离: {np.median(distances):.4f}")
            print(f"    - 距离范围: {np.min(distances):.4f} - {np.max(distances):.4f}")
    
    print("\n===== 分析完成 =====")

def main():
    parser = argparse.ArgumentParser(description="点云数据集分析工具")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="数据集根目录")
    parser.add_argument("--sample_size", type=int, default=10, help="要分析的样本数量")
    
    args = parser.parse_args()
    
    # 更新数据集路径
    dataset_dir = args.dataset_dir
    
    print(f"分析数据集: {dataset_dir}")
    print("分析数据集结构...")
    structure = analyze_dataset_structure(dataset_dir)
    
    print(f"找到 {structure['num_objects']} 个物体")
    for obj_id, obj_info in structure["objects"].items():
        print(f"  物体 {obj_id}: {obj_info['num_poses']} 个位姿")
    
    print("\n分析点云数据...")
    results = analyze_point_clouds(structure, dataset_dir)
    
    print("分析点云间空间关系...")
    relationships = analyze_spatial_relationships(results)
    
    print("生成统计图表...")
    plot_point_count_distribution(results)
    plot_density_distribution(results)
    plot_bounding_box_sizes(results)
    plot_complexity_metrics(results)
    
    print_summary_statistics(results, relationships)

if __name__ == "__main__":
    main()
