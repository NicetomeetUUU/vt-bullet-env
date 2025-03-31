#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import open3d as o3d


def simplify_mesh(input_file, output_file, target_size_mb=2.0, verbose=True, max_iterations=5):
    """
    使用Open3D简化OBJ模型文件，确保输出文件大小不超过指定大小
    
    参数:
        input_file (str): 输入OBJ文件路径
        output_file (str): 输出OBJ文件路径
        target_size_mb (float): 目标文件大小，单位为MB
        verbose (bool): 是否打印详细信息
        max_iterations (int): 最大简化迭代次数
    
    返回:
        bool: 简化是否成功
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载网格
    if verbose:
        print(f"正在加载模型: {input_file}")
    
    mesh = o3d.io.read_triangle_mesh(input_file)
    
    # 输出原始网格信息
    if verbose:
        print(f"原始网格信息:")
        print(f"  - 顶点数量: {len(mesh.vertices)}")
        print(f"  - 三角形数量: {len(mesh.triangles)}")
    
    # 检查原始文件大小
    original_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    if verbose:
        print(f"原始文件大小: {original_size_mb:.2f} MB")
    
    if original_size_mb <= target_size_mb:
        print(f"文件已经小于目标大小 {target_size_mb} MB，无需简化")
        # 仍然保存一份副本作为输出
        o3d.io.write_triangle_mesh(output_file, mesh)
        return True
    
    # 初始简化比例 - 对于非常大的文件，我们使用更激进的初始简化比例
    initial_ratio = min(0.05, (target_size_mb * 0.9) / original_size_mb)
    original_triangles = len(mesh.triangles)
    current_mesh = mesh
    
    for iteration in range(max_iterations):
        # 计算当前迭代的目标三角形数量
        if iteration == 0:
            target_ratio = initial_ratio
        else:
            # 每次迭代减少一半的三角形数量
            target_ratio = target_ratio * 0.5
        
        target_triangles = max(100, int(original_triangles * target_ratio))
        
        if verbose:
            print(f"迭代 {iteration+1}/{max_iterations} - 目标简化比例: {target_ratio:.6f}")
            print(f"目标三角形数量: {target_triangles}")
        
        # 进行网格简化
        if verbose:
            print("正在简化网格...")
        
        # 使用四边形网格简化方法
        simplified_mesh = current_mesh.simplify_quadric_decimation(target_triangles)
        
        # 输出简化后的网格信息
        if verbose:
            print(f"简化后的网格信息:")
            print(f"  - 顶点数量: {len(simplified_mesh.vertices)}")
            print(f"  - 三角形数量: {len(simplified_mesh.triangles)}")
        
        # 保存简化后的网格
        if verbose:
            print(f"正在保存简化后的模型: {output_file}")
        
        o3d.io.write_triangle_mesh(output_file, simplified_mesh)
        
        # 检查输出文件大小
        if os.path.exists(output_file):
            output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            if verbose:
                print(f"输出文件大小: {output_size_mb:.2f} MB")
            
            # 如果达到目标大小，则返回成功
            if output_size_mb <= target_size_mb:
                if verbose:
                    print(f"成功达到目标大小，迭代 {iteration+1}/{max_iterations}")
                return True
            
            # 如果仍然太大，但已经达到最大迭代次数，则尝试更激进的简化
            if iteration == max_iterations - 1:
                # 最后一次尝试 - 极度简化
                extreme_target = max(50, int(target_triangles * 0.2))  # 只保留20%的三角形
                if verbose:
                    print(f"最终尝试 - 极度简化到 {extreme_target} 个三角形")
                
                simplified_mesh = mesh.simplify_quadric_decimation(extreme_target)
                o3d.io.write_triangle_mesh(output_file, simplified_mesh)
                
                final_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                if verbose:
                    print(f"最终文件大小: {final_size_mb:.2f} MB")
                
                # 如果仍然太大，我们尝试使用另一种简化方法
                if final_size_mb > target_size_mb:
                    try:
                        # 尝试使用体素下采样方法
                        voxel_size = 0.05  # 起始体素大小
                        for _ in range(3):  # 尝试3种不同的体素大小
                            if verbose:
                                print(f"尝试体素下采样，体素大小: {voxel_size}")
                            
                            downsampled = simplified_mesh.voxel_down_sample(voxel_size)
                            o3d.io.write_triangle_mesh(output_file, downsampled)
                            
                            final_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                            if verbose:
                                print(f"体素下采样后文件大小: {final_size_mb:.2f} MB")
                            
                            if final_size_mb <= target_size_mb:
                                return True
                            
                            # 增加体素大小以进一步减少点数
                            voxel_size *= 2
                    except Exception as e:
                        if verbose:
                            print(f"体素下采样失败: {str(e)}")
                
                return final_size_mb <= target_size_mb
            
            # 更新当前网格为简化后的网格，准备下一次迭代
            current_mesh = simplified_mesh
        else:
            print(f"错误: 无法保存到 {output_file}")
            return False
    
    # 如果达到这里，说明所有迭代都完成了但仍然没有达到目标大小
    return False


def process_models(base_dir, start_idx=0, end_idx=87, target_size_mb=2.0):
    """
    处理指定范围内的所有模型
    
    参数:
        base_dir (str): 模型基础目录
        start_idx (int): 起始模型索引
        end_idx (int): 结束模型索引
        target_size_mb (float): 目标文件大小，单位为MB
    """
    total_models = end_idx - start_idx + 1
    processed = 0
    success = 0
    failures = []
    
    start_time = time.time()
    
    for idx in range(start_idx, end_idx + 1):
        model_dir = os.path.join(base_dir, f"{idx:03d}")
        
        if not os.path.exists(model_dir):
            print(f"警告: 目录 {model_dir} 不存在，跳过")
            continue
        
        input_file = os.path.join(model_dir, "object_visual_original.obj")
        output_file = os.path.join(model_dir, "object_visual.obj")
        
        if not os.path.exists(input_file):
            print(f"警告: 输入文件 {input_file} 不存在，跳过")
            continue
        
        print(f"\n处理模型 {idx:03d} ({processed+1}/{total_models})")
        
        try:
            result = simplify_mesh(input_file, output_file, target_size_mb)
            processed += 1
            
            if result:
                success += 1
                print(f"成功: 模型 {idx:03d} 简化完成")
            else:
                failures.append(idx)
                print(f"失败: 模型 {idx:03d} 简化失败")
        except Exception as e:
            failures.append(idx)
            print(f"错误: 处理模型 {idx:03d} 时发生异常: {str(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印总结
    print("\n" + "="*50)
    print(f"处理完成! 耗时: {elapsed_time:.2f} 秒")
    print(f"总共处理: {processed}/{total_models} 个模型")
    print(f"成功: {success} 个模型")
    print(f"失败: {len(failures)} 个模型")
    
    if failures:
        print("失败的模型索引:")
        for idx in failures:
            print(f"  - {idx:03d}")


def main():
    # 设置基础目录为相对路径 ../models
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    print(f"基础目录: {base_dir}")
    
    # 处理模型，目标大小为2MB
    process_models(base_dir, start_idx=0, end_idx=87, target_size_mb=2.0)


if __name__ == "__main__":
    main()
