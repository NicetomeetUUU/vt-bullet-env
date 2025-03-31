#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import xml.etree.ElementTree as ET


def create_original_urdf(model_dir):
    """
    为指定模型目录创建使用原始模型的URDF文件
    
    参数:
        model_dir (str): 模型目录路径
    
    返回:
        bool: 是否成功创建URDF文件
    """
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"错误: 目录 {model_dir} 不存在")
        return False
    
    # 检查原始URDF文件是否存在
    urdf_path = os.path.join(model_dir, "object.urdf")
    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件 {urdf_path} 不存在")
        return False
    
    # 检查原始模型文件是否存在
    original_obj_path = os.path.join(model_dir, "object_visual_original.obj")
    if not os.path.exists(original_obj_path):
        print(f"警告: 原始模型文件 {original_obj_path} 不存在，跳过")
        return False
    
    try:
        # 解析URDF文件
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # 查找visual/geometry/mesh元素并修改filename属性
        mesh_elements = root.findall(".//visual/geometry/mesh")
        if not mesh_elements:
            print(f"错误: 在 {urdf_path} 中未找到mesh元素")
            return False
        
        for mesh in mesh_elements:
            # 将mesh的filename属性改为object_visual_original.obj
            mesh.set("filename", "object_visual_original.obj")
        
        # 创建新的URDF文件
        output_path = os.path.join(model_dir, "object_original.urdf")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        
        print(f"成功: 已创建 {output_path}")
        return True
    
    except Exception as e:
        print(f"错误: 处理 {urdf_path} 时发生异常: {str(e)}")
        return False


def process_all_models(base_dir, start_idx=0, end_idx=87):
    """
    处理指定范围内的所有模型目录
    
    参数:
        base_dir (str): 模型基础目录
        start_idx (int): 起始模型索引
        end_idx (int): 结束模型索引
    """
    total_models = end_idx - start_idx + 1
    processed = 0
    success = 0
    failures = []
    
    for idx in range(start_idx, end_idx + 1):
        model_dir = os.path.join(base_dir, f"{idx:03d}")
        
        print(f"\n处理模型 {idx:03d} ({processed+1}/{total_models})")
        
        try:
            result = create_original_urdf(model_dir)
            processed += 1
            
            if result:
                success += 1
            else:
                failures.append(idx)
        except Exception as e:
            failures.append(idx)
            print(f"错误: 处理模型 {idx:03d} 时发生异常: {str(e)}")
    
    # 打印总结
    print("\n" + "="*50)
    print(f"处理完成!")
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
    
    # 处理所有模型
    process_all_models(base_dir, start_idx=0, end_idx=87)


if __name__ == "__main__":
    main()
