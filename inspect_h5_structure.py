#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查H5文件结构
"""

import h5py
import sys
import argparse

def print_h5_structure(name, obj):
    """打印H5文件结构"""
    if isinstance(obj, h5py.Dataset):
        print(f"数据集: {name}, 形状: {obj.shape}, 类型: {obj.dtype}")
    else:
        print(f"组: {name}")

def inspect_h5_file(file_path):
    """检查H5文件结构"""
    with h5py.File(file_path, 'r') as f:
        print(f"H5文件: {file_path}")
        print("="*50)
        
        # 遍历文件结构
        f.visititems(print_h5_structure)
        
        # 检查元数据
        if 'metadata' in f:
            print("\n元数据示例:")
            metadata = f['metadata'][0]
            if metadata.dtype.names:
                for name in metadata.dtype.names:
                    print(f"  {name}: {metadata[name]}")
        
        # 检查点云组
        for group_name in ['cropped_object', 'cropped_surface', 'cropped_surface_context', 'tactile_left', 'tactile_right']:
            if group_name in f:
                print(f"\n{group_name}组内容:")
                group = f[group_name]
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        dataset = group[key]
                        print(f"  {key}: 形状 {dataset.shape}, 类型 {dataset.dtype}")

def main():
    parser = argparse.ArgumentParser(description="检查H5文件结构")
    parser.add_argument('--file_path', type=str, required=True, help='H5文件路径')
    
    args = parser.parse_args()
    inspect_h5_file(args.file_path)

if __name__ == '__main__':
    main()
