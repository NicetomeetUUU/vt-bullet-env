#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
点云数据集构建运行脚本
"""

import os
import argparse
import logging
import sys
from envs.dataset_builder import main as dataset_builder_main

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_builder_run.log')
    ]
)
logger = logging.getLogger('run_dataset_builder')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行点云数据集构建工具")
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='原始数据集目录')
    parser.add_argument('--output_dir', type=str, default='./processed_dataset', help='处理后的输出目录')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析数据集而不处理')
    parser.add_argument('--parallel', action='store_true', help='使用并行处理')
    parser.add_argument('--num_workers', type=int, default=4, help='并行处理的工作进程数')
    parser.add_argument('--strategy', type=str, default='multiscale_hybrid', 
                        choices=['uniform', 'adaptive_density', 'curvature_based', 'multiscale_hybrid'],
                        help='采样策略')
    parser.add_argument('--storage', type=str, default='hierarchical',
                        choices=['single_array', 'scattered', 'hierarchical'],
                        help='存储模式')
    parser.add_argument('--max_object', type=int, default=2500, help='物体点云的最大点数')
    parser.add_argument('--max_surface', type=int, default=25000, help='表面点云的最大点数')
    parser.add_argument('--max_context', type=int, default=10000, help='上下文点云的最大点数')
    parser.add_argument('--include_all_objects', action='store_true', help='包含所有物体，不排除问题物体')
    parser.add_argument('--specific_object', type=str, help='只处理特定物体ID，例如"000"')
    
    args = parser.parse_args()
    
    # 构建传递给dataset_builder的参数列表
    builder_args = []
    
    # 添加基本参数
    builder_args.extend(['--dataset_dir', args.dataset_dir])
    builder_args.extend(['--output_dir', args.output_dir])
    builder_args.extend(['--sampling_strategy', args.strategy])
    builder_args.extend(['--storage_mode', args.storage])
    
    # 添加可选参数
    if args.analyze_only:
        builder_args.append('--analyze_only')
    
    if args.parallel:
        builder_args.append('--parallel')
        builder_args.extend(['--num_workers', str(args.num_workers)])
    
    # 设置最大点数
    builder_args.extend(['--max_object', str(args.max_object)])
    builder_args.extend(['--max_surface', str(args.max_surface)])
    builder_args.extend(['--max_context', str(args.max_context)])
    
    # 如果选择包含所有物体
    if hasattr(args, 'include_all_objects') and args.include_all_objects:
        builder_args.append('--include_all_objects')
        
    # 如果指定了特定物体ID
    if hasattr(args, 'specific_object') and args.specific_object:
        builder_args.extend(['--specific_object', args.specific_object])
    
    # 重置sys.argv以传递参数给dataset_builder_main
    sys.argv = [sys.argv[0]] + builder_args
    
    # 运行数据集构建器
    logger.info(f"运行数据集构建器，参数: {' '.join(builder_args)}")
    dataset_builder_main()

if __name__ == '__main__':
    main()
# python run_dataset_builder.py --dataset_dir /home/iccd-simulator/code/vt-bullet-env/dataset --output_dir /home/iccd-simulator/code/vt-bullet-env/data/processed_dataset --specific_object 000