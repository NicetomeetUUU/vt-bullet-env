#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os

def remove_ur5_collision(input_urdf, output_urdf):
    """
    移除UR5机器人的所有碰撞标签，保留夹爪的碰撞标签
    
    参数:
        input_urdf (str): 输入URDF文件路径
        output_urdf (str): 输出URDF文件路径
    """
    # 解析URDF文件
    tree = ET.parse(input_urdf)
    root = tree.getroot()
    
    # UR5机器人的链接名称列表（需要移除碰撞的链接）
    ur5_links = [
        "base_link", 
        "shoulder_link", 
        "upper_arm_link", 
        "forearm_link", 
        "wrist_1_link", 
        "wrist_2_link", 
        "wrist_3_link", 
        "ee_link"
    ]
    
    # 遍历所有链接
    for link in root.findall(".//link"):
        link_name = link.get("name")
        
        # 如果是UR5的链接，移除其碰撞标签
        if link_name in ur5_links:
            collision_elements = link.findall("collision")
            for collision in collision_elements:
                link.remove(collision)
            print(f"已移除链接 {link_name} 的碰撞标签")
    
    # 保存修改后的URDF文件
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"已将修改后的URDF保存到 {output_urdf}")

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_urdf = "/home/iccd-simulator/code/vt-bullet-env/envs/urdf/ur5_robotiq_140.urdf"
    output_urdf = "/home/iccd-simulator/code/vt-bullet-env/envs/urdf/ur5_robotiq_140_remove_collision.urdf"
    
    # 移除UR5碰撞标签
    remove_ur5_collision(input_urdf, output_urdf)