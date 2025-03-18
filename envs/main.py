import os
import numpy as np
import pybullet as p
from collector import TactileDataCollector, visualize_pointcloud
from utilities import ModelLoader
import cv2
import open3d as o3d
from tqdm import tqdm
import json
import os

def get_grasp_poses_from_pcd(pcd, temp_dir="/datasets/tmp_files", grasp_env_name="anygrasp", sample_num=4,
                            grasp_script_path="/home/iccd-simulator/code/vt-bullet-env/functions/anygrasp_sdk/grasp_detection/get_grasp_poses.py"):
    """
    从点云中获取抓取位姿，通过调用另一个conda环境下的程序
    Args:
        pcd: Open3D点云对象
        temp_dir: 临时保存点云的目录
        grasp_env_name: 包含抓取算法的conda环境名称
        grasp_script_path: 抓取算法脚本的路径，如果为None则使用默认路径
    Returns:
        List of grasp poses: 每个抓取位姿包含位置和方向信息
    """
    import subprocess
    temp_file_path = os.path.join(temp_dir, "temp_pointcloud.pcd")
    
    # 确保目录存在
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    
    # 保存点云
    o3d.io.write_point_cloud(temp_file_path, pcd)
    print(f"已将点云临时保存到: {temp_file_path}")
    
    output_json_path = os.path.join(temp_dir, "grasp_poses.json")
    
    checkpoint_path = "/home/iccd-simulator/code/vt-bullet-env/functions/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
    # 构建命令
    cmd = f"conda run -n {grasp_env_name} python {grasp_script_path} --checkpoint_path {checkpoint_path} --points_path {temp_file_path} --output {output_json_path} --sample_num {sample_num} --top_down_grasp"
    
    # 执行命令
    try:
        print(f"执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("命令执行成功，输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        return []
    
    # 4. 读取结果
    try:
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                grasp_poses = json.load(f)
            print(f"成功读取{len(grasp_poses)}个抓取位姿")
            
            # 如果需要，可以在这里对抓取位姿进行转换或处理
            # 例如，将抓取位姿从相机坐标系转换到世界坐标系
            # processed_grasp_poses = process_grasp_poses(grasp_poses)
            # return processed_grasp_poses
            
            return grasp_poses
        else:
            print(f"警告: 抓取位姿文件不存在: {output_json_path}")
            return []
    except Exception as e:
        print(f"读取抓取位姿时出错: {str(e)}")
        return []
    finally:
        # 5. 清理临时文件（可选）
        # 如果你想保留这些文件用于调试，可以注释掉下面的代码
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(output_json_path):
            os.remove(output_json_path)

def dataset_collect():
    """采集数据集，包括：
    1. 加载物体
    2. 获取相机点云
    3. 截取物体边界框内的点云
    4. 转换到物体坐标系
    5. 保存处理后的点云
    """
    # 创建数据采集器和保存目录
    save_dir = './datasets'
    os.makedirs(save_dir, exist_ok=True)
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=True)
    # 遍历000-088的所有物体
    for obj_idx in tqdm(range(3, 4)):
        obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
        urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object.urdf"
        
        if not os.path.exists(urdf_path):
            print(f"跳过不存在的物体: {urdf_path}")
            continue
        
        collector.save_dir = os.path.join(save_dir, obj_dir)
        if not os.path.exists(collector.save_dir):
            os.makedirs(collector.save_dir, exist_ok=True)
        obj_id = collector.load_object(urdf_path, position=(0.5, 0, 0.1))
        # 获取相机图像和点云
        rgb, _, _ = collector.get_camera_image()
        cv2.imwrite("rgb.png", rgb)
        pcd = collector.env.camera.get_point_cloud_world()
        cropped_pcd = collector.crop_pointcloud_by_aabb(pcd, obj_id, margin=0.01)
        camera_pcd = collector.transform_pointcloud(cropped_pcd, source_frame='world', target_frame='camera')
        collector.save_pointcloud(pcd=camera_pcd, file_name=f"{obj_dir}_camera_pcd")
        obj_pcd = collector.transform_to_object_frame(cropped_pcd, obj_id)
        collector.save_pointcloud(pcd=obj_pcd, file_name=f"{obj_dir}_object_pcd")
        # 保存抓取位姿
        input("Press Enter to continue...")
        collector.execute_grasp_from_file(f"{collector.save_dir}/grasp_pose.json")
        #grasp_poses = get_grasp_poses_from_pcd(obj_pcd)

        # 移除物体
        collector.remove_all_objects()
    
    del collector
    print("数据集采集完成！")

def move_to_pose_test():
    """测试机器人移动到抓取位姿"""
    collector = TactileDataCollector(save_dir='./datasets', visualize_gui=True)
    collector.move_to_grasp_pose(target_pos=[0.2, 0.3, 0.1], target_orn=[0, 0, 0, 1])
    input("Press Enter to continue...")
    collector.control_gripper(gripper_width=0.04)
    input("Press Enter to continue...")
    collector.control_gripper(gripper_width=0.14)
    collector.remove_all_objects()
    del collector


def camera_test():
    collector = TactileDataCollector(save_dir='./grasp_data')
    urdf_path = "/home/iccd-simulator/code/vt-bullet-env/models/000/object.urdf"
    model_loader = ModelLoader(urdf_file=urdf_path)
    position = (0, 0, 0.1)
    # obj_info = model_loader.load_object(position=position)
    # print(f"物体添加成功，ID: {obj_info}")
    obj = p.loadURDF(urdf_path, basePosition=position)
    #obj_cube = p.loadURDF("cube.urdf", basePosition=position)
    for _ in range(100):
        p.stepSimulation()
    # 采集相机观测
    rgb, depth, seg = collector.get_camera_image()
    print(f"\n相机图像尺寸: RGB {rgb.shape}, Depth {depth.shape}")
    for _ in range(10):
        collector.env.step_simulation()
    os.makedirs(collector.save_dir, exist_ok=True)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(collector.save_dir, 'rgb.png'), rgb_bgr)
    
    # 保存深度图像
    # 将深度值归一化到0-255范围
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    cv2.imwrite(os.path.join(collector.save_dir, 'depth.png'), depth_norm)
    # 转换为点云并可视化
    pcd = depth_to_pointcloud(depth, rgb, collector.env.camera)
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # 可视化点云和坐标系
    o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                    window_name='Point Cloud Viewer',
                                    width=1280,
                                    height=720,
                                    point_show_normal=False)
    o3d.io.write_point_cloud(os.path.join(collector.save_dir, 'point_cloud.pcd'), pcd)


if __name__ == '__main__':
    # camera_test()
    # function_test()
    dataset_collect()
    # grasp_test()
    # move_to_pose_test()                