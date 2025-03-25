import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# print(f"PYOPENGL_PLATFORM设置为: {os.environ.get('PYOPENGL_PLATFORM')}")
import numpy as np
import pybullet as p
from collector import TactileDataCollector, visualize_pointcloud
from utilities import ModelLoader
import cv2
import open3d as o3d
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pdb

def show_collision_mode():
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)  # 显示线框
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)  # 关闭RGB预览
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)  # 关闭深度预览
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # 关闭分割预览

def get_grasp_poses_from_pcd(pcd, temp_dir="/home/iccd-simulator/code/vt-bullet-env/dataset/tmp_files", grasp_env_name="anygrasp", sample_num=4,
                            grasp_script_path="/home/iccd-simulator/code/vt-bullet-env/functions/anygrasp_sdk/grasp_detection/get_grasp_poses.py",
                            grasp_poses_save_path="/home/iccd-simulator/code/vt-bullet-env/dataset/tmp_files/grasp_poses"):
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
    
    checkpoint_path = "/home/iccd-simulator/code/vt-bullet-env/functions/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
    # 构建命令
    python_path = f"/home/iccd-simulator/anaconda3/envs/{grasp_env_name}/bin/python"
    cmd = f"{python_path} {grasp_script_path} --checkpoint_path {checkpoint_path} --points_path {temp_file_path} --output {grasp_poses_save_path} --sample_num {sample_num} --top_down_grasp"
    
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
    grasp_poses_files = f"{grasp_poses_save_path}.json"
    print(f"读取抓取位姿文件: {grasp_poses_files}")
    print(f"抓取位姿保存路径: {grasp_poses_save_path}")
    try:
        if os.path.exists(grasp_poses_files):
            with open(grasp_poses_files, 'r') as f:
                grasp_poses = json.load(f)
            print(f"成功读取{len(grasp_poses)}个抓取位姿")
            return grasp_poses
        else:
            print(f"警告: 抓取位姿文件不存在: {grasp_poses_files}")
            return []
    except Exception as e:
        print(f"读取抓取位姿时出错: {str(e)}")
        return []
    finally:
        # 5. 清理临时文件（可选）
        # 如果你想保留这些文件用于调试，可以注释掉下面的代码
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        # if os.path.exists(grasp_poses_files):
        #     os.remove(grasp_poses_files)
        pass

def save_grasp_poses(collector, obj_id, obj_dir):
    rgb, _, _ = collector.get_camera_image()
    cv2.imwrite("rgb.png", rgb)
    pcd = collector.env.camera.get_point_cloud_world()
    cropped_pcd = collector.crop_pointcloud_by_aabb(pcd, obj_id, margin=0.01)
    camera_pcd = collector.transform_pointcloud(cropped_pcd, source_frame='world', target_frame='camera')

    collector.save_pointcloud(pcd=camera_pcd, file_name=f"{obj_dir}_camera_pcd")
    obj_pcd = collector.transform_to_object_frame(cropped_pcd, obj_id)
    collector.save_pointcloud(pcd=obj_pcd, file_name=f"{obj_dir}_object_pcd")

    # 保存抓取位姿
    grasp_poses = get_grasp_poses_from_pcd(camera_pcd)
    return grasp_poses, cropped_pcd

def dataset_collect():
    # 创建数据采集器和保存目录
    save_dir = '../dataset'
    os.makedirs(save_dir, exist_ok=True)
    # 遍历000-088的所有物体
    for obj_idx in tqdm(range(3, 4)):
        collector = TactileDataCollector(save_dir=save_dir, visualize_gui=True)
        obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
        urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object.urdf"
        collector.move_to_joint_poses([0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636])
        collector.save_dir = os.path.join(save_dir, obj_dir)
        if not os.path.exists(collector.save_dir):
            os.makedirs(collector.save_dir, exist_ok=True)
        #obj_id = collector.load_object(urdf_path, position=(0, -0.5, 0.1))
        object_cfg = {
            "urdf_path": urdf_path,
            "base_position": (0, -0.5, 0.1),
            "global_scaling": 1.0,
            "use_fixed_base": True,
        }
        obj_id = collector.load_object(object_cfg)
        p.changeDynamics(obj_id, -1,
                mass=0.0,  # 设置质量为0使其固定
                )
        pcd = collector.env.camera.get_point_cloud_world()
        cropped_pcd = collector.crop_pointcloud_by_aabb(pcd, obj_id, margin=0.01)
        # grasp_poses = save_grasp_poses(collector, obj_id, obj_dir)
        grasp_poses = json.load(open("/home/iccd-simulator/code/vt-bullet-env/dataset/tmp_files/grasp_poses.json"))
        # show_collision_mode()
        tactile_pointcloud_list, tactile_poses = collector.execute_grasps(grasp_poses)
        show_three_pointclouds(obj_id, collector, cropped_pcd, tactile_pointcloud_list, tactile_poses)

        input("Press Enter to continue...")
        del collector
    print("所有物体数据集采集完成！")

def show_three_pointclouds(obj_id, collector, object_pcd, tactile_pointcloud_list, camera_poses):
    """将物体点云和触觉传感器点云拼接到一起并可视化
    
    Args:
        object_pcd: 物体点云（Open3D点云对象）
        tactile_pointcloud_list: 触觉传感器点云列表（包含两个点云）
        camera_poses: 相机位姿列表
    """
    try:
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="物体与触觉传感器点云集成可视化", width=1280, height=720)
        
        # 添加物体点云（白色）
        if object_pcd is not None:
            # 如果是数组，转换为Open3D点云对象
            if isinstance(object_pcd, np.ndarray):
                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(object_pcd)
            else:
                obj_pcd = object_pcd
            
            # 设置物体点云颜色为白色
            obj_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
            obj_pcd_in_obj_frame = collector.transform_to_object_frame(obj_pcd, obj_id)
            vis.add_geometry(obj_pcd_in_obj_frame)
            vis.add_geometry(obj_pcd)
            print(f"物体点云统计: 总点数={len(np.asarray(obj_pcd.points))}, 最小值={np.min(np.asarray(obj_pcd.points), axis=0)}, 最大值={np.max(np.asarray(obj_pcd.points), axis=0)}")
        
        # 添加触觉传感器点云（亮绿色和黄色）
        for j, points in enumerate(tactile_pointcloud_list):
            # 如果points是列表或为空，跳过
            if isinstance(points, list) and len(points) == 0:
                continue
            
            # 如果points是列表，转换为numpy数组
            if isinstance(points, list):
                points_np = np.array(points)
            else:
                points_np = points
            
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            
            # 设置点云颜色（使用对比度高的颜色）
            if j == 0:
                pcd.paint_uniform_color([0, 1, 0])  # 亮绿色
            else:
                pcd.paint_uniform_color([1, 1, 0])  # 黄色
            
            pcd_in_obj_frame = collector.transform_to_object_frame(pcd, obj_id)
            vis.add_geometry(pcd_in_obj_frame)
            vis.add_geometry(pcd)
            print(f"触觉传感器{j+1}点云统计: 总点数={len(points_np)}, 最小值={np.min(points_np, axis=0)}, 最大值={np.max(points_np, axis=0)}")
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # 添加相机坐标系
        for i, camera_pose in enumerate(camera_poses):
            # 创建坐标系（设置更小的大小以避免挡住点云）
            camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            # 应用相机位姿变换，显示完整的相机位姿（位置和方向）
            camera_frame.transform(camera_pose)
            vis.add_geometry(camera_frame)
            print(f"相机{i+1}位姿:\n{camera_pose}")
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 3.0  # 增大点大小以提高可见度
        opt.show_coordinate_frame = False  # 禁用Open3D内置的坐标轴显示
        
        # 设置初始视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])  # 设置视角朝向
        ctr.set_lookat([0, 0, 0])  # 设置视线中心
        ctr.set_up([0, -1, 0])  # 设置上方向
        
        # 运行可视化器
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"点云集成可视化失败: {e}")

if __name__ == '__main__':
    dataset_collect()              