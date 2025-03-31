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
from scipy.spatial.transform import Rotation as R
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



euler_orientations = [
    [0.0000, 0.0000, 0.0000],
    [0.2618, 0.8345, 0.4854],
    [0.5236, 1.1071, 0.9709],
    [0.7854, 0.9599, 1.4563],
    [1.0472, 0.5890, 1.9417],
    [1.3090, 0.1592, 2.4271],
    [1.5708, -0.2707, 2.9126],
    [1.8326, -0.6405, 3.3980],
    [2.0944, -0.8877, 3.8834],
    [2.3562, -0.9599, 4.3689],
    [2.6180, -0.8345, 4.8543],
    [2.8798, -0.5345, 5.3397],
    [3.1416, -0.0000, 5.8252],
    [3.4034, 0.5345, 0.5236],
    [3.6652, 0.8345, 1.0090],
    [3.9270, 0.9599, 1.4944],
    [4.1888, 0.8877, 1.9799],
    [4.4506, 0.6405, 2.4653],
    [4.7124, 0.2707, 2.9507],
    [4.9742, -0.1592, 3.4362],
    [5.2360, -0.5890, 3.9216],
    [5.4978, -0.9599, 4.4070],
    [5.7596, -1.1071, 4.8925],
    [6.0214, -0.8345, 5.3779]
]

def save_each_pose_each_object_camera_pcd():
    save_dir = '../dataset'
    os.makedirs(save_dir, exist_ok=True)
    # 遍历000-088的所有物体
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=False)
    # collector.move_to_joint_poses([0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
    #                             -1.5707970583733368, 0.0009377758247187636])
    for obj_idx in tqdm(range(0, 88)):
        obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
        first_save_dir = os.path.join(save_dir, obj_dir)
        os.makedirs(first_save_dir, exist_ok=True)
        for i, euler in enumerate(euler_orientations):
            cur_save_dir = os.path.join(first_save_dir, f"pose_{i:03d}")
            collector.save_dir = cur_save_dir
            os.makedirs(cur_save_dir, exist_ok=True)
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object.urdf"
            base_quat = R.from_euler('xyz', euler).as_quat()
            object_cfg = {
                "urdf_path": urdf_path,
                "base_position": (0, -0.5, 0.1),
                "base_orientation": base_quat,
                "global_scaling": 1.0,
                "use_fixed_base": True,
            }
            obj_id = collector.load_object(object_cfg)
            p.changeDynamics(obj_id, -1,
                    mass=0.0,  # 设置质量为0使其固定
                    )
            pcd = collector.env.camera.get_point_cloud_world()
            cropped_pcd = collector.crop_pointcloud_by_aabb(pcd, obj_id, margin=0.01)
            camera_pcd = collector.transform_pointcloud(cropped_pcd, source_frame='world', target_frame='camera')
            obj_pcd = collector.transform_to_object_frame(cropped_pcd, obj_id)
            rgb, _, _ = collector.get_camera_image()
            collector.save_pointcloud(pcd=camera_pcd, file_name=f"{obj_dir}_camera")
            collector.save_pointcloud(pcd=obj_pcd, file_name=f"{obj_dir}_object")
            rgb_dir = os.path.join(cur_save_dir, f"{obj_dir}_camera")
            # cv2.imwrite(f"{rgb_dir}_rgb.png", rgb)
            collector.remove_object(obj_id)
            
def visualize_saved_pcds():
    save_dir = '../dataset'
    os.makedirs(save_dir, exist_ok=True)
    for obj_idx in tqdm(range(68, 69)):
        obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
        for i, euler in enumerate(euler_orientations):
            cur_save_dir = os.path.join(save_dir, obj_dir, f"pose_{i:03d}")
            pcd_file = os.path.join(cur_save_dir, f"{obj_dir}_camera.pcd")
            visualize_pointcloud(pcd_file)
            

def dataset_collect():
    # 创建数据采集器和保存目录
    save_dir = '../dataset'
    os.makedirs(save_dir, exist_ok=True)
    # 遍历000-088的所有物体
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=True)
    for obj_idx in tqdm(range(3, 4)):
        for i, euler in enumerate(euler_orientations):
            obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object.urdf"
            collector.move_to_joint_poses([0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                                       -1.5707970583733368, 0.0009377758247187636])
            collector.save_dir = os.path.join(save_dir, obj_dir)
            if not os.path.exists(collector.save_dir):
                os.makedirs(collector.save_dir, exist_ok=True)

            base_quat = R.from_euler('xyz', euler).as_quat()
            object_cfg = {
                "urdf_path": urdf_path,
                "base_position": (0, -0.5, 0.1),
                "base_orientation": base_quat,
                "global_scaling": 1.0,
                "use_fixed_base": True,
            }
            obj_id = collector.load_object(object_cfg)
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            print(f"Object pose: {obj_pose}")
            p.changeDynamics(obj_id, -1,
                    mass=0.0,  # 设置质量为0使其固定
                    )
            pcd = collector.env.camera.get_point_cloud_world()
            cropped_pcd = collector.crop_pointcloud_by_aabb(pcd, obj_id, margin=0.01)
            camera_pcd = collector.transform_pointcloud(cropped_pcd, source_frame='world', target_frame='camera')
        # # grasp_poses = save_grasp_poses(collector, obj_id, obj_dir)
        # grasp_poses = json.load(open("/home/iccd-simulator/code/vt-bullet-env/dataset/tmp_files/grasp_poses.json"))
        # # show_collision_mode()
        # tactile_pointcloud_list, tactile_poses = collector.execute_grasps(grasp_poses)
        # show_three_pointclouds(obj_id, collector, cropped_pcd, tactile_pointcloud_list, tactile_poses)
        collector.remove_object(obj_id)
    print("所有物体数据集采集完成！")

def show_three_pointclouds(obj_id, collector, object_pcd, tactile_pointcloud_list, tactile_poses):
    """将物体点云和触觉传感器点云拼接到一起并可视化
    
    Args:
        object_pcd: 物体点云（Open3D点云对象）
        tactile_pointcloud_list: 触觉传感器点云列表（包含两个点云）
        tactile_poses: 触觉传感器位姿列表
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
        vis.add_geometry(coordinate_frame)

        # 添加相机坐标系
        for i, tactile_pose in enumerate(tactile_poses):
            # 创建坐标系（设置更小的大小以避免挡住点云）
            tactile_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            # 应用相机位姿变换，显示完整的相机位姿（位置和方向）
            tactile_frame.transform(tactile_pose)
            vis.add_geometry(tactile_frame)
        
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
    # save_each_pose_each_object_camera_pcd()              
    visualize_saved_pcds()         