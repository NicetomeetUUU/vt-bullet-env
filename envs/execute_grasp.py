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
            
def visualize_saved_pcds():
    save_dir = '../dataset'
    os.makedirs(save_dir, exist_ok=True)
    for obj_idx in tqdm(range(3, 4)):
        obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
        for i, euler in enumerate(euler_orientations):
            cur_save_dir = os.path.join(save_dir, obj_dir, f"pose_{i:03d}")
            pcd_file = os.path.join(cur_save_dir, f"{obj_dir}_camera.pcd")
            visualize_pointcloud(pcd_file)

def grasp_and_vis_pcds():
    # 创建数据采集器和保存目录
    save_dir = '../dataset'
    if not os.path.exists(save_dir):
        print(f"不存在目录: {save_dir}")
        return
    # 遍历3-88的所有物体
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=True)
    # show_collision_mode()
    for obj_idx in tqdm(range(41, 42)):
        for i, euler in enumerate(euler_orientations): 
            if i != 0:
                continue
            obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_original.urdf"
            surface_pcd_file = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_surface_points.npy"
            pcd_and_grasp_poses_dir = os.path.join(save_dir, obj_dir, f"pose_{i:03d}")
            if not os.path.exists(pcd_and_grasp_poses_dir):
                print(f"不存在目录: {pcd_and_grasp_poses_dir}")
                return
            # collector.save_dir = pcd_and_grasp_poses_dir

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
            object_pcd_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_object.pcd")
            surface_pcd = np.load(surface_pcd_file)
            object_pcd = o3d.io.read_point_cloud(object_pcd_file)

            grasp_poses_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_grasp_poses.json")
            grasp_poses = json.load(open(grasp_poses_file))

            for j, grasp_pose in enumerate(grasp_poses):
                tactile_pointclouds, rgbs, depths, tactile_poses, world_grasp_position, world_grasp_quaternion = collector.execute_grasp(grasp_pose)
                obj_grasp_position, obj_grasp_quaternion = collector.transform_grasp_to_object_frame(world_grasp_position, world_grasp_quaternion, obj_id)
                grasp_origin_data = grasp_pose['origin_data']
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="物体与触觉传感器点云集成可视化", width=1280, height=720)
                vis.add_geometry(object_pcd)
                for k, tactile_pointcloud in enumerate(tactile_pointclouds):
                    points = np.array(tactile_pointcloud)
                    if len(points) <= 100:
                        print(f"[警告] 触觉传感器{k+1}点云点数太少: {len(points)}")
                        continue
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    if k == 0:
                        pcd.paint_uniform_color([0, 1, 0])  # 亮绿色
                    else:
                        pcd.paint_uniform_color([1, 1, 0])  # 黄色
                    pcd = collector.transform_to_object_frame(pcd, obj_id)
                    vis.add_geometry(pcd)
                obj_grasp_matrix = np.eye(4)
                obj_grasp_matrix[:3, :3] = R.from_quat(obj_grasp_quaternion).as_matrix()
                obj_grasp_matrix[:3, 3] = obj_grasp_position
                obj_grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
                obj_grasp_frame.transform(obj_grasp_matrix)
                vis.add_geometry(obj_grasp_frame)
                vis.run()
                vis.destroy_window()
                collector.reset()
            collector.remove_object(obj_id)

def crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, pcd, crop_width=0.10, crop_height=0.06, crop_depth=0.08):
    """按照抓取位姿截取点云
    
    参数:
        obj_grasp_position: 物体坐标系下的抓取位置 [x, y, z]
        obj_grasp_quaternion: 物体坐标系下的抓取姿态四元数 [x, y, z, w]
        pcd: 需要截取的点云，可以是numpy数组或Open3D点云对象
        crop_width: 截取的宽度（y轴方向），默认10cm
        crop_height: 截取的高度（z轴方向），默认6cm
        crop_depth: 截取的深度（x轴方向），默认12cm（前后各6cm）
    
    返回:
        cropped_pcd: 截取后的点云（与输入类型相同）
    """
    # 将四元数转换为旋转矩阵
    r = R.from_quat(obj_grasp_quaternion)
    rotation_matrix = r.as_matrix()
    
    # 获取抓取坐标系的三个轴
    x_axis = rotation_matrix[:, 0]  # 抓取方向
    y_axis = rotation_matrix[:, 1]  # 夹爪开合方向
    z_axis = rotation_matrix[:, 2]  # 垂直于抓取平面
    
    # 判断输入点云类型并转换为numpy数组
    is_o3d = isinstance(pcd, o3d.geometry.PointCloud)
    if is_o3d:
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
    elif isinstance(pcd, np.ndarray):
        points = pcd
    else:
        raise TypeError("点云类型必须是Open3D点云对象或numpy数组")
    
    # 如果点云为空，直接返回
    if len(points) == 0:
        print("警告: 输入点云为空")
        return pcd
    
    # 将点云转换到以抓取点为原点的坐标系
    centered_points = points - obj_grasp_position
    
    # 计算点在三个轴上的投影
    x_proj = np.abs(np.dot(centered_points, x_axis))
    y_proj = np.abs(np.dot(centered_points, y_axis))
    z_proj = np.abs(np.dot(centered_points, z_axis))
    
    # 筛选在长方体内的点
    half_depth = crop_depth / 2  # x轴方向前后各8cm
    half_width = crop_width / 2  # y轴方向
    half_height = crop_height / 2  # z轴方向
    
    mask = (x_proj <= half_depth) & (y_proj <= half_width) & (z_proj <= half_height)
    
    # 注意：我们直接从原始点云中筛选点，这样保持在原坐标系中
    # 不需要将centered_points转换回原坐标系，因为我们只是用它来计算投影
    cropped_points = points[mask]
    
    # 如果截取后的点云为空，给出警告
    if len(cropped_points) == 0:
        print("警告: 截取后的点云为空，请检查截取参数或抓取位姿")
        # 返回一个包含抓取点的点云，方便调试
        cropped_points = np.array([obj_grasp_position])
    
    # 根据输入类型返回相应格式的点云
    if is_o3d:
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
        # 如果原点云有颜色，也截取颜色
        if pcd.has_colors():
            cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        return cropped_pcd
    else:
        return cropped_points
    
def downsample_pcd(pcd, voxel_size=0.0002):
    if isinstance(pcd, o3d.geometry.PointCloud):
        return pcd.voxel_down_sample(voxel_size)
    elif isinstance(pcd, np.ndarray):
        return voxel_downsample(pcd, voxel_size)
    else:
        raise TypeError("点云类型必须是Open3D点云对象或numpy数组")

def crop_pointclouds():
    # 创建数据采集器和保存目录
    save_dir = '../dataset'
    if not os.path.exists(save_dir):
        print(f"不存在目录: {save_dir}")
        return
    # 遍历3-88的所有物体
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=False)
    # show_collision_mode()
    for obj_idx in tqdm(range(0, 88)):
        for i, euler in enumerate(euler_orientations): 
            obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_original.urdf"
            surface_pcd_file = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_surface_points.pcd"
            pcd_and_grasp_poses_dir = os.path.join(save_dir, obj_dir, f"pose_{i:03d}")
            if not os.path.exists(pcd_and_grasp_poses_dir):
                print(f"不存在目录: {pcd_and_grasp_poses_dir}")
                return
            # collector.save_dir = pcd_and_grasp_poses_dir

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
            object_pcd_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_object.pcd")
            surface_pcd = o3d.io.read_point_cloud(surface_pcd_file)
            object_pcd = o3d.io.read_point_cloud(object_pcd_file)

            grasp_poses_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_grasp_poses.json")
            grasp_poses = json.load(open(grasp_poses_file))

            for j, grasp_pose in enumerate(grasp_poses):
                world_grasp_position, world_grasp_quaternion, _ = collector.transform_grasp_to_world_frame(grasp_pose)
                tactile_and_cropped_pcd_save_dir = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_grasp_{j:03d}")
                if not os.path.exists(tactile_and_cropped_pcd_save_dir):
                    continue
                # os.makedirs(tactile_and_cropped_pcd_save_dir, exist_ok=True)
                tactile_grasp_pose_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_grasp_pose.json")
                obj_grasp_position, obj_grasp_quaternion = collector.transform_grasp_to_object_frame(world_grasp_position, world_grasp_quaternion, obj_id)
                cropped_surface_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, surface_pcd)
                cropped_surface_pcd = downsample_pcd(cropped_surface_pcd)
                # visualize_pointcloud(cropped_surface_pcd)
                cropped_surface_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_surface.pcd")
                cropped_surface_context_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, surface_pcd, crop_width=0.2, crop_height=0.12, crop_depth=0.16)
                cropped_surface_context_pcd = downsample_pcd(cropped_surface_context_pcd, voxel_size=0.001)
                # visualize_pointcloud(cropped_surface_context_pcd)
                cropped_surface_context_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_surface_context.pcd")
                o3d.io.write_point_cloud(cropped_surface_context_pcd_file, cropped_surface_context_pcd)
                o3d.io.write_point_cloud(cropped_surface_pcd_file, cropped_surface_pcd)
                cropped_object_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, object_pcd)
                # visualize_pointcloud(cropped_object_pcd)
                cropped_object_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_object.pcd")
                o3d.io.write_point_cloud(cropped_object_pcd_file, cropped_object_pcd)
                collector.reset()
            collector.remove_object(obj_id)
                
    print("所有数据集截取完成！")

def grasp_tactile_and_cropped_pcd_data_collect():
    # 创建数据采集器和保存目录
    save_dir = '../dataset'
    if not os.path.exists(save_dir):
        print(f"不存在目录: {save_dir}")
        return
    # 遍历3-88的所有物体
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=True)
    # show_collision_mode()
    for obj_idx in tqdm(range(0, 1)):
        for i, euler in enumerate(euler_orientations): 
            obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_original.urdf"
            surface_pcd_file = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object_surface_points.pcd"
            pcd_and_grasp_poses_dir = os.path.join(save_dir, obj_dir, f"pose_{i:03d}")
            if not os.path.exists(pcd_and_grasp_poses_dir):
                print(f"不存在目录: {pcd_and_grasp_poses_dir}")
                return
            # collector.save_dir = pcd_and_grasp_poses_dir

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
            object_pcd_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_object.pcd")
            surface_pcd = o3d.io.read_point_cloud(surface_pcd_file)
            object_pcd = o3d.io.read_point_cloud(object_pcd_file)

            grasp_poses_file = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_grasp_poses.json")
            grasp_poses = json.load(open(grasp_poses_file))

            for j, grasp_pose in enumerate(grasp_poses):
                tactile_pointclouds, rgbs, depths, tactile_poses, world_grasp_position, world_grasp_quaternion = collector.execute_grasp(grasp_pose)
                # pdb.set_trace()
                pcd_left = None
                pcd_right = None
                rgb_left = None
                rgb_right = None
                for k, points in enumerate(tactile_pointclouds):
                    # 如果points是列表或为空，跳过
                    if points is None or (isinstance(points, list) and len(points) == 0 or len(points) <= 100):
                        # 保证左右都有触觉数据要么没必要做了
                        break
                    # 如果points是列表，转换为numpy数组
                    if isinstance(points, list):
                        points_np = np.array(points)
                    else:
                        points_np = points
                    
                    # 创建Open3D点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_np)
                    pcd = collector.transform_to_object_frame(pcd, obj_id)
                    
                    # 保存点云
                    if k == 0:
                        pcd_left = pcd
                        rgb_left = rgbs[k]
                    else:
                        pcd_right = pcd
                        rgb_right = rgbs[k]
                if pcd_left is None or pcd_right is None:
                    continue
                tactile_and_cropped_pcd_save_dir = os.path.join(pcd_and_grasp_poses_dir, f"{obj_dir}_grasp_{j:03d}")
                os.makedirs(tactile_and_cropped_pcd_save_dir, exist_ok=True)
                pdb.set_trace()
                # 保存当前抓取位姿
                tactile_grasp_pose_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_grasp_pose.json")
                with open(tactile_grasp_pose_file, 'w') as f:
                    json.dump(grasp_pose, f)
                # 保存触觉点云
                tactile_left_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_left.pcd")
                tactile_right_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_right.pcd")
                o3d.io.write_point_cloud(tactile_left_pcd_file, pcd_left)
                o3d.io.write_point_cloud(tactile_right_pcd_file, pcd_right)
                # 保存触觉位姿
                tactile_left_pose_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_left_pose.json")
                tactile_right_pose_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_right_pose.json")
                with open(tactile_left_pose_file, 'w') as f:
                    json.dump(tactile_poses[0].tolist(), f)
                with open(tactile_right_pose_file, 'w') as f:
                    json.dump(tactile_poses[1].tolist(), f)
                # 保存触觉RGB
                tactile_left_rgb_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_left_rgb.png")
                tactile_right_rgb_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_tactile_right_rgb.png")
                cv2.imwrite(tactile_left_rgb_file, rgb_left)
                cv2.imwrite(tactile_right_rgb_file, rgb_right)
                # 保存表面点云
                obj_grasp_position, obj_grasp_quaternion = collector.transform_grasp_to_object_frame(world_grasp_position, world_grasp_quaternion, obj_id)
                cropped_surface_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, surface_pcd)
                cropped_surface_pcd = downsample_pcd(cropped_surface_pcd)
                # visualize_pointcloud(cropped_surface_pcd)
                cropped_surface_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_surface.pcd")
                cropped_surface_context_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, surface_pcd, crop_width=0.2, crop_height=0.12, crop_depth=0.20)
                cropped_surface_context_pcd = downsample_pcd(cropped_surface_context_pcd, voxel_size=0.001)
                # visualize_pointcloud(cropped_surface_context_pcd)
                cropped_surface_context_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_surface_context.pcd")
                o3d.io.write_point_cloud(cropped_surface_context_pcd_file, cropped_surface_context_pcd)
                o3d.io.write_point_cloud(cropped_surface_pcd_file, cropped_surface_pcd)
                cropped_object_pcd = crop_pcd_by_grasp_pose(obj_grasp_position, obj_grasp_quaternion, object_pcd)
                # visualize_pointcloud(cropped_object_pcd)
                cropped_object_pcd_file = os.path.join(tactile_and_cropped_pcd_save_dir, f"{obj_dir}_cropped_object.pcd")
                o3d.io.write_point_cloud(cropped_object_pcd_file, cropped_object_pcd)
                collector.reset()
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
    # grasp_tactile_and_cropped_pcd_data_collect()              
    # visualize_saved_pcds()
    # test_remove_readd_object()       
    crop_pointclouds()  