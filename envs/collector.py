import os
import numpy as np
import pybullet as p
import pybulletX as px
import tacto
import open3d as o3d
import json
from env import VTGraspRefine
from robot import UR5Robotiq140
from utilities import Camera, ModelLoader
from actions import PoseAction, GripperCommand, GripperAction, JointAction
from scipy.spatial.transform import Rotation as R
import cv2

def get_transform(pos, orn):
    """将位置 (pos) 和旋转 (orn) 转换为齐次变换矩阵"""
    tx, ty, tz = pos
    rx, ry, rz, rw = orn  # orn 是四元数 [rx, ry, rz, rw]
    
    # 转换为旋转矩阵
    R = [
        [1 - 2*ry**2 - 2*rz**2, 2*rx*ry - 2*rz*rw, 2*rx*rz + 2*ry*rw],
        [2*rx*ry + 2*rz*rw, 1 - 2*rx**2 - 2*rz**2, 2*ry*rz - 2*rx*rw],
        [2*rx*rz - 2*ry*rw, 2*ry*rz + 2*rx*rw, 1 - 2*rx**2 - 2*ry**2]
    ]
    
    # 构建齐次矩阵
    T = [[R[0][0], R[0][1], R[0][2], tx],
         [R[1][0], R[1][1], R[1][2], ty],
         [R[2][0], R[2][1], R[2][2], tz],
         [0, 0, 0, 1]]
    return T


class TactileDataCollector:

    def __init__(self, save_dir, visualize_gui = True):
        """初始化触觉数据采集器
        
        Args:
            save_dir (str): 视觉数据保存、抓取位姿读取目录
            visualize_gui (bool): 是否可视化GUI
            object_file (str or list): 物体URDF文件路径(可以是列表类型)
        """
        self.save_dir = save_dir
        self.visualize_gui = visualize_gui
        # 初始化机器人和环境
        robot = UR5Robotiq140((0, 0, 0), (0, 0, 0))  # 绕z轴旋转90度
        self.env = VTGraspRefine(robot, vis=visualize_gui)
        os.makedirs(save_dir, exist_ok=True)

        self.obj_loaders = {}
        
        # 创建固定在世界坐标系中的相机
        camera_height = 0.5  # 相机的高度
        camera_position = [0,   # x坐标（正上方）
                         -0.5,    # y坐标（正上方）
                         camera_height]  # z坐标（高度）
        camera_target = [0, -0.5, 0]  # 相机观察的目标点（世界坐标系原点）
        up_vector = [0, 1, 0]  # 相机向上的方向（y轴正方向）
        
        camera = Camera(
            robot_id=None,  # 不附着在机器人上
            ee_id=None,    # 不附着在末端执行器上
            position=camera_position,
            target=camera_target,
            up_vector=up_vector
        )
        camera_pos, camera_orn = camera.get_pose()
        # 创建一个x,z轴反向的矩阵
        all_axis_flip = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        # 应用到相机旋转矩阵上
        camera_rot_matrix = p.getMatrixFromQuaternion(camera_orn)
        camera_rot_matrix = np.array(camera_rot_matrix).reshape(3, 3)
        pointcloud_base_orn_matrix = camera_rot_matrix @ all_axis_flip
        # 将旋转矩阵转换为四元数
        r = R.from_matrix(pointcloud_base_orn_matrix)
        self.pointcloud_base_orn = r.as_quat()  # 返回[x, y, z, w]格式的四元数
        self.pointcloud_base_pos = camera_pos
        
        # 设置环境的相机
        self.env.camera = camera
        
        for _ in range(50):
            self.env.step_simulation()
        
        # 设置触觉传感器
        self.digits = tacto.Sensor(width=120, height=160, visualize_gui=visualize_gui)
        self.digits.add_camera(robot._id, robot.digit_links)
        self.env.reset()
        for _ in range(50):
            self.env.step_simulation()
        
        # 初始化相机和触觉传感器的可视化
        # 渲染相机图像
        self.env.camera.shot()
        # 渲染触觉传感器数据
        tactile_rgb, tactile_depth, _, _ = self.digits.render()
        self.digits.updateGUI(tactile_rgb, tactile_depth)
        # 执行几步仿真使显示更稳定
        for _ in range(10):
            self.env.step_simulation()
        
        
    def load_object(self, object_cfg):
        '''
        加载物体到世界与digit渲染中
        Args:
            object_cfg (dict): 物体配置字典
        '''
        obj = px.Body(**object_cfg)
        self.digits.add_body(obj)
        obj_id = obj.id
        return obj_id
        
    def collect_tactile_data(self):
        """
        采集当前时刻的触觉数据
        
        Returns:
            tuple: (rgb_list, depth_list, pointcloud_list)
                - rgb_list: RGB图像列表
                - depth_list: 深度图列表
                - pointcloud_list: 相机坐标系下的点云列表
        """
        # 渲染触觉传感器数据
        rgb_list, depth_list, pointcloud_list, camera_poses = self.digits.render()
        # 更新GUI
        self.digits.updateGUI(rgb_list, depth_list)
        if len(rgb_list) > 0:
            cv2.imshow('RGB', rgb_list[0])
            cv2.waitKey(1)
            
        # 执行仿真步骤
        for _ in range(120):
            self.env.step_simulation()
        
        return rgb_list, depth_list, pointcloud_list, camera_poses

    def move_to_joint_poses(self, joint_poses):
        """控制机器人关节移动到目标位姿
        Args:
            joint_poses (list): 关节位姿列表
        """
        action = JointAction(joint_poses, gripper=GripperCommand(width=0.10))
        self.env.step_actions(action, 240)

    def move_to_grasp_pose(self, target_pos, target_orn, gripper_width=0.10):
        """控制机器人末端移动到目标抓取位姿
        
        Args:
            target_pos (tuple): 目标位置 (x, y, z)
            target_orn (tuple): 目标姿态四元数 (x, y, z, w)
            gripper_width (float): 夹爪开度，默认0.04表示完全打开
        """
        
        # 创建夹爪动作
        action = PoseAction(
            position=target_pos,
            orientation=target_orn,
            gripper=GripperCommand(width=gripper_width)
        )
        # 执行动作并添加延时使运动更流畅
        self.env.step_actions(action, 240)

    def control_gripper(self, gripper_width):
        """控制夹爪开合
        
        Args:
            gripper_width (float): 夹爪开合宽度，0.0表示完全关闭，0.14表示完全打开
        """
        # 创建夹爪动作
        action = GripperAction(
            gripper=GripperCommand(width=gripper_width)
        )
        # 执行动作并添加延时使运动更流畅
        self.env.step_actions(action, 120)
        
    def transform_to_world_frame(self,position, rotation_matrix):
        """将抓取位姿从点云坐标系转换到世界坐标系

        Args:
            position (np.ndarray): 抓取位姿的位置
            rotation_matrix (np.ndarray): 抓取位姿的旋转矩阵
        
        Returns:
            tuple: 转换后的抓取位姿位置和旋转矩阵
        """
        pointcloud_base_pos = np.array(self.pointcloud_base_pos)
        pointcloud_base_orn = np.array(self.pointcloud_base_orn)
        pointcloud_base_matrix = R.from_quat(pointcloud_base_orn)# 创建点云坐标系到世界坐标系的变换矩阵
        # 1. 创建旋转部分
        R_pointcloud_to_world = pointcloud_base_matrix.as_matrix()
        # 2. 创建平移部分
        t_pointcloud_to_world = pointcloud_base_pos
        # 3. 将抓取位姿从点云坐标系转换到世界坐标系
        # 3.1 转换位置
        position = R_pointcloud_to_world @ np.array(position) + t_pointcloud_to_world
        # 3.2 转换旋转矩阵
        rotation_matrix = R_pointcloud_to_world @ rotation_matrix
        return position, rotation_matrix

    def move_to_pre_grasp_pose(self, grasp):
        grasp_position = grasp['translation']
        grasp_rotation_matrix = np.array(grasp['rotation'])
        grasp_gripper_width = grasp['width']
        grasp_position, grasp_rotation_matrix = self.transform_to_world_frame(grasp_position, grasp_rotation_matrix)
        r = R.from_matrix(grasp_rotation_matrix)
        grasp_quaternion = r.as_quat()  # 返回[x, y, z, w]格式的四元数
        rot_matrix = np.array(p.getMatrixFromQuaternion(grasp_quaternion)).reshape(3, 3)
        x_axis = rot_matrix[:, 0]  # 旋转矩阵的第一列是x轴方向
        pre_grasp_position = np.array(grasp_position) - 0.06 * x_axis
        self.move_to_grasp_pose(pre_grasp_position, grasp_quaternion, grasp_gripper_width)
        return grasp_position, grasp_quaternion, grasp_gripper_width

    def save_tactile_data(self):
        pass

    def execute_grasps(self, grasp_poses):
        """从文件中读取抓取位姿并执行抓取

        Args:
            grasp_poses (list): 包含抓取位姿的列表
        Returns:
            None
        """
        # 遍历所有抓取位姿
        for i, grasp in enumerate(grasp_poses):
            print(f"\n执行第{i+1}个抓取位姿，得分: {grasp['score']:.4f}, 宽度: {grasp['width']:.4f}")
            grasp_position, grasp_quaternion, grasp_gripper_width = self.move_to_pre_grasp_pose(grasp)
            
            # 打开夹爪准备抓取
            self.move_to_grasp_pose(grasp_position, grasp_quaternion, 0.10)

            # 夹取
            self.control_gripper(gripper_width=0.01)
            rgb_list, depth_list, pointcloud_list, camera_poses = self.collect_tactile_data()

            self.save_tactile_data()
            # 重置位置
            # self.reset()
        return pointcloud_list, camera_poses
    
    def reset(self):
        """重置环境"""
        self.env.reset()
    
    def get_poses(self):
        """获取当前环境内所需位姿"""
        return self.env.get_poses()
    
    def get_camera_image(self):
        """获取相机图像
        
        Returns:
            tuple: (rgb, depth, seg)
                - rgb: RGB图像数据
                - depth: 深度图数据
                - seg: 分割图数据
        """
        return self.env.camera.shot(True)
        
    def save_pointcloud(self, pcd, file_name, format='pcd'):
        """保存物体点云
        Args:
            pcd: 可选的预先获取的点云对象，如果为None则自动获取
            format: 保存的点云格式，默认为'pcd'，支持'pcd'和'ply'
            file_name: 文件名
        Returns:
            o3d.geometry.PointCloud: 物体点云对象
        """
        save_path = os.path.join(self.save_dir, f"{file_name}.{format}")
        o3d.io.write_point_cloud(save_path, pcd)

    def transform_pointcloud(self, pcd, source_frame='camera', target_frame='world'):
        """转换点云坐标系
        Args:
            pcd: Open3D点云对象
            source_frame: 源坐标系，可选'camera'或'world'
            target_frame: 目标坐标系，可选'camera'或'world'
            
        Returns:
            o3d.geometry.PointCloud: 转换后的点云对象
        """
        # 如果源坐标系和目标坐标系相同，直接返回原点云
        if source_frame == target_frame:
            return pcd
            
        # 获取相机到世界坐标系的变换矩阵
        cam_pos, cam_orn = self.env.get_camera_pos()
        cam_rot_matrix = p.getMatrixFromQuaternion(cam_orn)
        cam_rot_matrix = np.array(cam_rot_matrix).reshape(3, 3)
        cam_pos = np.array(cam_pos)
        
        # 获取点云的点
        points = np.asarray(pcd.points)
        transformed_points = np.zeros_like(points)
        
        # 从相机坐标系到世界坐标系
        if source_frame == 'camera' and target_frame == 'world':
            for i in range(len(points)):
                # 相机坐标系中，z轴指向前方，x轴指向右方，y轴指向下方
                # 需要调整坐标轴使其与世界坐标系一致
                cam_point = np.array([points[i][0], -points[i][1], -points[i][2]])
                world_point = cam_rot_matrix.dot(cam_point) + cam_pos
                transformed_points[i] = world_point
        # 从世界坐标系到相机坐标系
        elif source_frame == 'world' and target_frame == 'camera':
            for i in range(len(points)):
                # 先移动到相机原点
                centered_point = points[i] - cam_pos
                # 应用逆旋转（转置矩阵就是逆矩阵，因为旋转矩阵是正交矩阵）
                cam_point = cam_rot_matrix.T.dot(centered_point)
                # 调整坐标轴使其与相机坐标系一致
                transformed_points[i] = np.array([cam_point[0], -cam_point[1], -cam_point[2]])
        else:
            raise ValueError(f"不支持的坐标系转换: {source_frame} -> {target_frame}")
        
        # 创建新的点云对象
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        if pcd.has_colors():
            transformed_pcd.colors = pcd.colors
        
        return transformed_pcd
        
    def crop_pointcloud_by_aabb(self, pcd, obj_id, margin=0.02):
        """根据物体的AABB边界框截取点云
        Args:
            pcd: Open3D点云对象（世界坐标系下）
            obj_id: 物体ID
            margin: 边界框裂量，单位为米
            
        Returns:
            o3d.geometry.PointCloud: 截取后的点云对象
        """
        # 获取物体的AABB边界框
        aabb_min, aabb_max = p.getAABB(obj_id)
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)
        
        # 截取物体AABB边界框内的点
        # 添加一些裂量以确保捕获到物体的所有点
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=aabb_min - margin,
            max_bound=aabb_max + margin
        )
        cropped_pcd = pcd.crop(bbox)
        return cropped_pcd
        
    def transform_to_object_frame(self, pcd, obj_id):
        """将点云转换到物体坐标系
        Args:
            pcd: Open3D点云对象（世界坐标系下）
            obj_id: 物体ID
            
        Returns:
            o3d.geometry.PointCloud: 物体坐标系下的点云对象
        """
        # 获取物体的位姿和方向
        obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
        obj_pos = np.array(obj_pos)
        obj_rot_matrix = np.array(p.getMatrixFromQuaternion(obj_orn)).reshape(3, 3)
        
        # 获取点云的点
        points = np.asarray(pcd.points)
        obj_frame_points = np.zeros_like(points)
        
        # 将点从世界坐标系转换到物体坐标系
        for i in range(len(points)):
            # 先移动到物体原点
            centered_point = points[i] - obj_pos
            # 应用逆旋转（转置矩阵就是逆矩阵，因为旋转矩阵是正交矩阵）
            obj_frame_point = obj_rot_matrix.T.dot(centered_point)
            obj_frame_points[i] = obj_frame_point
        
        # 创建物体坐标系下的点云
        obj_frame_pcd = o3d.geometry.PointCloud()
        obj_frame_pcd.points = o3d.utility.Vector3dVector(obj_frame_points)
        if pcd.has_colors():
            obj_frame_pcd.colors = pcd.colors
        
        print(f"已将点云转换到物体坐标系，物体位置: {obj_pos}, 点数: {len(obj_frame_points)}")
        return obj_frame_pcd
    
    def __del__(self):
        """析构函数，用于清理资源"""
        try:
            # 删除所有物体
            
            # 关闭GUI窗口
            if hasattr(self, 'digits') and hasattr(self.digits, '_gui'):
                self.digits._gui.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def visualize_pointcloud(pcd_or_path, window_name="点云查看器", background_color=[0.1, 0.1, 0.1], show_coordinate_frame=False, coordinate_frame_size=0.1, show_origin=True):
    """可视化Open3D点云对象或PLY文件
    Args:
        pcd_or_path: Open3D点云对象或PLY文件路径
        window_name: 窗口名称
        background_color: 背景颜色，默认为深灰色
        show_coordinate_frame: 是否显示坐标轴
        coordinate_frame_size: 坐标轴大小
        show_origin: 是否显示原点坐标轴，用于检查点云坐标系变换是否正确
    Returns:
        None
    """
    # 检查输入类型并加载点云
    if isinstance(pcd_or_path, str):
        # 输入是文件路径
        if not os.path.exists(pcd_or_path):
            print(f"错误：文件 {pcd_or_path} 不存在")
            return
        try:
            pcd = o3d.io.read_point_cloud(pcd_or_path)
            print(f"从文件加载点云: {pcd_or_path}")
        except Exception as e:
            print(f"加载PLY文件时出错：{e}")
            return
    elif isinstance(pcd_or_path, o3d.geometry.PointCloud):
        # 输入是Open3D点云对象
        pcd = pcd_or_path
    else:
        print(f"错误：输入类型不支持，应为字符串或Open3D点云对象，实际为 {type(pcd_or_path)}")
        return
    
    # 检查点云对象是否有效
    if pcd is None or len(np.asarray(pcd.points)) == 0:
        print("错误：点云对象为空或无点")
        return
    
    # 显示点云基本信息
    print(f"点云信息：")
    print(f"- 点数量：{len(np.asarray(pcd.points))}")
    print(f"- 是否有颜色：{pcd.has_colors()}")
    print(f"- 是否有法线：{pcd.has_normals()}")
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 添加原点坐标轴（始终显示）
    if show_origin:
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coordinate_frame_size, origin=[0, 0, 0])
        vis.add_geometry(origin_frame)
        print("原点坐标轴已添加，用于参照点云坐标系")
    
    # 添加点云中心坐标轴
    if show_coordinate_frame:
        # 获取点云的边界框信息
        points = np.asarray(pcd.points)
        if len(points) > 0:
            # 计算点云的中心点
            center = np.mean(points, axis=0)
            # 创建坐标轴并放置在点云中心
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coordinate_frame_size, origin=center)
            vis.add_geometry(coordinate_frame)
            print(f"点云中心坐标轴已添加: {center}")
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray(background_color)  # 背景颜色
    opt.point_size = 2.0  # 点大小
    # 禁用Open3D内置的坐标轴显示，因为我们已经手动添加了坐标轴
    opt.show_coordinate_frame = False
    
    # 设置初始视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()