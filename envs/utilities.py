import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np
import os

class ModelLoader:
    def __init__(self, urdf_file: str):
        """
        初始化模型加载器
        Args:
            urdf_file: URDF文件路径
        """
        self.urdf_file = urdf_file
        self.body_id = None

    def load_object(self, position=(0,0,0), scale=1.0):
        """
        加载模型到PyBullet世界
        Args:
            position: 模型的初始位置，默认(0,0,0)
            scale: 模型的统一缩放比例，默认1.0
        Returns:
            int: PyBullet中的body ID
        """
        print(f'Loading {self.urdf_file}')
        self.body_id = p.loadURDF(self.urdf_file,
                               basePosition=position,
                               globalScaling=scale)
        return self.body_id
    
    def remove_object(self):
        """从PyBullet世界中移除模型"""
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None

from scipy.spatial.transform import Rotation

class CoordinateFrameVisualizer:
    def __init__(self, axis_length=0.1, line_width=3.0):
        """
        初始化坐标系可视化器
        Args:
            axis_length: 坐标轴长度（米）
            line_width: 线条宽度
        """
        self.axis_length = axis_length
        self.line_width = line_width
        self.frame_ids = {}
    
    def visualize_frame(self, position, orientation, name='frame'):
        """
        可视化一个坐标系
        Args:
            position: [x, y, z] 坐标系原点位置
            orientation: [x, y, z, w] 四元数表示的方向
            name: 坐标系名称，用于更新或删除
        """
        # 如果已存在同名坐标系，先移除
        if name in self.frame_ids:
            self.remove_frame(name)
            
        # 计算坐标轴终点
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        
        x_end = np.array(position) + self.axis_length * rot_matrix[:, 0]
        y_end = np.array(position) + self.axis_length * rot_matrix[:, 1]
        z_end = np.array(position) + self.axis_length * rot_matrix[:, 2]
        
        # 绘制三个坐标轴（RGB对应XYZ）
        frame_ids = [
            p.addUserDebugLine(position, x_end, [1, 0, 0], self.line_width),  # X轴 - 红色
            p.addUserDebugLine(position, y_end, [0, 1, 0], self.line_width),  # Y轴 - 绿色
            p.addUserDebugLine(position, z_end, [0, 0, 1], self.line_width)   # Z轴 - 蓝色
        ]
        
        self.frame_ids[name] = frame_ids
    
    def remove_frame(self, name):
        """
        移除指定的坐标系
        Args:
            name: 要移除的坐标系名称
        """
        if name in self.frame_ids:
            for line_id in self.frame_ids[name]:
                p.removeUserDebugItem(line_id)
            del self.frame_ids[name]
    
    def remove_all_frames(self):
        """移除所有坐标系"""
        for name in list(self.frame_ids.keys()):
            self.remove_frame(name)

class PointCloudProcessor:
    def __init__(self):
        pass

    @staticmethod
    def depth_to_point_cloud(depth_img, intrinsic_matrix, depth_scale=1000.0):
        """将深度图转换为点云
        Args:
            depth_img: 深度图像 (H, W)
            intrinsic_matrix: 相机内参矩阵 (3, 3)
            depth_scale: 深度缩放因子
        Returns:
            points: 点云数据 (N, 3)
        """
        height, width = depth_img.shape
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # 生成像素网格
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - cx) * depth_img / fx
        y = (y - cy) * depth_img / fy
        z = depth_img

        # 将点云整形为(N, 3)的形式
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        # 移除无效点（深度为0或无穷大的点）
        mask = (points[:, 2] > 0) & (points[:, 2] < np.inf)
        points = points[mask]

        return points / depth_scale

    @staticmethod
    def camera_to_world(points, camera_pose):
        """将相机坐标系下的点云转换到世界坐标系
        Args:
            points: 相机坐标系下的点云数据 (N, 3)
            camera_pose: 相机位姿，包含:
                - position: 相机位置 [x, y, z]
                - orientation: 相机方向四元数 [x, y, z, w]
        Returns:
            world_points: 世界坐标系下的点云 (N, 3)
        """
        position = np.array(camera_pose[0])
        orientation = np.array(camera_pose[1])
        
        # 将四元数转换为旋转矩阵
        rotation = Rotation.from_quat(orientation).as_matrix()
        
        # 先旋转再平移
        world_points = points @ rotation.T + position
        return world_points

    @staticmethod
    def crop_points_near_grasp(points, grasp_pose, gripper_width):
        """截取抓取位姿附近的点云（立方体区域）
        Args:
            points: 点云数据 (N, 3)
            grasp_pose: 抓取位姿，包含:
                - position: 位置 [x, y, z]
                - orientation: 方向四元数 [x, y, z, w]
            gripper_width: 夹爪宽度
        Returns:
            cropped_points: 截取后的点云 (M, 3)
        """
        position = np.array(grasp_pose[0])
        orientation = np.array(grasp_pose[1])
        
        # 将点云变换到夹爪坐标系
        rotation = Rotation.from_quat(orientation).as_matrix()
        points_local = (points - position) @ rotation
        
        # 定义立方体边界
        half_width = gripper_width / 2
        bounds = np.array([
            [-half_width, half_width],  # x轴（夹爪宽度方向）
            [-half_width, half_width],  # y轴（夹爪厚度方向）
            [-half_width, half_width]   # z轴（夹爪长度方向）
        ])
        
        # 截取立方体区域内的点
        mask = np.all((
            points_local >= bounds[:, 0].reshape(1, 3)) & 
            (points_local <= bounds[:, 1].reshape(1, 3)
        ), axis=1)
        
        return points[mask]

    @staticmethod
    def transform_point_cloud(points, translation, rotation):
        """对点云进行旋转平移变换
        Args:
            points: 点云数据 (N, 3)
            translation: 平移向量 [x, y, z]
            rotation: 旋转矩阵 (3, 3) 或四元数 [x, y, z, w]
        Returns:
            transformed_points: 变换后的点云 (N, 3)
        """
        if isinstance(rotation, np.ndarray) and rotation.shape == (4,):  # 四元数
            rotation = Rotation.from_quat(rotation).as_matrix()
        
        transformed_points = points @ rotation.T + translation
        return transformed_points

class Camera:
    def __init__(self, robot_id=None, ee_id=None, size=(1280, 720), near=0.105, far=10.0, fov=69.4,
                 enable_noise=False, enable_distortion=False, position=None, target=None, up_vector=None):
        """Initialize camera with RealSense D435 parameters
        Args:
            robot_id: Robot ID if attached to a robot
            ee_id: End-effector ID if attached to end-effector
            size: Image resolution (width, height), D435 default depth resolution
            near: Minimum depth distance (meters)
            far: Maximum depth distance (meters)
            fov: Horizontal field of view (degrees), D435 default is 69.4°
            enable_noise: Enable depth noise simulation
            enable_distortion: Enable lens distortion simulation
            position: 相机在世界坐标系中的位置 (x,y,z)，如果为None则使用默认位置
            target: 相机观察的目标点 (x,y,z)，如果为None则观察原点
            up_vector: 相机向上的方向 (x,y,z)，如果为None则使用z轴正方向
        """
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov
        self.aspect = self.width / self.height
        
        # 如果相机附着在机器人上，保存相对位姿
        if robot_id is not None:
            self.relative_pos = (0.0, 0.0, 0.10)  # 相对位置：空间略有偏移，防止完全重合
            self.relative_orn = p.getQuaternionFromEuler((0, 0, 0))  # 相对方向：与夹爪保持一致

        else:
            # 确保使用传入的position参数
            camera_pos = position if position is not None else (1.0, 0.0, 0.5)
            camera_target = target if target is not None else (0.0, 0.0, 0.0)
            camera_up = up_vector if up_vector is not None else (0.0, 0.0, 1.0)
            
            self.view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=camera_target,
                cameraUpVector=camera_up
            )
            self.proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=self.aspect,
                nearVal=self.near,
                farVal=self.far
            )
            self.world_position = camera_pos
            self.target_position = camera_target
            self.up_vector = camera_up
        
        self.robot_id = robot_id
        self.ee_id = ee_id
        
        # 相机内参
        self.fx = self.width / (2 * np.tan(np.radians(self.fov/2)))
        self.fy = self.height / (2 * np.tan(np.radians(self.fov/self.aspect/2)))
        self.cx = self.width / 2
        self.cy = self.height / 2
        
        # 畔变参数 (D435 typical values)
        self.enable_distortion = enable_distortion
        self.k1 = 0.1  # 径向畔变系数
        self.k2 = 0.2
        self.k3 = 0.0
        self.p1 = 0.01  # 切向畔变系数
        self.p2 = 0.01
        
        # 深度噪声参数
        self.enable_noise = enable_noise
        self.depth_noise_mean = 0.0
        self.baseline_noise = 0.001  # 基线噪声
        self.depth_noise_factor = 0.001  # 与深度相关的噪声因子

    def _get_camera_matrices(self):
        """Get the current view and projection matrices
        Returns:
            tuple: (view_matrix, projection_matrix)
        """
        if self.robot_id is not None and self.ee_id is not None:
            # 获取末端执行器的位置和方向
            ee_state = p.getLinkState(self.robot_id, self.ee_id, computeForwardKinematics=True)
            ee_pos = ee_state[4]  # 世界坐标系中的位置
            ee_orn = ee_state[5]  # 世界坐标系中的方向
            
            # 计算相机的位置和方向
            cam_pos, cam_orn = p.multiplyTransforms(ee_pos, ee_orn, self.relative_pos, self.relative_orn)
            
            # 获取旋转矩阵（将四元数转换为3x3旋转矩阵）
            rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
            
            # 前方向量（矩阵的第一列）
            forward_vec = rot_matrix[:, 0]
            # 上方向（矩阵的第二列）
            up_vec = rot_matrix[:, 1]
            
            # 计算目标点（向前看1米）
            target_pos = np.array(cam_pos) + forward_vec
            
            # 更新视图矩阵
            view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vec)
        else:
            # 如果没有绑定到机器人，使用初始化时设置的视角
            view_matrix = self.view_matrix
            projection_matrix = self.proj_matrix
        
        return view_matrix, projection_matrix
        
    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        
        # 获取当前相机矩阵
        view_matrix, projection_matrix = self._get_camera_matrices()
        _view_matrix = np.array(view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(projection_matrix).reshape((4, 4), order='F')
        _transform = np.linalg.inv(_projection_matrix @ _view_matrix)
        
        position = _transform @ pix_pos
        position /= position[3]

        return position[:3]

    def _apply_distortion(self, x, y):
        """Apply lens distortion to pixel coordinates
        Args:
            x, y: Normalized pixel coordinates (relative to principal point)
        Returns:
            Distorted pixel coordinates
        """
        if not self.enable_distortion:
            return x, y
            
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        # 径向畔变
        radial = (1 + self.k1*r2 + self.k2*r4 + self.k3*r6)
        x_distorted = x * radial
        y_distorted = y * radial
        
        # 切向畔变
        x_distorted += 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
        y_distorted += self.p1*(r2 + 2*y*y) + 2*self.p2*x*y
        
        return x_distorted, y_distorted
    
    def _apply_depth_noise(self, depth):
        """Apply depth-dependent noise to depth image
        Args:
            depth: Depth image
        Returns:
            Noisy depth image
        """
        if not self.enable_noise:
            return depth
            
        # 基线噪声
        noise = np.random.normal(self.depth_noise_mean, self.baseline_noise, depth.shape)
        
        # 深度相关噪声
        depth_noise = np.random.normal(0, self.depth_noise_factor * depth, depth.shape)
        
        noisy_depth = depth + noise + depth_noise
        return np.clip(noisy_depth, self.near, self.far)
    
    def get_pose(self):
        """
        获取相机在世界坐标系中的位姿
        Returns:
            tuple: (position, orientation)
                - position: 相机位置 (x, y, z)
                - orientation: 相机姿态四元数 (x, y, z, w)
        """
        if self.robot_id is not None and self.ee_id is not None:
            # 获取末端执行器的位置和方向
            ee_state = p.getLinkState(self.robot_id, self.ee_id, computeForwardKinematics=True)
            ee_pos = ee_state[4]
            ee_orn = ee_state[5]
            
            # 计算相机的位置和方向
            cam_pos, cam_orn = p.multiplyTransforms(ee_pos, ee_orn, self.relative_pos, self.relative_orn)
            return cam_pos, cam_orn
        else:
            # 如果没有绑定到机器人，返回存储的相机位姿
            # 计算相机方向
            forward = np.array(self.target_position) - np.array(self.world_position)
            forward = forward / np.linalg.norm(forward)
            
            # 使用右手坐标系规则计算相机方向
            up = np.array(self.up_vector)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 构建旋转矩阵
            rot_matrix = np.array([right, up, -forward]).T
            
            # 转换为四元数
            r = Rotation.from_matrix(rot_matrix)
            orientation = r.as_quat()  # 返回 [x, y, z, w] 格式的四元数
            
            return self.world_position, orientation

    def shot(self):
        # 获取当前相机矩阵
        view_matrix, projection_matrix = self._get_camera_matrices()
        
        # 获取图像
        _w, _h, rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            view_matrix,
            projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 应用畔变
        if self.enable_distortion:
            # 创建网格点
            x = np.linspace(0, self.width-1, self.width)
            y = np.linspace(0, self.height-1, self.height)
            x, y = np.meshgrid(x, y)
            
            # 归一化坐标
            x = (x - self.cx) / self.fx
            y = (y - self.cy) / self.fy
            
            # 应用畔变
            x_distorted, y_distorted = self._apply_distortion(x, y)
            
            # 转回像素坐标
            x_pixels = x_distorted * self.fx + self.cx
            y_pixels = y_distorted * self.fy + self.cy
            
            # 重采样图像
            x_pixels = np.clip(x_pixels, 0, self.width-1)
            y_pixels = np.clip(y_pixels, 0, self.height-1)
            
            # 应用到RGB图像
            rgb_distorted = np.zeros_like(rgb)
            for c in range(rgb.shape[2]):
                rgb_distorted[:,:,c] = cv2.remap(rgb[:,:,c], x_pixels.astype(np.float32), 
                                                 y_pixels.astype(np.float32), cv2.INTER_LINEAR)
            rgb = rgb_distorted
        
        # 应用深度噪声
        depth = self._apply_depth_noise(depth)
        
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        # reference: https://stackoverflow.com/a/62247245
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        
        # 获取当前相机矩阵
        view_matrix, projection_matrix = self._get_camera_matrices()
        _view_matrix = np.array(view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(projection_matrix).reshape((4, 4), order='F')
        _transform = np.linalg.inv(_projection_matrix @ _view_matrix)
        
        position = _transform @ pix_pos.T
        position = position.T

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
