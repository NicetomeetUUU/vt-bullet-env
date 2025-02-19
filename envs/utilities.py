import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np


class Models:
    def load_objects(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError


class YCBModels(Models):
    def __init__(self, root, selected_names: tuple = ()):
        self.obj_files = glob.glob(root)
        self.selected_names = selected_names

        self.visual_shapes = []
        self.collision_shapes = []

    def load_objects(self):
        shift = [0, 0, 0]
        mesh_scale = [1, 1, 1]

        for filename in self.obj_files:
            # Check selected_names
            if self.selected_names:
                in_selected = False
                for name in self.selected_names:
                    if name in filename:
                        in_selected = True
                if not in_selected:
                    continue
            print('Loading %s' % filename)
            self.collision_shapes.append(
                p.createCollisionShape(shapeType=p.GEOM_MESH,
                                       fileName=filename,
                                       collisionFramePosition=shift,
                                       meshScale=mesh_scale))
            self.visual_shapes.append(
                p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=filename,
                                    visualFramePosition=shift,
                                    meshScale=mesh_scale))

    def __len__(self):
        return len(self.collision_shapes)

    def __getitem__(self, idx):
        return self.visual_shapes[idx], self.collision_shapes[idx]


class Camera:
    def __init__(self, robot_id=None, ee_id=None, size=(1280, 720), near=0.105, far=10.0, fov=69.4,
                 enable_noise=True, enable_distortion=True):
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
        """
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov
        self.aspect = self.width / self.height
        
        # 相机相对于末端执行器的偏移
        self.relative_pos = (0.0, 0.0, 0.10)  # 相对位置：空间略有偏移，防止完全重合
        self.relative_orn = p.getQuaternionFromEuler((0, 0, 0))  # 相对方向：与夹爪保持一致
        
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

        # 初始化时不需要计算变换矩阵，因为相机位置是动态的

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
            # 如果没有绑定到机器人，使用固定视角
            view_matrix = p.computeViewMatrix((1, 1, 1), (0, 0, 0), (0, 0, 1))
        
        # 计算投影矩阵
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        
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
