import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import cv2
import numpy as np
import open3d as o3d
import os
import xml.etree.ElementTree as ET

import time

class ModelLoader:
    def __init__(self, urdf_file: str):
        """
        初始化模型加载器
        Args:
            urdf_file: URDF文件路径
        """
        self.urdf_file = urdf_file
        self.obj_info = None

    # def _get_urdf_offset(self):
    #     """从 URDF 文件中读取 origin 偏移
    #     Args:
    #         urdf_file: URDF文件路径
    #     Returns:
    #         tuple: (x, y, z) 偏移量
    #     """
    #     tree = ET.parse(self.urdf_file)
    #     root = tree.getroot()
        
    #     # 获取第一个 link 中的 visual/origin 的 xyz
    #     origin = root.find('.//visual/origin')
    #     if origin is not None and 'xyz' in origin.attrib:
    #         offset_x, offset_y, offset_z = map(float, origin.get('xyz').split())
    #         return (offset_x, offset_y, offset_z)
    #     return (0, 0, 0)
    
    def load_object(self, position=(0,0,0), orientation=(0, 0, 0, 1), scale=1.0, name=None):
        """添加一个物体到环境中
        Args:
            position: 期望放置物体的位置
            scale: 缩放比例
            name: 物体名称，如果为None则自动生成
        Returns:
            int: 物体ID
        """
        print(f"[开始添加物体] URDF: {self.urdf_file}, 位置: {position}")
        if self.obj_info is not None:
            print(f"[警告] 物体信息不为空")
            self.remove_object()
        # 计算补偿后的位置
        # offset = self._get_urdf_offset()
        # final_position = (
        #     position[0] + offset[0],  # 添加x轴偏移
        #     position[1] - offset[1],
        #     position[2] - offset[2]  # 添加一个高度偏移确保物体不会穿透地面
        # )
        final_position = (
            position[0],  # 添加x轴偏移
            position[1],
            position[2] + 0.2  # 添加一个高度偏移确保物体不会穿透地面
        )
        
        # print(f"[加载物体] 最终位置: {final_position}")
        
        # 加载物体
        obj_id = p.loadURDF(self.urdf_file,
                            basePosition=final_position,
                            baseOrientation=orientation,
                            globalScaling=scale)
        if obj_id < 0:
            raise Exception(f"加载物体失败: {self.urdf_file}")

        # 记录物体信息
        self.obj_info = {
            'id': obj_id,
            'position': position,
            'name': name or f"object_{obj_id}"
        }
        
        print(f"[添加成功] 物体ID: {obj_id}")
        return self.get_object_info()

    def get_object_info(self):
        """获取物体信息
        Returns:
            dict: 物体信息(id, position, name)
        """
        return self.obj_info
    
    def remove_object(self):
        """从环境中移除本物体
        Return:
            bool: 是否删除成功
        """
        print(f"[开始删除物体] {self.obj_info['name']}")
        if self.obj_info is None:
            print("[警告] 物体信息为空")
            return True
        p.removeBody(self.obj_info['id'])
        self.obj_info = None
        return True


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

class Camera:
    def __init__(self, robot_id=None, ee_id=None, size=(1280, 720), near=0.01, far=10.0, fov=69.4,
                 position=None, target=None, up_vector=None):
        """Initialize camera parameters
        Args:
            robot_id: Robot ID if attached to a robot
            ee_id: End-effector ID if attached to end-effector
            size: Image resolution (width, height)
            near: Minimum depth distance (meters)
            far: Maximum depth distance (meters)
            fov: Horizontal field of view (degrees)
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
            print("相机内参矩阵：", self.view_matrix, self.proj_matrix)
            self.world_position = camera_pos
            self.target_position = camera_target
            self.up_vector = camera_up
        
        self.robot_id = robot_id
        self.ee_id = ee_id

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
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=self.aspect,
                nearVal=self.near,
                farVal=self.far
            )
        else:
            # 如果没有绑定到机器人，使用初始化时设置的视角
            view_matrix = self.view_matrix
            proj_matrix = self.proj_matrix
        
        return view_matrix, proj_matrix
        
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

    def get_intrinsics(self):
        """获取相机内参
        Returns:
            tuple: (fx, fy, cx, cy)
                - fx, fy: 焦距
                - cx, cy: 主点坐标
        Note:
            基于FOV和图像尺寸计算内参
            对于1280x720的分辨率，考虑实际的像素比例
        """
        # 水平FOV的一半（弧度）
        fov_h = np.radians(self.fov / 2)
        
        # 计算水平方向的焦距
        fx = self.width / (2 * np.tan(fov_h))
        
        # 对于1280x720，垂直方向的焦距需要根据实际像素比例计算
        # 1280:720 = 16:9
        fy = fx * (self.height / self.width)  # 保持像素的物理大小比例
        
        # 主点坐标（图像中心）
        cx = self.width / 2   # 640
        cy = self.height / 2  # 360
        
        print(f"计算的内参值：\nfx={fx:.2f}\nfy={fy:.2f}\ncx={cx}\ncy={cy}")
        return fx, fy, cx, cy

    def add_noise(self, rgb, depth, apply_noise=True, rgb_noise_std=0.01, depth_noise_std=0.0001, depth_missing_prob=0.001):
        """为图像添加真实世界的噪声
        Args:
            rgb: RGB图像 (H, W, 3)
            depth: 深度图像 (H, W)
            apply_noise: 是否应用噪声
            rgb_noise_std: RGB噪声的标准差（0-255）
            depth_noise_std: 深度噪声的标准差（0-1）
            depth_missing_prob: 深度缺失的概率（0-1）
        Returns:
            tuple: (noisy_rgb, noisy_depth)
        """
        if not apply_noise:
            return rgb, depth
            
        # 添加RGB噪声
        noisy_rgb = rgb.astype(np.float32)
        noise = np.random.normal(0, rgb_noise_std, rgb.shape)
        noisy_rgb += noise
        noisy_rgb = np.clip(noisy_rgb, 0, 255).astype(np.uint8)
        
        # 添加深度噪声
        noisy_depth = depth.copy()
        # 只在有效的深度值上添加噪声
        valid_mask = (depth > 0) & (depth < 1)
        if valid_mask.any():
            noise = np.random.normal(0, depth_noise_std, depth.shape)
            noisy_depth[valid_mask] += noise[valid_mask]
            noisy_depth = np.clip(noisy_depth, 0, 1)
        
        # 模拟深度缺失
        missing_mask = np.random.random(depth.shape) < depth_missing_prob
        noisy_depth[missing_mask & valid_mask] = 0
        
        return noisy_rgb, noisy_depth

    def shot(self, apply_noise=False):
        """获取相机图像
        Args:
            apply_noise: 是否添加真实世界的噪声
        Returns:
            tuple: (rgb, depth, seg)
                - rgb: RGB图像 (H, W, 3)
                - depth: 深度图像 (H, W)，范围[0,1]表示[near,far]之间的深度
                - seg: 分割图像 (H, W)
        """
        # 获取当前相机矩阵
        view_matrix, proj_matrix = self._get_camera_matrices()
        
        # 获取图像
        _, _, rgb, depth, seg = p.getCameraImage(
            self.width, self.height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        if apply_noise:
            rgb, depth = self.add_noise(rgb, depth)
        
        return rgb, depth, seg

    def get_point_cloud_world(self, max_depth=1.0, min_depth=0.01):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        width = self.width
        height = self.height
        view_matrix, proj_matrix = self._get_camera_matrices()
        # get a depth image
        # "infinite" depths will have a value close to 1
        image_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        depth = image_arr[3]
        rgb = image_arr[2][:, :, :3]

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        rgb_flat = rgb.reshape(-1, 3)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(rgb_flat.astype(np.float64)/255.0)

        filtered_points = points[(points[:, 2] >= min_depth) & (points[:, 2] <= max_depth)]
        filtered_colors = rgb_flat[(points[:, 2] >= min_depth) & (points[:, 2] <= max_depth)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64)/255.0)
        return pcd
