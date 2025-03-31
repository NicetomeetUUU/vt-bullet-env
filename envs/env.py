import time

import numpy as np
import pybullet as p
import pybullet_data
import pybulletX as px
from utilities import ModelLoader, Camera, CoordinateFrameVisualizer
from collections import namedtuple
from tqdm import tqdm

class VTGraspRefine:

    SIMULATION_STEP_DELAY = 1 / 240.
    def __init__(self, robot, camera=None, vis=False) -> None:
        """
        初始化VTGraspRefine环境
        Args:
            robot: 机器人对象
            camera: 可选，相机对象
            vis: 是否显示可视化界面
        """
        self.robot = robot
        self.camera = camera

        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        
        # 环境定义
        client = px.init(mode=p.GUI if self.vis else p.DIRECT)
        self.planeID = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.5])
        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        
        # 初始化坐标系可视化器
        if self.vis:
            self.frame_visualizer = CoordinateFrameVisualizer(axis_length=0.1)
            self._update_coordinate_frames()

    def step_simulation(self):
        """
        以自定义的步长进行仿真
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)
    
    def _update_coordinate_frames(self):
        """更新所有坐标系的可视化"""
        # 更新末端执行器坐标系
        if not self.vis:
            return
        
        ee_pos, ee_orn = self.robot.get_ee_pos()
        self.frame_visualizer.visualize_frame(ee_pos, ee_orn, 'end_effector')
        
        # 更新相机坐标系
        if self.camera:
            cam_pos, cam_orn = self.camera.get_pose()
            self.frame_visualizer.visualize_frame(cam_pos, cam_orn, 'camera')
        
        # 更新digit传感器坐标系（如果有）
        left_digit_id = self.robot._id
        right_digit_id = self.robot._id
        for joint_name in self.robot.digit_joint_names:
            joint_info = [j for j in self.robot.joints if j[1] == joint_name][0]
            if 'left' in joint_name:
                left_digit_id = joint_info[0]
            elif 'right' in joint_name:
                right_digit_id = joint_info[0]
        
        if left_digit_id != self.robot._id:
            left_state = p.getLinkState(self.robot._id, left_digit_id)
            self.frame_visualizer.visualize_frame(left_state[0], left_state[1], 'left_digit')
        
        if right_digit_id != self.robot._id:
            right_state = p.getLinkState(self.robot._id, right_digit_id)
            self.frame_visualizer.visualize_frame(right_state[0], right_state[1], 'right_digit')

    def calculate_digit_to_ee_transform(self):
        """计算DIGIT传感器到末端执行器(EE)的坐标系转换关系"""
        # 获取末端执行器的位置和方向
        from scipy.spatial.transform import Rotation as R
        ee_pos, ee_orn = self.robot.get_ee_pos()
        ee_rot = R.from_quat(ee_orn).as_matrix()
        
        # 创建末端执行器的变换矩阵
        ee_transform = np.eye(4)
        ee_transform[:3, :3] = ee_rot
        ee_transform[:3, 3] = ee_pos
        
        # 获取DIGIT传感器的链接ID
        left_digit_id = None
        right_digit_id = None
        for joint_name in self.robot.digit_joint_names:
            joint_info = [j for j in self.robot.joints if j[1] == joint_name][0]
            if 'left' in joint_name:
                left_digit_id = joint_info[0]
            elif 'right' in joint_name:
                right_digit_id = joint_info[0]
        
        # 计算左侧DIGIT传感器的变换矩阵
        left_transform = None
        if left_digit_id is not None:
            left_state = p.getLinkState(self.robot._id, left_digit_id)
            left_pos, left_orn = left_state[0], left_state[1]
            left_rot = R.from_quat(left_orn).as_matrix()
            
            left_transform = np.eye(4)
            left_transform[:3, :3] = left_rot
            left_transform[:3, 3] = left_pos
        
        # 计算右侧DIGIT传感器的变换矩阵
        right_transform = None
        if right_digit_id is not None:
            right_state = p.getLinkState(self.robot._id, right_digit_id)
            right_pos, right_orn = right_state[0], right_state[1]
            right_rot = R.from_quat(right_orn).as_matrix()
            
            right_transform = np.eye(4)
            right_transform[:3, :3] = right_rot
            right_transform[:3, 3] = right_pos
        
        # 计算DIGIT传感器到EE的变换矩阵
        left_digit_to_ee = None
        right_digit_to_ee = None
        
        if left_transform is not None:
            # 计算左侧DIGIT传感器到EE的变换矩阵: T_ee_to_world^(-1) * T_digit_to_world
            ee_transform_inv = np.linalg.inv(ee_transform)
            left_digit_to_ee = ee_transform_inv @ left_transform
        
        if right_transform is not None:
            # 计算右侧DIGIT传感器到EE的变换矩阵: T_ee_to_world^(-1) * T_digit_to_world
            ee_transform_inv = np.linalg.inv(ee_transform)
            right_digit_to_ee = ee_transform_inv @ right_transform
        
        return left_digit_to_ee, right_digit_to_ee

    def step_actions(self, actions, steps):
        """
        执行当前动作并进行仿真
        Args:
            actions: 单个动作或动作列表，可以是JointAction或PoseAction
        """
        from actions import ActionWrapper, JointAction, PoseAction, GripperAction
        
        # 创建动作执行器
        action_wrapper = ActionWrapper(self.robot)
        
        # 处理单个动作或动作列表
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
            
        # 执行每个动作
        for action in actions:
            # 执行动作
            action_wrapper.execute_action(action)
            for _ in range(steps):
                self.step_simulation()
        self._update_coordinate_frames()

    def get_ee_pos(self):
        pos, orn = self.robot.get_ee_pos()
        return pos, orn

    def get_camera_pos(self):
        return self.camera.get_pose()

    def get_object_pos(self, object_id):
        pos, orn = p.getBasePositionAndOrientation(object_id)
        return pos, orn

    def get_all_pos(self):
        """
        获取环境中所有重要对象的位姿信息
        Returns:
            dict: 包含以下键值对：
                - 'end_effector': (position, orientation) 末端执行器位姿
                - 'camera': (position, orientation) 相机位姿
                - 'object': (position, orientation) 物体位姿（如果有）
                - 'target': (position, orientation) 目标位姿（如果有）
        """
        poses = {
            'end_effector': self.get_ee_pos(),
            'camera': self.get_camera_pos()
        }
            
        # 如果有目标位置，添加目标位姿
        if hasattr(self, 'target_pos') and hasattr(self, 'target_orn'):
            poses['target'] = (self.target_pos, self.target_orn)
            
        return poses
    
    def reset(self):
        self.robot.reset()

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())
        return obs

    def close(self):
        px.close()