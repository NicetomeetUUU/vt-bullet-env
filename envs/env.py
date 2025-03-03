import time

import numpy as np
import pybullet as p
import pybullet_data

from utilities import ModelLoader, Camera, CoordinateFrameVisualizer
from collections import namedtuple
from tqdm import tqdm

class VTGraspRefine:

    SIMULATION_STEP_DELAY = 1 / 240.
    def __init__(self, robot, models: ModelLoader = None, camera=None, vis=False) -> None:
        """
        初始化VTGraspRefine环境
        Args:
            robot: 机器人对象
            models: 可选，模型加载器
            camera: 可选，相机对象
            vis: 是否显示可视化界面
        """
        self.robot = robot
        self.camera = camera
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        
        # 环境定义
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.planeID = p.loadURDF("plane.urdf")
        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        
        # 初始化坐标系可视化器
        if self.vis:
            self.frame_visualizer = CoordinateFrameVisualizer(axis_length=0.1)
            self._update_coordinate_frames()
            
        # 物体管理
        self.objects = {}
        self.object_list = []  # 兼容旧的接口
        self._object_counter = 0  # 用于生成唯一的物体名称

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
            
            # 执行物理仪真和渲染
            for _ in range(steps):  # 执行1秒的仿真步骤
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
        
        # 如果有物体，添加物体位姿
        if hasattr(self, 'object_id'):
            poses['object'] = self.get_object_pos(self.object_id)
            
        # 如果有目标位置，添加目标位姿
        if hasattr(self, 'target_pos') and hasattr(self, 'target_orn'):
            poses['target'] = (self.target_pos, self.target_orn)
            
        return poses

    def _get_urdf_offset(self, urdf_file: str):
        """从 URDF 文件中读取 origin 偏移
        Args:
            urdf_file: URDF文件路径
        Returns:
            tuple: (x, y, z) 偏移量
        """
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        # 获取第一个 link 中的 visual/origin 的 xyz
        origin = root.find('.//visual/origin')
        if origin is not None and 'xyz' in origin.attrib:
            offset_x, offset_y, offset_z = map(float, origin.get('xyz').split())
            return (offset_x, offset_y, offset_z)
        return (0, 0, 0)
    
    def add_object(self, urdf_file: str, position=(0,0,0), scale=1.0, name=None):
        """添加一个物体到环境中
        Args:
            urdf_file: URDF文件路径
            position: 期望放置物体的位置
            scale: 缩放比例
            name: 物体名称，如果为None则自动生成
        Returns:
            int: 物体ID
        """
        print(f"[开始添加物体] URDF: {urdf_file}, 位置: {position}")
        print(f"当前物体列表: {self.object_list}")
        
        # 如果没有指定名称，生成一个唯一的名称
        if name is None:
            while True:
                name = f"object_{self._object_counter}"
                self._object_counter += 1
                if name not in self.objects:
                    break
        
        # 检查物体是否已存在
        if name in self.objects:
            print(f"[警告] 物体 {name} 已存在，先删除旧的")
            self.remove_object(name)
        
        try:
            # 计算补偿后的位置
            offset = self._get_urdf_offset(urdf_file)
            
            # 根据当前物体数量添加偏移，避免重叠
            x_offset = len(self.object_list) * 0.2  # 每个物体间间20cm
            
            final_position = (
                position[0] + x_offset - offset[0],  # 添加x轴偏移
                position[1] - offset[1],
                position[2] - offset[2] + 0.3  # 添加一个高度偏移确保物体不会穿透地面
            )
            
            print(f"[加载物体] 最终位置: {final_position}")
            
            # 加载物体
            obj_id = p.loadURDF(urdf_file,
                              basePosition=final_position,
                              globalScaling=scale)
            
            if obj_id < 0:
                raise Exception(f"加载物体失败: {urdf_file}")
            
            # 记录物体信息
            obj_info = {
                'id': obj_id,
                'urdf': urdf_file,
                'position': position,
                'name': name
            }
            self.objects[name] = obj_info
            self.object_list.append(obj_info)  # 兼容旧的接口
            
            print(f"[添加成功] 物体ID: {obj_id}")
            
            # 执行几步仿真以确保物体稳定
            for _ in range(24):
                self.step_simulation()
                
            return obj_id
            
        except Exception as e:
            print(f"[错误] 添加物体失败: {e}")
            return None
        
        # 执行几步仿真以确保物体稳定
        for _ in range(24):
            self.step_simulation()
        
        return obj_id
    
    def remove_object(self, name_or_id):
        """从环境中移除一个物体
        Args:
            name_or_id: 物体名称或ID
        """
        print(f"[开始删除物体] {name_or_id}")
        print(f"当前物体列表: {self.object_list}")
        
        try:
            # 如果传入的是ID，查找对应的名称
            if isinstance(name_or_id, int):
                for obj_info in self.object_list:
                    if obj_info['id'] == name_or_id:
                        name_or_id = obj_info['name']
                        print(f"[找到物体] ID {name_or_id} 对应的名称: {obj_info['name']}")
                        break
            
            # 如果找到了物体，删除它
            if name_or_id in self.objects:
                obj_info = self.objects[name_or_id]
                print(f"[删除物体] ID: {obj_info['id']}, 名称: {name_or_id}")
                
                try:
                    p.removeBody(obj_info['id'])
                    print(f"[删除成功] 物体ID: {obj_info['id']}")
                except p.error as e:
                    print(f"[警告] 删除物体时发生 PyBullet 错误: {e}")
                
                # 从列表和字典中移除
                self.object_list = [obj for obj in self.object_list if obj['id'] != obj_info['id']]
                del self.objects[name_or_id]
                
                # 执行几步仿真以确保场景稳定
                for _ in range(24):
                    self.step_simulation()
            else:
                print(f"[警告] 未找到物体: {name_or_id}")
                    
        except Exception as e:
            print(f"[错误] 删除物体时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def remove_all_objects(self):
        """移除所有物体"""
        print("[开始删除所有物体]")
        try:
            # 先获取所有物体ID
            all_ids = [obj_info['id'] for obj_info in self.object_list]
            
            # 直接从 PyBullet 中删除所有物体
            for obj_id in all_ids:
                try:
                    p.removeBody(obj_id)
                    print(f"[删除成功] 物体ID: {obj_id}")
                except p.error as e:
                    print(f"[警告] 删除物体 {obj_id} 时发生错误: {e}")
            
            # 清空列表和字典
            self.objects.clear()
            self.object_list.clear()
            
            # 执行几步仿真以确保场景稳定
            for _ in range(24):
                self.step_simulation()
                
            print("[删除完成] 所有物体已清除")
            
        except Exception as e:
            print(f"[错误] 删除所有物体时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def reset(self):
        self.robot.reset()
        self.remove_all_objects()
        self._object_counter = 0  # 重置物体计数器

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
        p.disconnect(self.physicsClient)