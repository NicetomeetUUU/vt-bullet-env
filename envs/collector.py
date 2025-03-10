import os
import numpy as np
import pybullet as p
import tacto
from env import VTGraspRefine
from robot import UR5Robotiq140
from utilities import Camera, ModelLoader
import math
from actions import PoseAction, GripperCommand

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
        robot = UR5Robotiq140((0, 0.5, 0), (0, 0, math.pi/2))  # 绕z轴旋转90度
        self.env = VTGraspRefine(robot, vis=visualize_gui)
        os.makedirs(save_dir, exist_ok=True)

        self.obj_loaders = {}
        
        # 创建固定在世界坐标系中的相机
        camera_height = 0.5  # 相机的高度
        camera_position = [0,   # x坐标（正上方）
                         0,    # y坐标（正上方）
                         camera_height]  # z坐标（高度）
        camera_target = [0, 0, 0]  # 相机观察的目标点（世界坐标系原点）
        up_vector = [0, 1, 0]  # 相机向上的方向（y轴正方向）
        
        camera = Camera(
            robot_id=None,  # 不附着在机器人上
            ee_id=None,    # 不附着在末端执行器上
            position=camera_position,
            target=camera_target,
            up_vector=up_vector
        )
        print("展示信息camera:", camera.get_pose())
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
        tactile_rgb, tactile_depth = self.digits.render()
        self.digits.updateGUI(tactile_rgb, tactile_depth)
        # 执行几步仿真使显示更稳定
        for _ in range(10):
            self.env.step_simulation()
        
    def load_object(self, object_path, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        """加载物体
        Args:
            object_path (str): 物体的路径
        """
        obj_loader = ModelLoader(object_path)
        obj_info = obj_loader.load_object(position=position, orientation=orientation)
        obj_id = obj_info['id']
        self.obj_loaders[obj_id] = obj_loader
        for _ in range(100):
            self.env.step_simulation()
        return obj_id

    def remove_object(self, obj_id):
        if obj_id in self.obj_loaders:
            obj_loader = self.obj_loaders[obj_id]
            success = obj_loader.remove_object()
            if not success:
                raise Exception(f"移除物体失败: {obj_id}")
            del self.obj_loaders[obj_id]
            for _ in range(100):
                self.env.step_simulation()
            return True
        return False
    
    def remove_all_objects(self):
        """移除所有物体"""
        # 先获取所有物体ID的列表
        obj_ids = list(self.obj_loaders.keys())
        # 遍历列表删除物体
        for obj_id in obj_ids:
            self.remove_object(obj_id)
        
    def collect_tactile_data(self):
        """采集当前时刻的触觉数据"""
        rgb, depth = self.digits.render()
        if (self.visualize_gui):
            self.digits.updateGUI(rgb, depth)
            # 增加物理仿真步数和延时使渲染更流畅
        for _ in range(48):
            self.env.step_simulation()
        return rgb, depth
        
    def move_to_grasp_pose(self, target_pos, target_orn, gripper_width=0.04):
        """控制机器人末端移动到目标抓取位姿
        
        Args:
            target_pos (tuple): 目标位置 (x, y, z)
            target_orn (tuple): 目标姿态四元数 (x, y, z, w)
            gripper_width (float): 夹爪开度，默认0.04表示完全打开
        """
        
        # 创建位姿动作
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
        from actions import GripperAction, GripperCommand
        
        # 创建夹爪动作
        action = GripperAction(
            gripper=GripperCommand(width=gripper_width)
        )
        # 执行动作并添加延时使运动更流畅
        self.env.step_actions(action, 120)
    
    def reset(self):
        """重置环境"""
        self.env.reset()
        self.digits.add_camera(self.env.robot._id, self.env.robot.digit_links)
    
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
    
    def __del__(self):
        """析构函数，用于清理资源"""
        try:
            # 删除所有物体
            
            # 关闭GUI窗口
            if hasattr(self, 'digits') and hasattr(self.digits, '_gui'):
                self.digits._gui.close()
            
            # 断开PyBullet连接
            if hasattr(self, 'env') and self.env.physicsClient is not None:
                p.disconnect(self.env.physicsClient)
        except Exception as e:
            print(f"Error during cleanup: {e}")