import os
import numpy as np
import pybullet as p
import tacto    
from tqdm import tqdm
from env import VTGraspRefine
from robot import UR5Robotiq140
from utilities import Camera, ModelLoader
from grasp_bridge import GraspBridge
from visualization import PoseVisualizer
import time
import math
import cv2

def display_camera_data(rgb, depth, seg):
    # 转换 RGB 图像格式
    rgb = rgb[:, :, :3]  # 移除 alpha 通道
    rgb = (rgb * 255).astype(np.uint8)  # 转换为0-255范围
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
    
    # 归一化深度图像用于显示
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 归一化分割图像用于显示
    seg_normalized = cv2.normalize(seg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 创建组合图像
    combined = np.hstack((rgb, depth_colormap, cv2.cvtColor(seg_normalized, cv2.COLOR_GRAY2BGR)))
    
    # 显示图像
    cv2.imshow('Camera View (RGB | Depth | Segmentation)', combined)
    cv2.waitKey(1)  # 1ms 延迟，允许窗口更新


def user_control_demo():
    # 创建机器人
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    grasp_bridge = GraspBridge(save_dir='./grasp_data')
    object_00 = ModelLoader(urdf_file="/home/iccd-simulator/code/vt-bullet-env/models/002/object.urdf")
    
    # 创建位姿可视化器
    visualizer = PoseVisualizer(size=0.1)
    
    # 初始化环境（这会加载机器人）
    env = VTGraspRefine(robot, vis=True)
    
    # 现在机器人已经加载，可以创建相机
    camera = Camera(
        robot_id=robot._id,
        ee_id=robot.end_effector_index(),
        near=0.1,
        far=5.0,
        fov=60
    )
    
    # 设置环境的相机（如果想绑定相机到机械臂末端，必须等robot载入到环境内后执行）
    env.camera = camera

    # 导入物体（可以动态导入倒是）
    object_00.load_object(position=(0, 0, 0.3))
    env.step_simulation()
    
    # 设置触觉传感器
    digits = tacto.Sensor(width=120, height=160, visualize_gui=True)
    digits.add_camera(robot.id, robot.digit_links)
    env.reset()
    env.step_simulation()
    
    # 导入动作类
    from actions import JointAction, PoseAction, GripperCommand
    
    while True:    
        # 获取当前状态
        current_pos, current_orn = robot.get_ee_pos()
        print('Current end-effector position:', current_pos)
        
        # 获取相机和触觉传感器数据
        rgb, depth, seg = camera.shot()
        color, tactile_depth = digits.render()
        digits.updateGUI(color, tactile_depth)
        # display_camera_data(rgb, depth, seg)
        
        # 等待用户输入
        cmd = input('Commands: (q)uit, (j)oint control, (p)ose control, (o)pen gripper, (c)lose gripper: ')
        
        if cmd.lower() == 'q':
            break
        elif cmd.lower() == 'j':
            # 关节空间控制示例
            joint_action = JointAction(
                joint_positions=[0, -1.57, 1.57, -1.57, -1.57, 0],
                gripper=GripperCommand(width=0.085)
            )
            env.step_actions(joint_action)
            
        elif cmd.lower() == 'p':
            # 末端位姿控制示例
            next_pos = (current_pos[0] + 0.1, current_pos[1], current_pos[2] + 0.1)
            pose_action = PoseAction(
                position=next_pos,
                orientation=current_orn,
                gripper=GripperCommand(width=0.085)
            )
            env.step_actions(pose_action)
            
        elif cmd.lower() == 'o':
            # 开合器示例
            current_pos, current_orn = robot.get_ee_pos()
            gripper_action = PoseAction(
                position=current_pos,
                orientation=current_orn,  # 使用当前姿态
                gripper=GripperCommand(width=0.140)
            )
            env.step_actions(gripper_action)
            
        elif cmd.lower() == 'c':
            # 关合器示例
            current_pos, current_orn = robot.get_ee_pos()
            gripper_action = PoseAction(
                position=current_pos,
                orientation=current_orn,  # 使用当前姿态
                gripper=GripperCommand(width=0.0)
            )
            env.step_actions(gripper_action)
        
        # 执行仿真步骤
        for i in range(20):  # 减少循环次数
            # 每5步更新一次传感器数据
            env.step_simulation()

            if i % 5 == 0:
                # 获取并更新位姿可视化
                ee_pos, ee_orn = robot.get_ee_pos()
                visualizer.update_pose('End Effector', ee_pos, ee_orn)
                
                camera_pos, camera_orn = camera.get_pose()
                visualizer.update_pose('Camera', camera_pos, camera_orn)
                
                # # 获取触觉传感器位姿
                # digits_pos = p.getLinkState(robot._id, robot.digit_id)[0]
                # digits_orn = p.getLinkState(robot._id, robot.digit_id)[1]
                # visualizer.update_pose('Digits', digits_pos, digits_orn)
                
                # 更新传感器数据
                rgb, depth, seg = camera.shot()
                color, tactile_depth = digits.render()
                digits.updateGUI(color, tactile_depth)
                # display_camera_data(rgb, depth, seg)

class TactileDataCollector:

    def __init__(self, save_dir, visualize_gui = True):
        """初始化触觉数据采集器
        
        Args:
            save_dir (str): 视觉数据保存、抓取位姿读取目录
            visualize_gui (bool): 是否可视化GUI
        """
        self.save_dir = save_dir
        robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
        self.env = VTGraspRefine(robot, vis=visualize_gui)
        os.makedirs(save_dir, exist_ok=True)
        # 创建相机
        camera = Camera(
            robot_id=robot._id,
            ee_id=robot.end_effector_index(),
            near=0.1,
            far=5.0,
            fov=60
        )
        object_00 = ModelLoader(urdf_file="/home/iccd-simulator/code/vt-bullet-env/models/002/object.urdf")
        # 设置环境的相机（如果想绑定相机到机械臂末端，必须等robot载入到环境内后执行）
        self.env.camera = camera
        # 导入物体（可以动态导入倒是）
        object_00.load_object(position=(0, 0, 0.3))
        self.env.step_simulation()
        
        # 设置触觉传感器
        self.digits = tacto.Sensor(width=120, height=160, visualize_gui=visualize_gui)
        self.digits.add_camera(robot._id, robot.digit_links)
        self.env.reset()
        self.env.step_simulation()
        
        # 初始化对象列表
        self.object_list = []

    def add_object(self, urdf_file, position):
        """添加物体到场景
        
        Args:
            urdf_file (str): URDF文件路径
            position (tuple): 物体位置 (x, y, z)
            
        Returns:
            int: 物体的ID
        """
        object_loaded = ModelLoader(urdf_file=urdf_file)
        obj_id = object_loaded.load_object(position=position)
        self.object_list.append({
            'id': obj_id,
            'urdf': urdf_file,
            'position': position
        })
        self.env.step_simulation()
        return obj_id
        
    def remove_objects(self, object_ids=None):
        """删除场景中的物体
        
        Args:
            object_ids (list, optional): 要删除的物体ID列表。如果为None，删除所有物体
        """
        if object_ids is None:
            # 删除所有物体
            for obj in self.object_list:
                p.removeBody(obj['id'])
            self.object_list.clear()
        else:
            # 删除指定ID的物体
            self.object_list = [obj for obj in self.object_list if obj['id'] not in object_ids]
            for obj_id in object_ids:
                p.removeBody(obj_id)
        
        self.env.step_simulation()
        
        
    def collect_tactile_data(self):
        """采集当前时刻的触觉数据"""
        rgb, depth = self.digits.render()
        if (self.visualize_gui):
            self.digits.updateGUI(rgb, depth)
        return rgb, depth
        
    def move_to_grasp_pose(self, target_pos, target_orn, gripper_width=0.04):
        """控制机器人末端移动到目标抓取位姿
        
        Args:
            target_pos (tuple): 目标位置 (x, y, z)
            target_orn (tuple): 目标姿态四元数 (x, y, z, w)
            gripper_width (float): 夹爪开度，默认0.04表示完全打开
        """
        from actions import PoseAction, GripperCommand
        
        # 创建位姿动作
        action = PoseAction(
            position=target_pos,
            orientation=target_orn,
            gripper=GripperCommand(width=gripper_width)
        )
        
        # 执行动作
        self.env.step_actions(action, 100)
        
    def control_gripper(self, gripper_width):
        """控制夹爪开合
        
        Args:
            gripper_width (float): 夹爪开合宽度，0.0表示完全关闭，0.04表示完全打开
        """
        from actions import GripperAction, GripperCommand
        
        # 创建夹爪动作
        action = GripperAction(
            gripper=GripperCommand(width=gripper_width)
        )
        # 执行动作
        self.env.step_actions(action, 10)
    
    def reset(self):
        """重置环境"""
        self.env.reset()
    
    def get_poses(self):
        """获取当前抓取位姿"""
        return self.env.get_poses()
    
    def __del__(self):
        """析构函数，用于清理资源"""
        try:
            # 删除所有物体
            self.remove_objects()
            
            # 关闭GUI窗口
            if hasattr(self, 'digits'):
                self.digits.close()
            
            # 断开PyBullet连接
            if hasattr(self, 'env') and self.env.physicsClient is not None:
                p.disconnect(self.env.physicsClient)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
if __name__ == '__main__':
    user_control_demo()
