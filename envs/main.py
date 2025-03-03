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

class TactileDataCollector:

    def __init__(self, save_dir, visualize_gui = True):
        """初始化触觉数据采集器
        
        Args:
            save_dir (str): 视觉数据保存、抓取位姿读取目录
            visualize_gui (bool): 是否可视化GUI
        """
        self.save_dir = save_dir
        self.visualize_gui = visualize_gui
        
        # 初始化机器人和环境
        robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
        self.env = VTGraspRefine(robot, vis=visualize_gui)
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建固定在世界坐标系中的相机
        camera_distance = 0.5  # 相机到观察点的距离
        camera_position = [0,  # x
                         -camera_distance * math.cos(math.pi/4),  # y
                          0.5]  # z
        camera_target = [0, 0, 0]  # 相机观察的目标点（世界坐标系原点）
        up_vector = [0, 0, 1]  # 相机向上的方向
        
        camera = Camera(
            robot_id=None,  # 不附着在机器人上
            ee_id=None,    # 不附着在末端执行器上
            near=0.1,
            far=5.0,
            fov=60,
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
        
        # 初始化对象列表
        self.object_list = []
        
        # 初始化相机和触觉传感器的可视化
        if visualize_gui:
            # 渲染相机图像
            rgb, depth, seg = self.env.camera.shot()
            # 渲染触觉传感器数据
            tactile_rgb, tactile_depth = self.digits.render()
            self.digits.updateGUI(tactile_rgb, tactile_depth)
            # 执行几步仿真使显示更稳定
            for _ in range(10):
                self.env.step_simulation()
        
        rgb, depth, seg = self.env.camera.shot()

    def add_object(self, urdf_file, position=(0, 0, 0)):
        """添加物体到场景
        Args:
            urdf_file (str): URDF文件路径
            position (tuple): 期望放置物体的位置 (x, y, z)，默认为原点
        Returns:
            int: 物体的ID
        """
        # 直接调用环境的add_object方法
        return self.env.add_object(urdf_file=urdf_file, position=position)
        
    def remove_objects(self, object_ids=None):
        """删除场景中的物体
        Args:
            object_ids (int or list, optional): 要删除的物体ID或ID列表。如果为None，删除所有物体
        """
        try:
            if object_ids is None:
                # 删除所有物体
                self.env.remove_all_objects()
            else:
                # 将单个ID转换为列表
                if isinstance(object_ids, int):
                    object_ids = [object_ids]
                    
                # 删除指定ID的物体
                for obj_id in object_ids:
                    self.env.remove_object(obj_id)
                    
        except Exception as e:
            print(f"删除物体时发生错误: {e}")
        
        
    def collect_tactile_data(self):
        """采集当前时刻的触觉数据"""
        rgb, depth = self.digits.render()
        if (self.visualize_gui):
            self.digits.updateGUI(rgb, depth)
            # 增加物理仿真步数和延时使渲染更流畅
            for i in range(20):
                self.env.step_simulation()
                time.sleep(0.01)  # 10ms延时
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
        self.env.step_actions(action, 240)
    
    def reset(self):
        """重置环境"""
        self.env.reset()
    
    def get_poses(self):
        """获取当前抓取位姿"""
        return self.env.get_poses()
    
    def get_camera_image(self):
        """获取相机图像
        
        Returns:
            tuple: (rgb, depth, seg)
                - rgb: RGB图像数据
                - depth: 深度图数据
                - seg: 分割图数据
        """
        return self.env.camera.shot()
    
    def __del__(self):
        """析构函数，用于清理资源"""
        try:
            # 删除所有物体
            self.remove_objects()
            
            # 关闭GUI窗口
            if hasattr(self, 'digits') and hasattr(self.digits, '_gui'):
                self.digits._gui.close()
            
            # 断开PyBullet连接
            if hasattr(self, 'env') and self.env.physicsClient is not None:
                p.disconnect(self.env.physicsClient)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
if __name__ == '__main__':
    import time
    import numpy as np
    
    # 创建环境
    collector = TactileDataCollector(save_dir='./grasp_data')
    
    try:
        while True:
            print("\n可用命令:")
            print("1: 添加物体")
            print("2: 移除物体")
            print("3: 移动到抓取位姿")
            print("4: 控制夹爪")
            print("5: 采集触觉数据")
            print("6: 重置环境")
            print("q: 退出")
            
            cmd = input('\n请输入命令: ')
            
            if cmd == 'q':
                break
                
            elif cmd == '1':
                # 添加物体
                urdf_path = "/home/iccd-simulator/code/vt-bullet-env/models/000/object.urdf"
                position = (0.3, 0, 0.1)
                obj_id = collector.add_object(urdf_path)
                #obj_id = collector.add_object(urdf_path, position)
                print(f"物体添加成功，ID: {obj_id}")
                
            elif cmd == '2':
                # 移除物体
                collector.remove_objects()
                print("所有物体已移除")
                
            elif cmd == '3':
                # 测试不同的抓取位姿
                poses = [
                    # 位置，姿态（四元数）
                    ((0.3, 0.5, 0.3), (0, 0, 0, 1)),  # 上方
                ]
                
                for pos, orn in poses:
                    print(f"\n移动到位置: {pos}, 姿态: {orn}")
                    collector.move_to_grasp_pose(pos, orn)
                    time.sleep(1)
                
            elif cmd == '4':
                # 控制夹爪
                gripper_width = float(input("请输入夹爪开合宽度（0.0表示完全关闭，0.14表示完全打开）："))
                collector.control_gripper(gripper_width)
                # 执行足够多的仿真步骤以确保动作完成
                
            elif cmd == '5':
                # 采集触觉数据
                rgb, depth = collector.collect_tactile_data()
                # print(f"\n触觉数据尺寸: RGB {rgb.shape}, Depth {depth.shape}")
                
            elif cmd == '6':
                # 重置环境
                collector.reset()
                print("环境已重置")
            
            else:
                print("无效命令")
                
    finally:
        # 清理资源
        collector.__del__()
