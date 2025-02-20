import os

import numpy as np
import pybullet as p
import tacto    

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import Panda, UR5Robotiq85, UR5Robotiq140
from utilities import YCBModels, Camera
import time
import math
import cv2
import numpy as np

def display_camera_data(rgb, depth, seg):
    # 转换 RGB 图像格式
    rgb = rgb[:, :, :3]  # 移除 alpha 通道
    
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
    # ycb_models = YCBModels(
    #     os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    # )
    # camera = None
    # robot = Panda((0, 0.5, 0), (0, 0, math.pi))
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    env = ClutteredPushGrasp(robot, None, vis=True)
    
    # 获取机器人末端执行器的ID
    ee_id = robot.end_effector_index()
    
    # 创建绑定到末端执行器的相机
    camera = Camera(
        robot_id=robot.id,  # 使用robot.id而不是robot.robot_model.uid
        ee_id=ee_id,
        size=(320, 320),
        near=0.1,
        far=5.0,
        fov=60
    )

    # 设置触觉传感器
    digits = tacto.Sensor(width=120, height=160, visualize_gui=True)
    digits.add_camera(robot.id, robot.digit_links)
    env.camera = camera  # 更新环境中的相机
    env.reset()
    # env.SIMULATION_STEP_DELAY = 0
    obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
    cnt = 0
    while True:    
        if cnt == 3:
            cnt = 0
            robot.reset()
        
        current_pos = robot.get_ee_pos()
        print(current_pos)
        # 获取并显示相机数据
        rgb, depth, seg = camera.shot()
        print("Depth data type:", type(depth))
        print("Depth data shape:", depth.shape)
        print("Depth data sample:", depth)
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        # display_camera_data(rgb, depth, seg)
        robot.reset()
        robot.move_gripper(0.140)
        input('Press Enter to continue...')
        # robot.move_ee(current_pos, 'q_end')
        # 执行多步模拟以确保运动完成
        for _ in range(100):
            env.step_simulation()
            # 更新相机视图
            rgb, depth, seg = camera.shot()
            color, depth = digits.render()
            digits.updateGUI(color, depth)
            # display_camera_data(rgb, depth, seg)
        input('Press Enter to continue...')
        robot.move_gripper(0)
        for _ in range(100):
            env.step_simulation()
            # 更新相机视图
            rgb, depth, seg = camera.shot()
            color, depth = digits.render()
            digits.updateGUI(color, depth)
            # display_camera_data(rgb, depth, seg)
        # print(obs, reward, done, info)
        cnt += 1

if __name__ == '__main__':
    user_control_demo()
