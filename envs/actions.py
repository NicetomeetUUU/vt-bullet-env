from dataclasses import dataclass
from typing import List, Union, Tuple
import numpy as np

@dataclass
class GripperCommand:
    """夹爪控制命令"""
    width: float  # 夹爪开合宽度

@dataclass
class JointAction:
    """关节空间控制命令"""
    joint_positions: List[float]  # 关节角度列表
    gripper: GripperCommand  # 夹爪命令

@dataclass
class PoseAction:
    """末端位姿控制命令"""
    position: Tuple[float, float, float]  # 位置 (x, y, z)
    orientation: Tuple[float, float, float, float]  # 四元数 (x, y, z, w)
    gripper: GripperCommand  # 夹爪命令

@dataclass
class GripperAction:
    """夹爪控制命令"""
    gripper: GripperCommand  # 夹爪命令

class ActionWrapper:
    """动作包装器，用于统一处理不同类型的控制命令"""
    def __init__(self, robot):
        self.robot = robot

    def execute_action(self, action: Union[JointAction, PoseAction, GripperAction]):
        """执行控制命令
        
        Args:
            action: 可以是关节控制、位姿控制或夹爪控制命令
        """
        if isinstance(action, JointAction):
            # 执行关节空间控制
            self.robot.move_joints(action.joint_positions)
            # 控制夹爪
            self.robot.move_gripper(action.gripper.width)
        elif isinstance(action, PoseAction):
            # 执行末端位姿控制
            self.robot.move_hand_to_pose(action.position, action.orientation)
            # 控制夹爪
            self.robot.move_gripper(action.gripper.width)
        elif isinstance(action, GripperAction):
            # 控制夹爪的同时保持机器人当前姿态
            self.robot.move_gripper(action.gripper.width)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")
