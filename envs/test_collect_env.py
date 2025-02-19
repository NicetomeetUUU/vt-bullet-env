import pybullet as p_bar
import time
import numpy as np

class RobotEnv:
    def __init__(self, robot, vis = True):
        self.robot = robot
        self.vis = vis

        self.physicsClient = p_bar.connect(p_bar.GUI if self.vis else p_bar.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def reset(self):
        """重置机器人"""
        self.robot.reset()

    def close(self):
        """关闭环境"""
        p.disconnect(self.physicsClient)