import time
import logging

import hydra
import numpy as np

import pybullet as p
import pybulletX as px

import tacto

from robot import Panda, UR5Robotiq85, UR5Robotiq140
    
log = logging.getLogger(__name__)

@hydra.main(config_path="cfg", config_name="grasp")
def main(cfg):
    # Initialize digits
    digits = tacto.Sensor(**cfg.tacto)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))

    # [21, 24]
    digits.add_camera(robot.id, robot.digit_links)

if __name__ == "__main__":
    main()