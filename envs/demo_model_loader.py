import pybullet as p
import time
from robot import UR5Robotiq140
from utilities import ModelLoader, Camera
from env import VTGraspRefine

def demo_model_loader():
    # 创建机器人
    robot = UR5Robotiq140((0, 0.5, 0), (0, 0, 0))
    
    # 创建相机
    camera = Camera(
        robot_id=robot._id,
        ee_id=robot.end_effector_index(),
        near=0.1,
        far=5.0,
        fov=60
    )
    
    # 初始化环境
    env = VTGraspRefine(robot, camera=camera, vis=True)
    
    # 创建模型加载器
    cube_loader = ModelLoader(urdf_file="./urdf/cube.urdf")
    
    try:
        # 加载一个立方体到环境中
        cube_id = cube_loader.load_object(position=(0, 0, 0.1), scale=1.0)
        print(f"加载的立方体ID: {cube_id}")
        
        # 等待一段时间观察
        time.sleep(5)
        
        # 移除立方体
        cube_loader.remove_object()
        print("立方体已移除")
        
        # 再次加载立方体，但这次位置和大小不同
        cube_id = cube_loader.load_object(position=(0.2, 0.2, 0.1), scale=0.5)
        print(f"重新加载的立方体ID: {cube_id}")
        
        # 再等待一段时间
        time.sleep(5)
        
    finally:
        # 清理环境
        env.close()

if __name__ == '__main__':
    demo_model_loader()
