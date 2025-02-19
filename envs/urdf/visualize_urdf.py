import pybullet as p
import pybullet_data

# 连接到物理引擎
p.connect(p.GUI)

# 设置搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载URDF文件
urdf_path = '/home/iccd-simulator/code/vt-bullet-env/envs/urdf/test_urdf.urdf'
robot_id = p.loadURDF(urdf_path)

# 绘制坐标轴
axis_length = 0.5
p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], 3)  # X轴
p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], 3)  # Y轴
p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], 3)  # Z轴

# 运行仿真
while True:
    p.stepSimulation()
