import pybullet as p
import numpy as np

class PoseVisualizer:
    def __init__(self, size=0.1):
        """初始化位姿可视化器
        
        Args:
            size: 坐标轴的长度，默认0.1米
        """
        self.size = size
        self._line_ids = {}
        
    def update_pose(self, name: str, position, orientation):
        """更新某个位姿的可视化
        
        Args:
            name: 位姿的标识名称
            position: 位置 [x, y, z]
            orientation: 方向四元数 [x, y, z, w]
        """
        # 计算旋转矩阵
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # 创建三个轴的终点
        x_end = np.array(position) + self.size * rot_matrix[:, 0]  # X轴（红色）
        y_end = np.array(position) + self.size * rot_matrix[:, 1]  # Y轴（绿色）
        z_end = np.array(position) + self.size * rot_matrix[:, 2]  # Z轴（蓝色）
        
        # 如果已经存在，先删除旧的线条
        if name in self._line_ids:
            for line_id in self._line_ids[name]:
                p.removeUserDebugItem(line_id)
        
        # 绘制新的坐标轴
        lines = []
        # X轴（红色）
        lines.append(p.addUserDebugLine(position, x_end, [1, 0, 0], 2))
        # Y轴（绿色）
        lines.append(p.addUserDebugLine(position, y_end, [0, 1, 0], 2))
        # Z轴（蓝色）
        lines.append(p.addUserDebugLine(position, z_end, [0, 0, 1], 2))
        
        # 添加文本标签
        lines.append(p.addUserDebugText(name, position, [1, 1, 1]))
        
        self._line_ids[name] = lines
        
    def clear(self):
        """清除所有可视化元素"""
        for lines in self._line_ids.values():
            for line_id in lines:
                p.removeUserDebugItem(line_id)
        self._line_ids.clear()
