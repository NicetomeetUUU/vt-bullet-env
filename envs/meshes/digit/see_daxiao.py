import os
import numpy as np
from stl import mesh

your_mesh = mesh.Mesh.from_file('digit.STL')
vertices = your_mesh.vectors.reshape(-1, 3)

# 计算包围盒尺寸（X, Y, Z 长度）
min_coords = np.min(vertices, axis=0)
max_coords = np.max(vertices, axis=0)
dimensions = max_coords - min_coords

print(f"模型尺寸 (X, Y, Z): {dimensions}")
print(f"最小坐标: {min_coords}")
print(f"最大坐标: {max_coords}")