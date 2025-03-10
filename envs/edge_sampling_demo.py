import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial import KDTree
import time

def detect_edges(points, normals, threshold_angle=45):
    """检测边缘点
    通过比较相邻点的法向量来检测边缘
    Args:
        points: np.array, 形状为(N, 3)的点云数据
        normals: np.array, 形状为(N, 3)的法向量数据
        threshold_angle: float, 判定为边缘的角度阈值（度）
    Returns:
        edge_mask: np.array, 布尔数组标识边缘点
    """
    # 构建KD树来找到近邻点
    tree = KDTree(points)
    # 找到每个点的最近5个邻居
    distances, indices = tree.query(points, k=5)
    
    # 计算法向量夹角
    edge_mask = np.zeros(len(points), dtype=bool)
    threshold_cos = np.cos(np.radians(threshold_angle))
    
    for i in range(len(points)):
        neighbors_idx = indices[i][1:]  # 排除自身
        # 计算与邻居的法向量夹角
        cos_angles = np.abs(np.dot(normals[neighbors_idx], normals[i]))
        # 如果有任何邻居的法向量夹角大于阈值，认为是边缘点
        if np.any(cos_angles < threshold_cos):
            edge_mask[i] = True
            
    return edge_mask

def sample_surface(obj_id, base_voxel_size=0.004, edge_voxel_size=0.002):
    """对物体表面进行采样，边缘处更密集
    Args:
        obj_id: int, PyBullet中的物体ID
        base_voxel_size: float, 非边缘区域的采样密度
        edge_voxel_size: float, 边缘区域的采样密度
    Returns:
        points: np.array, 采样得到的点云
        normals: np.array, 对应的法向量
    """
    # 获取物体的AABB包围盒
    aabb_min, aabb_max = p.getAABB(obj_id)
    
    # 创建密集采样网格
    x = np.arange(aabb_min[0], aabb_max[0], edge_voxel_size)
    y = np.arange(aabb_min[1], aabb_max[1], edge_voxel_size)
    z = np.arange(aabb_min[2], aabb_max[2], edge_voxel_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    # 从每个网格点向各个方向发射射线
    directions = [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ]
    
    points = []
    normals = []
    
    for start_pos in zip(xx.flatten(), yy.flatten(), zz.flatten()):
        for direction in directions:
            result = p.rayTest(start_pos, 
                             [start_pos[0] + direction[0]*0.01,
                              start_pos[1] + direction[1]*0.01,
                              start_pos[2] + direction[2]*0.01])
            
            if result[0][0] == obj_id:  # 如果射线击中了目标物体
                hit_pos = result[0][3]
                hit_normal = result[0][4]
                points.append(hit_pos)
                normals.append(hit_normal)
    
    points = np.array(points)
    normals = np.array(normals)
    
    # 去除重复点
    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    unique_normals = normals[unique_indices]
    
    # 检测边缘点
    edge_mask = detect_edges(unique_points, unique_normals)
    
    # 对非边缘区域进行降采样
    non_edge_points = unique_points[~edge_mask]
    non_edge_normals = unique_normals[~edge_mask]
    
    # 使用体素化进行降采样
    voxel_grid = {}
    for i, point in enumerate(non_edge_points):
        voxel_key = tuple(np.floor(point / base_voxel_size))
        if voxel_key not in voxel_grid:
            voxel_grid[voxel_key] = (point, non_edge_normals[i])
    
    # 合并边缘点和降采样后的非边缘点
    final_points = np.vstack([
        unique_points[edge_mask],
        np.array([p for p, _ in voxel_grid.values()])
    ])
    final_normals = np.vstack([
        unique_normals[edge_mask],
        np.array([n for _, n in voxel_grid.values()])
    ])
    
    return final_points, final_normals

def main():
    # 初始化PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # 加载平面和物体
    planeId = p.loadURDF("plane.urdf")
    objId = p.loadURDF("duck.urdf", [0, 0, 1])
    
    # 采样物体表面
    points, normals = sample_surface(objId)
    
    # 可视化采样点
    for point in points:
        p.addUserDebugPoints([point], [[1,0,0]], pointSize=3)
    
    # 保持窗口打开
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    main()
