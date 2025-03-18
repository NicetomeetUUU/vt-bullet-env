import numpy as np
import open3d as o3d
import argparse

def visualize_point_cloud(points_file, normals_file=None, info_file=None):
    """
    可视化点云数据
    Args:
        points_file: 点云文件路径（.npy格式）
        normals_file: 法向量文件路径（.npy格式，可选）
        info_file: 模型信息文件路径（.json格式，可选）
    """
    # 加载点云数据
    
    points = np.load(points_file)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 如果有法向量数据，加载并设置
    if normals_file:
        normals = np.load(normals_file)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 为点云设置随机颜色
    colors = np.random.uniform(0, 1, size=(len(points), 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    
    # 如果有模型信息，显示质心
    geometries = [pcd, coordinate_frame]
    if info_file:
        import json
        with open(info_file, 'r') as f:
            info = json.load(f)
            center_mass = info['basic_info']['center_mass']
            
            # 创建质心球体
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(center_mass)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            geometries.append(sphere)
    
    # 可视化
    o3d.visualization.draw_geometries(geometries,
                                    window_name="点云可视化",
                                    width=800,
                                    height=600)

def main():
    parser = argparse.ArgumentParser(description="点云可视化工具")
    parser.add_argument("points_file", help="点云文件路径（.npy格式）")
    parser.add_argument("--normals", help="法向量文件路径（.npy格式，可选）", default=None)
    parser.add_argument("--info", help="模型信息文件路径（.json格式，可选）", default=None)
    args = parser.parse_args()
    
    visualize_point_cloud(args.points_file, args.normals, args.info)

if __name__ == "__main__":
    main()
