import urdfpy
import trimesh
import numpy as np
from trimesh.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix

def parse_urdf_collisions(urdf_path):
    """
    解析URDF文件中的碰撞体信息
    返回：碰撞体列表（包含类型、尺寸、位姿）
    """
    robot = urdfpy.URDF.load(urdf_path)
    collisions = []
    
    for link in robot.links:
        for collision in link.collisions:
            # 提取几何类型和尺寸
            geom = collision.geometry
            geom_type = None
            geom_size = {}
            
            if geom.box is not None:
                geom_type = "box"
                geom_size = {"size": geom.box.size}
            elif geom.cylinder is not None:
                geom_type = "cylinder"
                geom_size = {
                    "radius": geom.cylinder.radius,
                    "length": geom.cylinder.length
                }
            elif geom.sphere is not None:
                geom_type = "sphere"
                geom_size = {"radius": geom.sphere.radius}
            elif geom.mesh is not None:
                geom_type = "mesh"
                geom_size = {
                    "filename": geom.mesh.filename,
                    "scale": geom.mesh.scale
                }
            
            # 提取位姿（位置和旋转）
            origin = collision.origin
            if origin is None:
                # 默认位姿：无平移，无旋转
                pos = [0.0, 0.0, 0.0]
                rot_matrix = np.eye(4)
            else:
                # 直接处理为4x4齐次变换矩阵
                if isinstance(origin, np.ndarray):
                    # 位置为矩阵的平移部分 [0:3, 3]
                    pos = origin[0:3, 3]
                    rot_matrix = origin
                else:
                    # 兼容旧版本urdfpy（假设origin有translation和matrix属性）
                    pos = origin.translation
                    rot_matrix = origin.matrix
            
            # 将旋转矩阵转换为四元数
            try:
                rot_quat = quaternion_from_matrix(rot_matrix)
            except:
                rot_quat = [1.0, 0.0, 0.0, 0.0]  # 默认无旋转
            
            collisions.append({
                "link_name": link.name,
                "type": geom_type,
                "size": geom_size,
                "position": pos,
                "rotation": rot_quat
            })
    
    return collisions

def visualize_collisions(collisions):
    """
    可视化碰撞体
    """
    scene = trimesh.Scene()
    
    for col in collisions:
        # 根据类型生成几何体
        if col["type"] == "box":
            mesh = trimesh.creation.box(col["size"]["size"])
        elif col["type"] == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=col["size"]["radius"],
                height=col["size"]["length"]
            )
        elif col["type"] == "sphere":
            mesh = trimesh.creation.icosphere(
                radius=col["size"]["radius"],
                subdivisions=2  # 控制球体细分级别
            )
        elif col["type"] == "mesh":
            print(f"Warning: Mesh files are not loaded here. File: {col['size']['filename']}")
            continue
        else:
            continue
        
        # 应用位姿变换
        transform = translation_matrix(col["position"]) @ quaternion_matrix(col["rotation"])
        mesh.apply_transform(transform)
        
        # 添加到场景
        scene.add_geometry(mesh)
    
    # 显示场景
    scene.show()

if __name__ == "__main__":
    urdf_path = "path/to/your/robot.urdf"  # 替换为你的URDF文件路径
    collisions = parse_urdf_collisions(urdf_path)
    
    # 打印碰撞体规格
    print("===== Collision Bodies =====")
    for col in collisions:
        print(f"Link: {col['link_name']}")
        print(f"Type: {col['type']}")
        print(f"Size: {col['size']}")
        print(f"Position: {col['position']}")
        print(f"Rotation (quaternion): {col['rotation']}\n")
    
    # 可视化
    visualize_collisions(collisions)

    # urdf_path = "./urdf/ur5_robotiq_140.urdf"  # 替换为你的URDF文件路径