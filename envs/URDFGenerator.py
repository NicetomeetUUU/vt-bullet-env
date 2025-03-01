import os
import glob
import numpy as np
import pybullet as p
import trimesh
from pathlib import Path

import os
import glob
from pathlib import Path
import numpy as np
import trimesh
import json

class URDFGenerator:
    def __init__(self, model_dir: str, output_dir: str = None):
        """
        初始化URDF生成器
        Args:
            model_dir: 模型文件目录
            output_dir: URDF文件输出目录，默认为model_dir/urdf
        """
        self.model_dir = model_dir
        if output_dir is None:
            output_dir = os.path.join(model_dir, 'urdf')
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取obj文件
        obj_files = glob.glob(os.path.join(model_dir, "*.obj"))
        
        if not obj_files:
            raise ValueError("当前目录下缺少obj文件")
        
        self.obj_file = obj_files[0]
        self.urdf_file = os.path.join(output_dir, "object.urdf")

    def _analyze_mesh(self, mesh_file):
        """
        详细分析mesh文件，获取几何和物理属性
        Args:
            mesh_file: mesh文件路径
        Returns:
            dict: 包含详细mesh信息的字典
        """
        # 使用trimesh加载模型
        mesh = trimesh.load(mesh_file)
        
        # 基本几何信息
        center_mass = mesh.center_mass
        bounding_box = mesh.bounds
        dimensions = bounding_box[1] - bounding_box[0]
        volume = mesh.volume
        surface_area = mesh.area
        
        # 计算主惯性轴和主惯性矩
        inertia = mesh.moment_inertia
        principal_inertia, principal_axes = np.linalg.eigh(inertia)
        
        # 采样表面点云和法向量
        points, face_indices = trimesh.sample.sample_surface(mesh, count=1000)
        normals = mesh.face_normals[face_indices]
        
        # 计算凸包
        convex_hull = mesh.convex_hull
        
        # 分析网格拓扑
        is_watertight = mesh.is_watertight
        euler_number = mesh.euler_number
        
        return {
            'basic_info': {
                'center_mass': center_mass.tolist(),
                'bounding_box': bounding_box.tolist(),
                'dimensions': dimensions.tolist(),
                'volume': float(volume),
                'surface_area': float(surface_area)
            },
            'inertial_info': {
                'inertia_tensor': inertia.tolist(),
                'principal_inertia': principal_inertia.tolist(),
                'principal_axes': principal_axes.tolist()
            },
            'surface_info': {
                'points': points.tolist(),
                'normals': normals.tolist()
            },
            'topology_info': {
                'vertex_count': len(mesh.vertices),
                'face_count': len(mesh.faces),
                'is_watertight': is_watertight,
                'euler_number': euler_number,
                'genus': int(euler_number/2)
            },
            'convex_hull_info': {
                'volume': float(convex_hull.volume),
                'area': float(convex_hull.area)
            }
        }

    def _create_simplified_collision_mesh(self, mesh):
        """
        创建简化的碰撞网格
        Args:
            mesh: 原始trimesh网格
        Returns:
            simplified_mesh: 简化的网格
        """
        # 使用凸包作为碰撞网格
        collision_mesh = mesh.convex_hull
        
        # 如果顶点数过多，使用体素化简化
        if len(collision_mesh.vertices) > 100:
            # 使用体素化进行简化
            voxel_size = np.max(collision_mesh.extents) / 10.0
            collision_mesh = collision_mesh.voxelized(pitch=voxel_size).fill().as_boxes().convex_hull
        
        return collision_mesh

    def generate_urdf(self):
        """
        生成URDF文件和相关信息
        Returns:
            str: 生成的URDF文件路径
        """
        # 使用obj文件进行几何和物理分析
        mesh = trimesh.load(self.obj_file)
        mesh_info = self._analyze_mesh(self.obj_file)
        
        # 复制视觉模型并创建简化的碰撞模型
        visual_file = os.path.join(self.output_dir, "object_visual.obj")
        collision_file = os.path.join(self.output_dir, "object_collision.obj")
        
        # 加载并简化网格作为碰撞模型
        mesh = trimesh.load(self.obj_file)
        collision_mesh = self._create_simplified_collision_mesh(mesh)
        
        # 保存模型
        import shutil
        shutil.copy2(self.obj_file, visual_file)
        collision_mesh.export(collision_file)
        
        # 设置物理参数
        mass = 1.0  # kg
        density = mass / mesh_info['basic_info']['volume'] if mesh_info['basic_info']['volume'] > 0 else 1.0
        inertia = mesh_info['inertial_info']['inertia_tensor']
        center = mesh_info['basic_info']['center_mass']
        
        # 生成URDF文件
        urdf_content = f'''
<?xml version="1.0"?>
<robot name="object">
    <link name="base_link">
        <visual>
            <origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{os.path.basename(visual_file)}" scale="1 1 1"/>
            </geometry>
            <material name="material_0">
                <color rgba="0.8 0.8 0.8 1.0"/>
            </material>
        </visual>
        <collision>
            <origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{os.path.basename(collision_file)}" scale="1 1 1"/>
            </geometry>
        </collision>
        <inertial>
            <origin xyz="{center[0]} {center[1]} {center[2]}" rpy="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" ixz="{inertia[0][2]}"
                     iyy="{inertia[1][1]}" iyz="{inertia[1][2]}" izz="{inertia[2][2]}"/>
        </inertial>
    </link>
</robot>'''
        
        with open(self.urdf_file, 'w') as f:
            f.write(urdf_content)
        
        # 保存详细的模型信息
        info_path = os.path.join(self.output_dir, "object_info.json")
        with open(info_path, 'w') as f:
            json.dump(mesh_info, f, indent=4)
        
        # 保存表面点云和法向量
        points_path = os.path.join(self.output_dir, "object_surface_points.npy")
        normals_path = os.path.join(self.output_dir, "object_surface_normals.npy")
        np.save(points_path, mesh_info['surface_info']['points'])
        np.save(normals_path, mesh_info['surface_info']['normals'])
        
        print(f"生成了URDF文件和相关信息：{self.output_dir}")
        print(f"  - 视觉模型：{os.path.basename(visual_file)}")
        print(f"  - 碰撞模型（简化）：{os.path.basename(collision_file)}")
        
        return self.urdf_file

    def get_model_info(self):
        """获取模型的详细信息
        Returns:
            dict: 模型信息字典
        """
        info_path = os.path.join(self.output_dir, "object_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        return None

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置模型目录和输出目录
    model_dir = os.path.join(current_dir, "../models/002")
    output_dir = model_dir  # 直接在同一目录下生成
    
    # 创建生成器
    generator = URDFGenerator(model_dir=model_dir, output_dir=output_dir)
    
    # 生成URDF文件
    urdf_files = generator.generate_urdf()

if __name__ == "__main__":
    main()