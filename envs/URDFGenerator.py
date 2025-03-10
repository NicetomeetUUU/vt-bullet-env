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
    def __init__(self, model_dir: str, output_dir: str = None, voxel_size: float = 0.004):
        """
        初始化URDF生成器
        Args:
            model_dir: 模型文件目录
            output_dir: URDF文件输出目录，默认为model_dir/urdf
            voxel_size: 点云采样的体素大小（米），默认4mm
        """
        self.model_dir = model_dir
        if output_dir is None:
            output_dir = os.path.join(model_dir, 'urdf')
        self.output_dir = output_dir
        self.voxel_size = voxel_size  # 保存体素大小设置
        
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
        self.mesh = mesh  # 保存mesh对象供后续使用
        
        # 基本几何信息
        center_mass = mesh.center_mass
        bounding_box = mesh.bounds
        dimensions = bounding_box[1] - bounding_box[0]
        volume = mesh.volume
        surface_area = mesh.area
        
        # 计算主惯性轴和主惯性矩
        inertia = mesh.moment_inertia
        principal_inertia, principal_axes = np.linalg.eigh(inertia)
        
        # 密集采样表面点云，每平方毫米采样100个点
        points_per_area = 100  # 每平方毫米的点数
        total_points = int(surface_area * 1e6 * points_per_area)  # 转换为平方毫米
        points, _ = trimesh.sample.sample_surface(mesh, count=total_points)
        
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
                'point_density': points_per_area
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
        
    def get_dense_point_cloud(self):
        """获取密集点云数据，使用设定的体素大小进行采样，并将点云中心对齐到URDF坐标系
        
        Returns:
            np.ndarray: points array with shape (N, 3)
        """
        if not hasattr(self, 'mesh'):
            self.mesh = trimesh.load(self.obj_file)
        
        # 使用设置的体素大小
        voxel_size = self.voxel_size
        
        # 计算物体尺寸
        bbox = self.mesh.bounds
        object_size = np.max(bbox[1] - bbox[0])
        
        # 计算每平方米的点数
        points_per_area = 1 / (voxel_size ** 2)
        
        # 计算总采样点数
        total_points = int(self.mesh.area * points_per_area)
        
        # 确保最少有1000个点
        total_points = max(1000, total_points)
        
        # 采样点云
        points, _ = trimesh.sample.sample_surface(self.mesh, count=total_points)
        
        # 将点云中心对齐到URDF坐标系（mesh的质心）
        points_center = np.mean(points, axis=0)
        urdf_origin = self.mesh.center_mass
        
        # 平移点云到URDF原点
        points = points - points_center + urdf_origin
        
        return points

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

    def save_dense_point_cloud(self, prefix='dense'):
        """保存密集点云数据
        
        Args:
            voxel_size: 体素大小（米）。如果为None，则根据物体尺寸自动计算
            prefix: 文件名前缀
            
        Returns:
            dict: 保存的文件路径信息
        """
        # 获取密集点云数据
        points = self.get_dense_point_cloud()
        
        # 计算物体尺寸和体素大小
        bbox = self.mesh.bounds
        object_size = np.max(bbox[1] - bbox[0])
        actual_voxel_size = np.sqrt(self.mesh.area / len(points))
        
        # 构建文件路径
        points_path = os.path.join(self.output_dir, f"{prefix}_points.npy")
        metadata_path = os.path.join(self.output_dir, f"{prefix}_metadata.json")
        
        # 保存点云数据
        np.save(points_path, points)
        
        # 保存元数据
        metadata = {
            'object_info': {
                'size': float(object_size),
                'surface_area': float(self.mesh.area),
                'bounding_box': self.mesh.bounds.tolist(),
                'center_mass': self.mesh.center_mass.tolist()
            },
            'sampling_info': {
                'target_voxel_size': float(self.voxel_size),  # 使用设置的体素大小
                'actual_voxel_size': float(actual_voxel_size),
                'total_points': len(points),
                'points_per_area': float(len(points) / self.mesh.area)
            },
            'files': {
                'points': os.path.basename(points_path)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # 删除多余的print
        
        return {
            'points_path': points_path,
            'metadata_path': metadata_path
        }
    
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
        urdf_content = f'''<?xml version="1.0"?>
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
        
        # 同时保存密集点云数据
        self.save_dense_point_cloud()
        # # 保存表面点云和法向量
        # points_path = os.path.join(self.output_dir, "object_surface_points.npy")
        # normals_path = os.path.join(self.output_dir, "object_surface_normals.npy")
        # np.save(points_path, mesh_info['surface_info']['points'])
        # np.save(normals_path, mesh_info['surface_info']['normals'])
        
        # 删除多余的print
        
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
    
    def verify_coordinate_systems(self):
        """验证URDF坐标系和点云坐标系的一致性
        
        Returns:
            tuple: (是否一致的布尔值, 质心差异值)
        """
        # 1. 获取URDF中的原点（mesh的质心）
        urdf_origin = self.mesh.center_mass
        
        # 2. 获取采样点云
        points = self.get_dense_point_cloud()
        
        # 3. 计算点云的质心
        cloud_center = np.mean(points, axis=0)
        
        # 4. 计算差异
        diff = np.linalg.norm(urdf_origin - cloud_center)
        
        return diff < 1e-6, diff  # 允许1e-6的误差

def main():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_models_dir = os.path.join(current_dir, "../models")
    
    # 设置体素大小（2mm）
    voxel_size = 0.002
    
    # 记录处理结果
    results = {
        'success': [],
        'failed': []
    }
    
    # 遍历000-087的所有目录
    total_objects = 88
    for i in range(total_objects):
        model_id = f"{i:03d}"
        model_dir = os.path.join(base_models_dir, model_id)
        
        print(f"Processing [{model_id}] ({i+1}/{total_objects})")
        
        if not os.path.exists(model_dir):
            results['failed'].append({'id': model_id, 'reason': 'Directory not found'})
            continue
            
        try:
            generator = URDFGenerator(
                model_dir=model_dir,
                output_dir=model_dir,
                voxel_size=voxel_size
            )
            
            urdf_file = generator.generate_urdf()
            results['success'].append({
                'id': model_id,
                'urdf': urdf_file
            })
            
        except Exception as e:
            results['failed'].append({
                'id': model_id,
                'reason': str(e)
            })
    
    # 保存处理结果
    result_file = os.path.join(base_models_dir, 'generation_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

def check_coordinate_systems(model_dir):
    generator = URDFGenerator(
        model_dir=model_dir,
        output_dir=model_dir,
        voxel_size=0.002
    )
    # 首先加载mesh
    generator._analyze_mesh(generator.obj_file)
    # 然后验证坐标系
    is_consistent, diff = generator.verify_coordinate_systems()
    print(f'坐标系一致性检查结果：')
    print(f'是否一致：{is_consistent}')
    print(f'质心差异：{diff} 米')

if __name__ == "__main__":
    main()
    check_coordinate_systems("../models/000")