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
        
    def _create_vhacd_collision_mesh(self, input_obj_path, output_obj_path):
        """
        使用PyBullet的VHACD算法创建高质量的碰撞网格
        Args:
            input_obj_path: 输入OBJ文件路径
            output_obj_path: 输出VHACD碰撞体OBJ文件路径
        Returns:
            bool: 是否成功生成VHACD碰撞体
        """
        print(f"正在使用VHACD生成碰撞体: {os.path.basename(input_obj_path)} -> {os.path.basename(output_obj_path)}")
        
        # 设置VHACD参数
        vhacd_params = {
            "resolution": 2000000,         # 提高体素分辨率（原值：1000000）
            "depth": 32,                   # 增加最大递归深度（原值：20）
            "concavity": 0.0005,           # 降低最大允许凹度（新增参数，更精确）
            "planeDownsampling": 4,        # 保持不变
            "convexhullDownsampling": 4,   # 保持不变
            "alpha": 0.04,                 # 略微降低（原值：0.05）
            "beta": 0.04,                  # 略微降低（原值：0.05）
            "gamma": 0.00025,              # 降低最小体积阈值（原值：0.00125）
            "pca": 0,                      # 保持不变
            "mode": 0,                     # 保持不变
            "maxNumVerticesPerCH": 128,    # 增加每个凸包最大顶点数（原值：64）
            "minVolumePerCH": 0.00001,     # 降低每个凸包最小体积（原值：0.0001）
            "convexhullApproximation": 0   # 禁用凸包近似（新增参数，更精确）
        }
        try:
            # 确保PyBullet已连接
            if not p.isConnected():
                client_id = p.connect(p.DIRECT)
                disconnect_after = True
            else:
                disconnect_after = False
                
            # 执行VHACD算法
            log_file = os.path.join(self.output_dir, "vhacd_log.txt")
            p.vhacd(
                input_obj_path,
                output_obj_path,
                log_file,  # fileNameLogging参数
                **vhacd_params
            )
            
            # 如果需要，断开连接
            if disconnect_after:
                p.disconnect(client_id)
                
            # 验证输出文件是否存在
            if os.path.exists(output_obj_path):
                print(f"VHACD碰撞体生成成功: {os.path.basename(output_obj_path)}")
                return True
            else:
                print(f"警告: VHACD碰撞体生成失败，输出文件不存在: {output_obj_path}")
                return False
                
        except Exception as e:
            print(f"VHACD碰撞体生成失败: {e}")
            return False

    def save_dense_point_cloud(self, prefix='dense', save_ply=False, color=None):
        """保存密集点云数据
        
        Args:
            prefix: 文件名前缀
            save_ply: 是否保存为PLY格式（包含颜色信息）
            color: 点云颜色，可以是RGB元组(r,g,b)或None（使用默认颜色）
            
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
        ply_path = os.path.join(self.output_dir, f"{prefix}_points.ply")
        
        # 保存点云数据（NPY格式）
        np.save(points_path, points)
        
        # 如果需要保存为PLY格式（带颜色信息）
        result_files = {
            'points_path': points_path,
            'metadata_path': metadata_path
        }
        
        if save_ply:
            # 创建带颜色的点云
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 设置颜色
            num_points = len(points)
            if color is None:
                # 默认使用物体的颜色（如果有）或浅灰色
                if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'to_color'):
                    # 从mesh获取颜色
                    try:
                        mesh_color = self.mesh.visual.to_color()
                        # 尝试不同的属性获取颜色
                        if hasattr(mesh_color, 'rgba'):
                            colors = np.tile(mesh_color.rgba[:3], (num_points, 1))
                        elif hasattr(mesh_color, 'vertex_colors') and mesh_color.vertex_colors.shape[0] > 0:
                            # 使用第一个顶点颜色
                            colors = np.tile(mesh_color.vertex_colors[0][:3], (num_points, 1))
                        elif hasattr(mesh_color, 'face_colors') and mesh_color.face_colors.shape[0] > 0:
                            # 使用第一个面颜色
                            colors = np.tile(mesh_color.face_colors[0][:3], (num_points, 1))
                        else:
                            # 默认浅灰色
                            colors = np.tile(np.array([0.8, 0.8, 0.8]), (num_points, 1))
                    except Exception as e:
                        print(f"获取mesh颜色时出错: {e}")
                        # 默认浅灰色
                        colors = np.tile(np.array([0.8, 0.8, 0.8]), (num_points, 1))
                else:
                    # 默认浅灰色
                    colors = np.tile(np.array([0.8, 0.8, 0.8]), (num_points, 1))
            else:
                # 使用指定的颜色
                colors = np.tile(np.array(color), (num_points, 1))
                
            # 确保颜色值在0.0-1.0范围内
            colors = np.clip(colors, 0.0, 1.0)
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存为PLY格式
            o3d.io.write_point_cloud(ply_path, pcd)
            result_files['ply_path'] = ply_path
        
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
                'points': os.path.basename(points_path),
                'ply': os.path.basename(ply_path) if save_ply else None
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return result_files
    
    def generate_urdf(self, use_vhacd=True):
        """
        生成URDF文件和相关信息
        Args:
            use_vhacd: 是否使用VHACD算法生成碰撞体
        Returns:
            str: 生成的URDF文件路径
        """
        # 使用obj文件进行几何和物理分析
        mesh = trimesh.load(self.obj_file)
        mesh_info = self._analyze_mesh(self.obj_file)
        
        # 复制视觉模型并创建碰撞模型
        visual_file = os.path.join(self.output_dir, "object_visual.obj")
        collision_file = os.path.join(self.output_dir, "object_collision.obj")
        vhacd_file = os.path.join(self.output_dir, "object_vhacd.obj")
        
        # 保存视觉模型
        import shutil
        shutil.copy2(self.obj_file, visual_file)
        
        # 根据选择的方法创建碰撞模型
        if use_vhacd:
            # 使用VHACD算法生成高质量碰撞体
            vhacd_success = self._create_vhacd_collision_mesh(visual_file, vhacd_file)
            if vhacd_success:
                collision_filename = os.path.basename(vhacd_file)
                print(f"使用VHACD碰撞体: {collision_filename}")
            else:
                # VHACD失败，回退到简化碰撞体
                print("VHACD生成失败，回退到简化碰撞体")
                collision_mesh = self._create_simplified_collision_mesh(mesh)
                collision_mesh.export(collision_file)
                collision_filename = os.path.basename(collision_file)
        else:
            # 使用简化的碰撞模型
            collision_mesh = self._create_simplified_collision_mesh(mesh)
            collision_mesh.export(collision_file)
            collision_filename = os.path.basename(collision_file)
        
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
                <mesh filename="{collision_filename}" scale="0.97 0.97 0.97"/>
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
        
        # 不再保存完整的mesh_info，因为文件太大
        # 如果需要保存一些关键信息，可以保存一个简化版本
        simplified_info = {
            'basic_info': mesh_info['basic_info'],
            'inertial_info': mesh_info['inertial_info'],
            'topology_info': mesh_info['topology_info'],
            'convex_hull_info': mesh_info['convex_hull_info']
        }
        
        # 保存简化版的模型信息
        info_path = os.path.join(self.output_dir, "object_info_simplified.json")
        with open(info_path, 'w') as f:
            json.dump(simplified_info, f, indent=4)
        
        # 同时保存密集点云数据（同时保存NPY和PLY格式）
        # 如果mesh有颜色信息，使用mesh颜色；否则使用默认灰色
        mesh_color = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'to_color'):
            try:
                mesh_color_obj = mesh.visual.to_color()
                # 尝试不同的属性获取颜色
                if hasattr(mesh_color_obj, 'rgba'):
                    mesh_color = mesh_color_obj.rgba[:3]  # 只取RGB部分
                elif hasattr(mesh_color_obj, 'vertex_colors') and mesh_color_obj.vertex_colors.shape[0] > 0:
                    # 使用第一个顶点颜色
                    mesh_color = mesh_color_obj.vertex_colors[0][:3]
                elif hasattr(mesh_color_obj, 'face_colors') and mesh_color_obj.face_colors.shape[0] > 0:
                    # 使用第一个面颜色
                    mesh_color = mesh_color_obj.face_colors[0][:3]
                else:
                    # 默认浅灰色
                    mesh_color = np.array([0.8, 0.8, 0.8])
            except Exception as e:
                print(f"获取mesh颜色时出错: {e}")
                # 如果获取颜色失败，使用默认颜色
                mesh_color = np.array([0.8, 0.8, 0.8])
        
        # 保存带颜色的PLY格式点云
        self.save_dense_point_cloud(prefix='dense', save_ply=True, color=mesh_color)
        
        # 保存表面点云和法向量（也保存为PLY格式）
        if 'surface_info' in mesh_info and 'points' in mesh_info['surface_info']:
            # 从mesh_info中获取表面点云数据
            surface_points = np.array(mesh_info['surface_info']['points'])
            
            # 保存为NPY格式
            points_path = os.path.join(self.output_dir, "object_surface_points.npy")
            np.save(points_path, surface_points)
            
            # 如果有法向量信息，也保存
            if 'normals' in mesh_info['surface_info']:
                normals_path = os.path.join(self.output_dir, "object_surface_normals.npy")
                np.save(normals_path, mesh_info['surface_info']['normals'])
            
            # 保存为PLY格式（带颜色）
            try:
                import open3d as o3d
                surface_pcd = o3d.geometry.PointCloud()
                surface_pcd.points = o3d.utility.Vector3dVector(surface_points)
                
                # 设置颜色（与上面相同的逻辑）
                num_points = len(surface_points)
                if mesh_color is not None:
                    colors = np.tile(np.array(mesh_color), (num_points, 1))
                else:
                    colors = np.tile(np.array([0.8, 0.8, 0.8]), (num_points, 1))
                
                # 确保颜色值在0.0-1.0范围内
                colors = np.clip(colors, 0.0, 1.0)
                
                surface_pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # 保存为PLY格式
                surface_ply_path = os.path.join(self.output_dir, "object_surface_points.ply")
                o3d.io.write_point_cloud(surface_ply_path, surface_pcd)
            except ImportError:
                print("警告：未安装open3d库，无法保存PLY格式点云")
            except Exception as e:
                print(f"保存PLY格式表面点云时出错：{e}")
        
        return self.urdf_file

    def get_model_info(self):
        """获取模型的详细信息
        Returns:
            dict: 模型信息字典
        """
        # 首先尝试读取简化版信息文件
        simplified_info_path = os.path.join(self.output_dir, "object_info_simplified.json")
        if os.path.exists(simplified_info_path):
            with open(simplified_info_path, 'r') as f:
                return json.load(f)
        
        # 兼容旧版本，如果简化版不存在，尝试读取完整版
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
            
            # 使用VHACD生成碰撞体
            urdf_file = generator.generate_urdf(use_vhacd=True)
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

def generate_vhacd_for_model(model_id):
    """为指定ID的模型生成VHACD碰撞体"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, f"../models/{model_id}")
    
    if not os.path.exists(model_dir):
        print(f"错误：模型目录不存在 {model_dir}")
        return False
        
    print(f"为模型 {model_id} 生成VHACD碰撞体...")
    generator = URDFGenerator(
        model_dir=model_dir,
        output_dir=model_dir,
        voxel_size=0.002
    )
    
    # 使用VHACD生成碰撞体
    urdf_file = generator.generate_urdf(use_vhacd=True)
    print(f"URDF文件已更新: {urdf_file}")
    return True

if __name__ == "__main__":
    # import sys
    from tqdm import tqdm  # 正确的导入方式
    
    # 选项一：处理所有模型
    print("开始为所有模型生成VHACD碰撞体...")
    for obj_idx in tqdm(range(0, 88)):
        model_id = f"{obj_idx:03d}"
        try:
            generate_vhacd_for_model(model_id)
        except Exception as e:
            print(f"处理模型 {model_id} 时出错: {e}")
            continue
    
    # 注释掉的命令行参数处理代码
    # if len(sys.argv) > 1 and sys.argv[1] == "vhacd":
    #     # 指定模型ID生成VHACD碰撞体
    #     if len(sys.argv) > 2:
    #         model_id = sys.argv[2]
    #         generate_vhacd_for_model(model_id)
    #     else:
    #         print("请指定模型ID，例如: python URDFGenerator.py vhacd 003")
    # else:
    #     # 默认行为
    #     main()
    #     check_coordinate_systems("../models/000")