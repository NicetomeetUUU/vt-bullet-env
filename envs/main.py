import os
import numpy as np
import pybullet as p
from collector import TactileDataCollector
from utilities import ModelLoader
from visualization import PoseVisualizer
import math
import cv2
import open3d as o3d
from tqdm import tqdm


def dataset_collect():
    """采集数据集，包括：
    1. 加载物体
    2. 获取相机点云
    3. 截取物体边界框内的点云
    4. 转换到物体坐标系
    5. 保存处理后的点云
    """
    # 创建数据采集器和保存目录
    save_dir = './dataset'
    os.makedirs(save_dir, exist_ok=True)
    collector = TactileDataCollector(save_dir=save_dir, visualize_gui=False)
    
    try:
        # 遍历000-088的所有物体
        for obj_idx in tqdm(range(89)):
            obj_dir = f"{obj_idx:03d}"  # 将数字转换为3位数的字符串
            urdf_path = f"/home/iccd-simulator/code/vt-bullet-env/models/{obj_dir}/object.urdf"
            
            if not os.path.exists(urdf_path):
                print(f"跳过不存在的物体: {urdf_path}")
                continue
                
            print(f"\n处理物体 {obj_dir}")
            
            try:
                # 加载物体到环境中
                obj_id = collector.load_object(urdf_path, position=(0, 0, 0.1))
                
                # 获取相机图像和点云
                rgb, depth, seg = collector.get_camera_image()
                full_pcd = depth_to_pointcloud(depth, rgb, collector.env.camera)
                
                # 获取AABB边界框
                aabb_min, aabb_max = p.getAABB(obj_id)
                
                # 筛选边界框内的点
                points = np.asarray(full_pcd.points)
                colors = np.asarray(full_pcd.colors)
                
                # 判断点是否在AABB内
                mask = np.all((points >= aabb_min) & (points <= aabb_max), axis=1)
                filtered_points = points[mask]
                filtered_colors = colors[mask]
                
                # 创建截取后的点云
                cropped_pcd = o3d.geometry.PointCloud()
                cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                cropped_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
                
                # 获取物体的位姿
                pos, orn = p.getBasePositionAndOrientation(obj_id)
                transform_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                
                # 将点云转换到物体坐标系
                cropped_pcd.translate(-np.array(pos))  # 平移
                cropped_pcd.rotate(transform_matrix.T)  # 旋转
                
                # 保存处理后的点云
                save_path = os.path.join(save_dir, f"{obj_dir}_pointcloud.ply")
                o3d.io.write_point_cloud(save_path, cropped_pcd)
                
                # 移除物体，准备处理下一个
                collector.remove_object(obj_id)
                
            except Exception as e:
                print(f"处理物体 {obj_dir} 时出错: {e}")
                continue
                
    finally:
        # 清理资源
        del collector
    
    print("数据集采集完成！")



def depth_to_pointcloud(depth, rgb=None, camera=None):
    """将深度图转换为点云
    Args:
        depth: 深度图，范围[0,1]表示[near,far]之间的深度
        rgb: RGB图像（可选）
        camera: 相机对象
    Returns:
        o3d.geometry.PointCloud: 点云对象
    """
    # 获取相机内参和位姿
    fx, fy, cx, cy = camera.get_intrinsics()

    # 注意：PyBullet返回的深度是非线性的，需要进行逆变换
    # 参考：https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    f = camera.far
    n = camera.near
    depth = f * n / (f - depth * (f - n))
    
    print("转换后深度值范围：", np.min(depth), np.max(depth))
    
    # 确保depth是2D数组
    if len(depth.shape) > 2:
        depth = depth[:,:,0]  # 取第一个通道
    
    # 创建像素坐标
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 计算3D点（相机坐标系）
    # 注意：PyBullet的相机坐标系是右手系，z轴指向前方
    z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # 创建点云
    points = np.stack([x, y, z], axis=-1)
    valid_mask = (z > camera.near) & (z < camera.far)
    valid_points = points[valid_mask]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # 如果有RGB图像，添加颜色
    if rgb is not None:
        if len(rgb.shape) == 3:
            if rgb.shape[2] == 4:  # RGBA格式
                rgb = rgb[:,:,:3]  # 只使用RGB通道
        colors = rgb[valid_mask] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def function_test():
    

    # 创建环境
    collector = TactileDataCollector(save_dir='./grasp_data', visualize_gui=False)
    urdf_path1 = "/home/iccd-simulator/code/vt-bullet-env/models/000/object.urdf"
    urdf_path2 = "/home/iccd-simulator/code/vt-bullet-env/models/001/object.urdf"
    try:
        while True:
            print("\n可用命令:")
            print("1: 添加物体")
            print("2: 移除物体")
            print("3: 移动到抓取位姿")
            print("4: 控制夹爪")
            print("5: 采集触觉数据")
            print("6: 重置环境")
            print("q: 退出")
            
            cmd = input('\n请输入命令: ')
            
            if cmd == 'q':
                break
                
            elif cmd == '0':
                position = (0, 0, 0.1)
                # collector.load_object(object_path=urdf_path1, position=position)
                collector.load_object(object_path=urdf_path2, position=position)
                collector.get_camera_image()
                
            elif cmd == '1':
                # 添加物体
                position = (0, 0, 0.1)
                # obj_info = model_loader.load_object(position=position)
                # print(f"物体添加成功，ID: {obj_info}")
                collector.load_object(object_path=urdf_path1, position=position)
                collector.get_camera_image()
                #obj_cube = p.loadURDF("cube.urdf", basePosition=position)
                
            elif cmd == '2':
                # 移除物体
                # success = model_loader.remove_object()
                # print(f"物体删除成功：{success}")
                #p.removeBody(obj_cube)
                collector.remove_all_objects()
                
            elif cmd == '3':
                # 测试不同的抓取位姿
                poses = [
                    # 位置，姿态（四元数）
                    ((0.3, 0.5, 0.3), (0, 0, 0, 1)),  # 上方
                ]
                
                for pos, orn in poses:
                    print(f"\n移动到位置: {pos}, 姿态: {orn}")
                    collector.move_to_grasp_pose(pos, orn)
                    time.sleep(1)
                
            elif cmd == '4':
                # 控制夹爪
                gripper_width = float(input("请输入夹爪开合宽度（0.0表示完全关闭，0.14表示完全打开）："))
                collector.control_gripper(gripper_width)
                # 执行足够多的仿真步骤以确保动作完成
                
            elif cmd == '5':
                # 采集触觉数据
                rgb, depth = collector.collect_tactile_data()
                # print(f"\n触觉数据尺寸: RGB {rgb.shape}, Depth {depth.shape}")
                
            elif cmd == '6':
                # 重置环境
                collector.reset()
                print("环境已重置")

            elif cmd == '7':
                import open3d as o3d
                # 采集相机观测
                rgb, depth, seg = collector.get_camera_image()
                print(f"\n相机图像尺寸: RGB {rgb.shape}, Depth {depth.shape}")
                for _ in range(10):
                    collector.env.step_simulation()
                os.makedirs(collector.save_dir, exist_ok=True)
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(collector.save_dir, 'rgb.png'), rgb_bgr)
                
                # 保存深度图像
                # 将深度值归一化到0-255范围
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_norm = depth_norm.astype(np.uint8)
                cv2.imwrite(os.path.join(collector.save_dir, 'depth.png'), depth_norm)
                # 转换为点云并可视化
                pcd = depth_to_pointcloud(depth, rgb, collector.env.camera)
                
                # 创建坐标系
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                
                # 可视化点云和坐标系
                o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                               window_name='Point Cloud Viewer',
                                               width=1280,
                                               height=720,
                                               point_show_normal=False)
                            
            
            else:
                print("无效命令")
                
    finally:
        # 清理资源
        collector.__del__()

def camera_test():
    collector = TactileDataCollector(save_dir='./grasp_data')
    urdf_path = "/home/iccd-simulator/code/vt-bullet-env/models/000/object.urdf"
    model_loader = ModelLoader(urdf_file=urdf_path)
    position = (0, 0, 0.1)
    # obj_info = model_loader.load_object(position=position)
    # print(f"物体添加成功，ID: {obj_info}")
    obj = p.loadURDF(urdf_path, basePosition=position)
    #obj_cube = p.loadURDF("cube.urdf", basePosition=position)
    for _ in range(100):
        p.stepSimulation()
    # 采集相机观测
    rgb, depth, seg = collector.get_camera_image()
    print(f"\n相机图像尺寸: RGB {rgb.shape}, Depth {depth.shape}")
    for _ in range(10):
        collector.env.step_simulation()
    os.makedirs(collector.save_dir, exist_ok=True)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(collector.save_dir, 'rgb.png'), rgb_bgr)
    
    # 保存深度图像
    # 将深度值归一化到0-255范围
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    cv2.imwrite(os.path.join(collector.save_dir, 'depth.png'), depth_norm)
    # 转换为点云并可视化
    pcd = depth_to_pointcloud(depth, rgb, collector.env.camera)
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # 可视化点云和坐标系
    o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                    window_name='Point Cloud Viewer',
                                    width=1280,
                                    height=720,
                                    point_show_normal=False)
    o3d.io.write_point_cloud(os.path.join(collector.save_dir, 'point_cloud.pcd'), pcd)


if __name__ == '__main__':
    # camera_test()
    function_test()