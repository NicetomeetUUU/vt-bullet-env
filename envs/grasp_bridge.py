import os
import cv2
import numpy as np
import socket
import struct
import threading
import json
from enum import IntEnum

class DataType(IntEnum):
    """TLV协议中的数据类型"""
    METADATA = 1  # 元数据（JSON格式）
    RGB = 2      # RGB图像数据
    DEPTH = 3    # 深度图像数据

class GraspBridge:
    def __init__(self, save_dir=None, host='localhost', port=12345):
        """Initialize the bridge between simulation and grasp detection.
        
        Args:
            save_dir: Directory to save temporary image data
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 初始化socket服务器
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f'等待客户端连接到 {host}:{port}...')
        
        # 在新线程中等待连接
        self.client_socket = None
        self.accept_thread = threading.Thread(target=self._accept_connection)
        self.accept_thread.daemon = True
        self.accept_thread.start()
        
    def _accept_connection(self):
        """等待并接受客户端连接"""
        while True:
            self.client_socket, addr = self.server_socket.accept()
            print(f'客户端已连接: {addr}')
    
    def _send_tlv(self, type_id: DataType, data: bytes):
        """发送TLV格式的数据
        
        Args:
            type_id: 数据类型
            data: 要发送的数据
        """
        if self.client_socket is None:
            return
            
        try:
            # 发送Type(1字节) + Length(4字节) + Value
            length = len(data)
            header = struct.pack('>BI', type_id, length)
            self.client_socket.sendall(header)
            self.client_socket.sendall(data)
        except (socket.error, BrokenPipeError) as e:
            print(f'发送数据错误: {e}')
            self.client_socket = None
    
    def _send_data(self, rgb: np.ndarray, depth: np.ndarray):
        """发送图像数据
        
        Args:
            rgb: RGB图像数据
            depth: 深度图像数据
        """
        if self.client_socket is None:
            return
            
        try:
            # 发送元数据
            metadata = {
                'rgb_shape': rgb.shape,
                'rgb_dtype': str(rgb.dtype),
                'depth_shape': depth.shape,
                'depth_dtype': str(depth.dtype)
            }
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            self._send_tlv(DataType.METADATA, metadata_bytes)
            
            # 发送RGB图像数据
            self._send_tlv(DataType.RGB, rgb.tobytes())
            
            # 发送深度图像数据
            self._send_tlv(DataType.DEPTH, depth.tobytes())
            
        except Exception as e:
            print(f'发送数据错误: {e}')
            self.client_socket = None
    
    def save_camera_data(self, rgb, depth):
        """Process and send camera data.
        
        Args:
            rgb: RGB image array (H, W, 3)
            depth: Depth image array (H, W)
        """
        # 处理图像数据
        rgb = (rgb * 255).astype(np.uint8)
        depth_mm = (depth * 1000).astype(np.uint16)  # 转换为毫米
        
        # 如果需要，保存到文件
        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, 'color.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.save_dir, 'depth.png'), depth_mm)
        
        # 通过socket发送
        self._send_data(rgb, depth_mm)
        
    def get_camera_intrinsics(self, camera):
        """Get camera intrinsics in the format needed by grasp detection.
        
        Args:
            camera: Camera object from simulation
        Returns:
            dict: Camera intrinsics parameters
        """
        return {
            'fx': camera.fx,
            'fy': camera.fy,
            'cx': camera.cx,
            'cy': camera.cy,
            'scale': 1000.0  # Convert meters to millimeters
        }

    def get_next_grasp_pose(self):
        # used to execute grasp detected
        pass
        
    def __del__(self):
        """清理资源"""
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
