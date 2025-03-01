import socket
import struct
import json
import cv2
import numpy as np
from enum import IntEnum

class DataType(IntEnum):
    """TLV协议中的数据类型"""
    METADATA = 1  # 元数据（JSON格式）
    RGB = 2      # RGB图像数据
    DEPTH = 3    # 深度图像数据

class GraspReceiver:
    def __init__(self, host='localhost', port=12345):
        """初始化接收端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f'正在连接到服务器 {host}:{port}...')
        self.client_socket.connect((host, port))
        print('已连接到服务器')
        
    def _receive_exact(self, size: int) -> bytes:
        """准确接收指定大小的数据
        
        Args:
            size: 需要接收的字节数
            
        Returns:
            bytes: 接收到的数据
        """
        data = b''
        while len(data) < size:
            chunk = self.client_socket.recv(min(size - len(data), 4096))
            if not chunk:
                raise ConnectionError('连接断开')
            data += chunk
        return data
    
    def _receive_tlv(self) -> tuple:
        """接收TLV格式的数据
        
        Returns:
            tuple: (type_id, data)
        """
        # 接收Type(1字节) + Length(4字节)
        header = self._receive_exact(5)
        type_id, length = struct.unpack('>BI', header)
        
        # 接收Value
        data = self._receive_exact(length)
        return DataType(type_id), data
        
    def receive_data(self) -> dict:
        """接收图像数据
        
        Returns:
            dict: 包含'rgb'和'depth'的字典
        """
        result = {}
        metadata = None
        
        # 接收三个TLV包：元数据、RGB图像、深度图像
        for _ in range(3):
            type_id, data = self._receive_tlv()
            
            if type_id == DataType.METADATA:
                metadata = json.loads(data.decode('utf-8'))
            elif type_id == DataType.RGB:
                if metadata is None:
                    raise ValueError('在接收元数据之前收到图像数据')
                rgb_shape = tuple(metadata['rgb_shape'])
                rgb_dtype = metadata['rgb_dtype']
                result['rgb'] = np.frombuffer(data, dtype=rgb_dtype).reshape(rgb_shape)
            elif type_id == DataType.DEPTH:
                if metadata is None:
                    raise ValueError('在接收元数据之前收到图像数据')
                depth_shape = tuple(metadata['depth_shape'])
                depth_dtype = metadata['depth_dtype']
                result['depth'] = np.frombuffer(data, dtype=depth_dtype).reshape(depth_shape)
                
        return result
        
    def run(self):
        """运行接收循环"""
        try:
            while True:
                # 接收数据
                data = self.receive_data()
                rgb = data['rgb']
                depth = data['depth']
                
                # 显示图像
                cv2.imshow('RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow('Depth', depth.astype(np.float32) / 1000)  # 转换回米显示
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except (socket.error, ConnectionError) as e:
            print(f'接收错误: {e}')
        finally:
            self.client_socket.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    receiver = GraspReceiver()
    receiver.run()
