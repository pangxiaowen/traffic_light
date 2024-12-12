from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import yaml
import os

np.set_printoptions(suppress=True)

def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(np.ndarray, ndarray_representer)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def rotation_matrix_from_angles(roll, pitch, yaw):
    # Roll, pitch, and yaw are rotations around x, y, and z axes, respectively
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw) 
    R_x = np.array([[1, 0, 0],
                    [0,np.cos(roll),-np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll) ]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    r_return  = R_x.dot(R_y.dot(R_z))
    return  r_return # Remember: Z, then Y, then X

# 原图宽， 前视为3840， 其余为1920
src_width = 3840

camera_name = [
    "front_camera",
    "fr_camera",
    "fl_camera",
    "rear_camera",
    "rl_camera",
    "rr_camera",
]


if __name__ == '__main__':
    
    # 摄像头名称
    name = ""
    # 旋转
    sensor2lidar_rotation = np.array()
    # 平移
    sensor2lidar_translation = np.array()
    # 内参
    cam_intrinsics = np.array()
    
    if name == "front_camera":
        src_width = 3840
    else:
        src_width = 1920
    
    rotation_matrix = sensor2lidar_rotation
    trans_matrix = sensor2lidar_translation
    
    sensor2lidar_matrix = np.eye(4)
    sensor2lidar_matrix[:3, :3] = rotation_matrix
    sensor2lidar_matrix[:3, 3] = trans_matrix
    
    lidar2sensor_matrix = np.linalg.inv(sensor2lidar_matrix)
    # print("ego2cam_matrix")
    # print(ego2cam_matrix)
    
    scale = 640 / src_width
    cam_intrinsic = cam_intrinsics['K'].reshape(3,3)
    # print("cam_intrinsics")
    # print(cam_intrinsic)
    
    cam_intrinsic = cam_intrinsic * scale
    cam_intrinsic[2,2] = 1
    # print("rescale cam_intrinsics")
    # print(cam_intrinsic)
    
    cam_intrinsic_matrix = np.eye(4)
    cam_intrinsic_matrix[:3,:3] = cam_intrinsic
    # print(cam_intrinsic_matrix)
    
    lidarimage_matrix = cam_intrinsic_matrix @ lidar2sensor_matrix
    print("lidarimage_matrix")
    print(lidarimage_matrix)
    
