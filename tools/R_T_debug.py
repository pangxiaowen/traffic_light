import json
import re
import math
import os
import argparse
from scipy.spatial.transform import Rotation as R
# from pyquaternion import Quaternion
import numpy as np
import cv2
# from pyproj import Transformer
# from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
# transformer = Transformer.from_crs("epsg:4326", "epsg:32650")

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

if __name__ == '__main__':

    cam_extrinsics = dict()
    
    # cam 2 ego
    cam_extrinsics_path = '/vad/data/vehicle_data/camera_calibration/CAM_FRONT_LEFT_extrinsics.json'
    with open(cam_extrinsics_path, 'r') as fh:
        cam_extrinsics = json.load(fh)
    cam_extrinsics['R'] = np.array(cam_extrinsics['R'])
    cam_extrinsics['T'] = np.array(cam_extrinsics['T']).reshape(1,3)
    # cam_extrinsics['R'] = [90, 0, 90]
    # cam_extrinsics['T'] = [1.8470115661621094, -0.0881275087594986, 1.3037421703338623]
    cam_intrinsic = np.array(
   [
    [
      1158.882568359375,
      0.0,
      963.6992797851563,
    ],
    [
      0.0,
      1158.860595703125,
      540.8660278320313,
    ],
    [
      0.00000000e+00, 
      0.00000000e+00, 
      1.00000000e+00
    ]
    ])
    
    rotation_matrix = np.linalg.inv(rotation_matrix_from_angles(*cam_extrinsics['R']))
    trans_matrix = cam_extrinsics['T']
    
    cam2ego_matrix = np.eye(4)
    cam2ego_matrix[:3, :3] = rotation_matrix
    cam2ego_matrix[:3, 3] = trans_matrix
    
    # test ego point
    ego_point= np.array([15.24456436931311, 4.49, 0.75, 1]).reshape(4,1)
    cam_point = np.linalg.inv(cam2ego_matrix) @ ego_point
    
    print("ego coordinate system:")
    print(ego_point[:3])
    
    print("cam coordinate system:")
    print(cam_point)
    # z轴归一化
    cam_point = cam_point[:3]/ cam_point[2, 0]
    #图像坐标系
    print("nomalized cam coordinate system:")
    print(cam_point)

    #像素坐标系
    pixel_point =  cam_intrinsic @cam_point
    print("pixel coordinate system:")
    print(pixel_point[:2])

    image_path = '/vad/data/hait_RT_data/CAM_FRONT_LEFT/img_6_1712282816.88690305_no_resized.jpg'
    # 读取图像
    image = cv2.imread(image_path)
    
    # 给定的像素坐标
    x = int(pixel_point[0,0])
    y = int(pixel_point[1,0])
    # 画点
    color = (0, 0, 255)  # BGR颜色代码，红色为(0, 0, 255)
    radius = 3  # 点的半径
    thickness = -1  # 填充点
    
    # 在图像上画点
    cv2.circle(image, (x, y), radius, color, thickness)
    
    # 显示图像
    cv2.imwrite('/vad/data/hait_RT_data/CAM_FRONT_LEFT/image_with_point_no_resized.jpg', image)
    print("image_with_point saved!")

