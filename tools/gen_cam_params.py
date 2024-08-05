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
    params_dir = "/home/pxw/project/tl_data/摄像头标定参数/bstcameraparam_598/"
    
    for name in camera_name:
        cam_extrinsics_path = os.path.join(params_dir,name + "_extrinsics.json")
        cam_intrinsics_path = os.path.join(params_dir,name + "_intrinsics.json")
        
        if name == "front_camera":
            src_width = 3840
        else:
            src_width = 1920
    
        with open(cam_extrinsics_path, 'r') as fh:
            cam_extrinsics = json.load(fh)

        with open(cam_intrinsics_path, 'r') as fh:
            cam_intrinsics = json.load(fh)
        
        cam_extrinsics['R'] = np.array(cam_extrinsics['R'])
        cam_extrinsics['T'] = np.array(cam_extrinsics['T']).reshape(1,3)
        cam_intrinsics['K'] = np.array(cam_intrinsics['K'])
        
        rotation_matrix = np.linalg.inv(rotation_matrix_from_angles(*cam_extrinsics['R']))
        trans_matrix = cam_extrinsics['T']
        
        cam2ego_matrix = np.eye(4)
        cam2ego_matrix[:3, :3] = rotation_matrix
        cam2ego_matrix[:3, 3] = trans_matrix
        # print("cam2ego_matrix")
        # print(cam2ego_matrix)
        
        ego2cam_matrix = np.linalg.inv(cam2ego_matrix)
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
        
        ego2image_matrix = cam_intrinsic_matrix @ ego2cam_matrix
        # print("ego2image_matrix")
        # print(ego2image_matrix)

        params = {"cam2ego_matrix":np.array(cam2ego_matrix), "ego2image_matrix":np.array(ego2image_matrix)}
        filename = name + '.json'
        with open(filename, 'w') as f:
            json.dump(params, f, cls=NumpyArrayEncoder)
        
    
