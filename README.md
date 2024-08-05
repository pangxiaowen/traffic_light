## 红绿灯识别

#### 编译
使用编译脚本即可编译该工程，so文件存放于lib目录下

```
./compile.sh
```

#### 接口文件
interface/traffic_light_interface.h

```
class TrafficLightInterface
{
public:
    TrafficLightInterface() = default;
    ~TrafficLightInterface() = default;

    bool init(const TrafficLightInterfaceParams &params);
    void process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output);
    bool release();

private:
    std::shared_ptr<TrafficLightInterfaceImpl> m_impl;
};
```

#### 模块输入&输出
```
struct Point3D
{
    double x, y, z;
};

struct TrafficLightInfo
{
    // 由红绿灯模块输出
    int id;                  // 目标ID
    int x, y, width, height; // 信号灯在图像上的左上角点坐标，以及width, height
    int color = 0;           // UNKNOWN = 0; RED = 1; YELLOW = 2; GREEN = 3; BLACK = 4;
    float confidence = 0.0;  // 置信度

    // 来自外部输入
    std::vector<Point3D> tl_3d_bbox; // UTM 坐标系 x, y, z, 中心点坐标
    double tl_width, tl_height;      // 信号灯的长宽
    int type;                        // STRAIGHT = 0;TURN_LEFT = 1;TURN_RIGHT = 2;STRAIGHT_TURN_LEFT = 3;STRAIGHT_TURN_RIGHT =4;CIRCULAR = 5;PEDESTRIAN = 6;CYCLIST = 7;UNKNOWN = 8;
};

struct VehicleInfo
{
    double time_stamp;   // 时间戳
    double x, y, z, yaw; // UTM 坐标系
};

struct TrafficLightInterfaceParams
{
    std::string config_path; // 配置文件路径 必填
};

struct TrafficLightInterfaceInput
{
    int width;                                   // 图像宽  3840
    int height;                                  // 图像高  2160
    void *image_data;                            // nv12 图像指针
    VehicleInfo vehicle_info;                    // 车辆位姿信息
    std::vector<TrafficLightInfo> traffic_infos; // 多个红绿灯信息
};

struct TrafficLightInterfaceOuput
{
    std::vector<TrafficLightInfo> traffic_infos; // 多个红绿灯信息
};

```

#### 测试用例
```
.
├── nppi_nv12bgr                                  # nppi 测试用例
├── traffic_light_interface_test_argb_json        # 接口测试用例
└── traffic_light_test                            # traffic_light 模块测试用例

```

* data/traffic_light.json 测试路线上所有红绿灯的位置信息
* data/can_bus_dump.json  车辆的can数据，其中yaw部分需要处理一下， yaw的坐标系北东地，需要转为东北天使用，具体参考测试用例


#### 重新量化模型
* model下面有两个模型可以进行量化： df_tl.onnx s2tld_epoch100.onnx
* 量化方法:
    1. ./build_trt.sh  df_tl.onnx xxxx 
    2. 将在model/build/engines/ 目录下生成对应的模型文件  xxxx.engine

* 若更换新的硬件设备，尽量重新量化模型。


#### 相机参数生成
* tools/gen_cam_params.py
* 修改参数目录
```
params_dir = "/home/pxw/project/tl_data/摄像头标定参数/bstcameraparam_598/"
```
* 运行 python3 tools/gen_cam_params.py
* 该目录下将会生成6个相机的参数文件
```
-rw-rw-r--  1 pxw pxw  592 Aug  5 17:19 fl_camera.json
-rw-rw-r--  1 pxw pxw  601 Aug  5 17:19 fr_camera.json
-rw-rw-r--  1 pxw pxw  601 Aug  5 17:19 front_camera.json
-rw-rw-r--  1 pxw pxw  598 Aug  5 17:19 rear_camera.json
-rw-rw-r--  1 pxw pxw  591 Aug  5 17:19 rl_camera.json
-rw-rw-r--  1 pxw pxw  595 Aug  5 17:19 rr_camera.json
```
* 红绿灯模块仅使用front_camera.json中的cam2ego_matrix, 还会使用front_camera的内参矩阵，将其填写到配置文件的对应位置。
* E2E模块使用所有json文件中的ego2image_matrix，将其填写到配置文件的对应位置，注意相机的对应关系。