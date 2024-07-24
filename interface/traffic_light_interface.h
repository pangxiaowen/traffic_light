#pragma once

#include <vector>
#include <string>
#include <memory>

namespace perception
{
    namespace interface
    {
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

        class TrafficLightInterfaceImpl;

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
    }
}