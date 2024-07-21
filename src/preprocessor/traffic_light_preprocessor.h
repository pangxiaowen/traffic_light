#include "base_traffic_light_preprocess.h"
#include <eigen3/Eigen/Eigen>

namespace perception
{
    namespace camera
    {
        /*
        1. 生成投影框--> 判断投影框是否在图像内部
        2. 需要 车辆的位姿， 信号灯的位置， 相机内外参数
        */

        class TrafficLightPreProcessor : public BaseTrafficLightPreProcess
        {
        public:
            TrafficLightPreProcessor() = default;
            ~TrafficLightPreProcessor() = default;

            bool init(const TrafficLightPreProcessParameter &params) override;
            void process(CameraFrame *frame) override;
            bool release() override;

        private:
            base::Point2D<double> convert_ego2image(Eigen::Vector4d ego_point);

        private:
            Eigen::Matrix4d m_camera_intrinsics;
            Eigen::Matrix4d m_camera2ego;
            Eigen::Matrix4d m_ego2image;
        };
    }
}
