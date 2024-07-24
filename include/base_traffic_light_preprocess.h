#pragma once

#include "camera_frame.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightPreProcessParameter
        {
            std::vector<double> camera_intrinsics; // 相机内参
            std::vector<double> camera2ego; // 相机到自车外参
        };

        class BaseTrafficLightPreProcess
        {
        public:
            BaseTrafficLightPreProcess() = default;
            virtual ~BaseTrafficLightPreProcess() = default;

            virtual bool init(const TrafficLightPreProcessParameter &params) = 0;
            virtual void process(CameraFrame *frame) = 0;
            virtual bool release() = 0;
        };
    }
}