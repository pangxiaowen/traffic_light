#pragma once

#include "camera_frame.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightPreProcessParameter
        {
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