#pragma once

#include "camera_frame.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightDetectionParameter
        {
            std::string model_path;
            float class_bbox_thresh = 0.7;
            float nms_thresh = 0.45;
        };

        class BaseTrafficLightDetector
        {
        public:
            BaseTrafficLightDetector() = default;
            virtual ~BaseTrafficLightDetector() = default;

            virtual bool init(const TrafficLightDetectionParameter &params) = 0;
            virtual void process(CameraFrame *frame) = 0;
            virtual bool release() = 0;
        };
    }
}