#pragma once

#include "base_traffic_light_detector.h"
#include "base_traffic_light_postprocess.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightParameter
        {
            TrafficLightDetectionParameter detector_params;
            TrafficLightPostProcessParameter postprocess_params;
        };

        class TrafficLight
        {
        public:
            bool init(const TrafficLightParameter &params);
            void process(CameraFrame *frame);
            bool release();

        private:
            std::shared_ptr<BaseTrafficLightDetector> m_tl_detector;
            std::shared_ptr<BaseTrafficLightPostProcess> m_tl_postprocess;
        };
    }
}