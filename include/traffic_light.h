#pragma once

#include "base_traffic_light_detector.h"
#include "base_traffic_light_postprocess.h"
#include "base_traffic_light_preprocess.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightParameter
        {
            TrafficLightPreProcessParameter preprocess_params;
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
            std::shared_ptr<BaseTrafficLightPreProcess> m_tl_preprocess;
            std::shared_ptr<BaseTrafficLightDetector> m_tl_detector;
            std::shared_ptr<BaseTrafficLightPostProcess> m_tl_postprocess;
        };
    }
}