#pragma once

#include "base_traffic_light_preprocess.h"
#include "base_traffic_light_detector.h"
#include "base_traffic_light_postprocess.h"
#include <yaml-cpp/yaml.h>
#include <iostream>

namespace perception
{
    namespace camera
    {
        namespace config
        {
            class TrafficLightConfig
            {
            public:
                int32_t Parse(const std::string &name);

            public:
                TrafficLightPreProcessParameter m_preprocess_params;
                TrafficLightDetectionParameter m_detector_params;
                TrafficLightPostProcessParameter m_postprocess_params;

            private:
                int32_t ParseTLPreProcessor();
                int32_t ParseTLDetector();
                int32_t ParseTLPostProcessor();

            private:
                YAML::Node m_yaml_node;
            };
        }
    }
}
