#include "parse_configs.h"
#include <iostream>

#define TRY(statement)   \
    if ((statement) < 0) \
    {                    \
        return -1;       \
    }

#define TRY_OR_ERROR(statement, str)                                        \
    if (!(statement))                                                       \
    {                                                                       \
        std::cout << "Unable to parse [" << (str) << "] from config file."; \
        return -1;                                                          \
    }

namespace perception
{
    namespace camera
    {
        namespace config
        {
            int32_t TrafficLightConfig::Parse(const std::string &name)
            {
                m_yaml_node = YAML::LoadFile(name);
                if (!m_yaml_node)
                {
                    std::cout << "Unable to parse component file " << name;
                    return -1;
                }

                TRY(ParseTLPreProcessor());
                TRY(ParseTLDetector());
                TRY(ParseTLPostProcessor());

                return 0;
            }
            
            int32_t TrafficLightConfig::ParseTLPreProcessor()
            {
                TRY_OR_ERROR(m_yaml_node["Traffic_Light_Preprocessor"], "Traffic_Light_Preprocessor");
                auto node = m_yaml_node["Traffic_Light_Preprocessor"];

                TRY_OR_ERROR(node["Camera_Intrinsic"], "Traffic_Light_Preprocessor.Camera_Intrinsic");
                for (int32_t x = 0; x < node["Camera_Intrinsic"].size(); ++x)
                {
                    for (int32_t y = 0; y < node["Camera_Intrinsic"][x].size(); ++y)
                    {
                        m_preprocess_params.camera_intrinsics.push_back(node["Camera_Intrinsic"][x][y].as<double>());
                    }
                }

                TRY_OR_ERROR(node["Camera2Ego"], "Traffic_Light_Preprocessor.Camera2Ego");
                for (int32_t x = 0; x < node["Camera2Ego"].size(); ++x)
                {
                    for (int32_t y = 0; y < node["Camera2Ego"][x].size(); ++y)
                    {
                        m_preprocess_params.camera2ego.push_back(node["Camera2Ego"][x][y].as<double>());
                    }
                }

                return 0;
            }

            int32_t TrafficLightConfig::ParseTLDetector()
            {

                TRY_OR_ERROR(m_yaml_node["Traffic_Light_Detector"], "Traffic_Light_Detector");
                auto node = m_yaml_node["Traffic_Light_Detector"];

                TRY_OR_ERROR(node["Model_Path"], "Traffic_Light_Detector.Model_Path");
                m_detector_params.model_path = node["Model_Path"].as<std::string>();

                TRY_OR_ERROR(node["Class_Bbox_Thresh"], "Traffic_Light_Detector.Class_Bbox_Thresh");
                m_detector_params.class_bbox_thresh = node["Class_Bbox_Thresh"].as<float>();

                TRY_OR_ERROR(node["NMS_Thresh"], "Traffic_Light_Detector.NMS_Thresh");
                m_detector_params.nms_thresh = node["NMS_Thresh"].as<float>();

                return 0;
            }

            int32_t TrafficLightConfig::ParseTLPostProcessor()
            {
                TRY_OR_ERROR(m_yaml_node["Traffic_Light_PostProcessor"], "Traffic_Light_PostProcessor");
                auto node = m_yaml_node["Traffic_Light_PostProcessor"];

                TRY_OR_ERROR(node["Track_Frame_Rate"], "Traffic_Light_PostProcessor.Track_Frame_Rate");
                m_postprocess_params.track_frame_rate = node["Track_Frame_Rate"].as<float>();

                TRY_OR_ERROR(node["Track_Buffer"], "Traffic_Light_PostProcessor.Track_Buffer");
                m_postprocess_params.track_buffer = node["Track_Buffer"].as<float>();

                TRY_OR_ERROR(node["Min_Number_Of_Track"], "Traffic_Light_PostProcessor.Min_Number_Of_Track");
                m_postprocess_params.min_number_of_track = node["Min_Number_Of_Track"].as<float>();

                return 0;
            }

        }
    }
}
