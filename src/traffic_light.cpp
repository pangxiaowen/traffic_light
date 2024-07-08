#include "traffic_light.h"
#include "detector/traffic_light_detector.h"
#include "postprocessor/traffic_light_postprocessor.h"

namespace perception
{
    namespace camera
    {
        bool TrafficLight::init(const TrafficLightParameter &params)
        {
            m_tl_detector = std::make_shared<TrafficLightDetection>();
            m_tl_detector->init(params.detector_params);

            m_tl_postprocess = std::make_shared<TrafficLightPostProcess>();
            m_tl_postprocess->init(params.postprocess_params);

            return true;
        }

        void TrafficLight::process(CameraFrame *frame)
        {
            m_tl_detector->process(frame);
            m_tl_postprocess->process(frame);
        }

        bool TrafficLight::release()
        {
            m_tl_postprocess->release();
            m_tl_detector->release();
            return true;
        }
    }
}