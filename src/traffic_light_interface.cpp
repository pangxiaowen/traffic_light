#pragma once

#include "traffic_light_interface.h"
#include "traffic_light.h"

namespace perception
{
    namespace interface
    {
        class TrafficLightInterfaceImpl
        {
        public:
            bool init(const TrafficLightInterfaceParams &params);
            void process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output);
            bool release();

        private:
            TrafficLightInterfaceParams m_params;
            std::shared_ptr<camera::TrafficLight> m_traffic_light;
        };

        bool TrafficLightInterfaceImpl::init(const TrafficLightInterfaceParams &params)
        {
            m_params = params;

            perception::camera::TrafficLightParameter tl_params;
            tl_params.detector_params.model_path = m_params.model_path;

            m_traffic_light = std::make_shared<camera::TrafficLight>();
            m_traffic_light->init(tl_params);

            return true;
        }

        void TrafficLightInterfaceImpl::process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output)
        {
            perception::camera::CameraFrame frame;
            frame.data_provider = input.image_data;
            frame.width = input.width;
            frame.height = input.height;
            frame.detection_roi = perception::base::Rect<int>{1150, 450, 960, 960};

            m_traffic_light->process(&frame);

            for (auto it : frame.track_detected_bboxes)
            {
                TrafficLightInfo info;
                info.id = it->status.track_id;
                info.color = static_cast<int>(it->status.color);
                info.x = it->region.detection_bbox.x;
                info.y = it->region.detection_bbox.y;
                info.width = it->region.detection_bbox.width;
                info.height = it->region.detection_bbox.height;

                output.traffic_infos.push_back(info);
            }
        }

        bool TrafficLightInterfaceImpl::release()
        {
            m_traffic_light->release();
            return true;
        }

        bool TrafficLightInterface::init(const TrafficLightInterfaceParams &params)
        {
            m_impl = std::make_shared<TrafficLightInterfaceImpl>();
            return m_impl->init(params);
        }

        void TrafficLightInterface::process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output)
        {
            m_impl->process(input, output);
        }
        bool TrafficLightInterface::release()
        {
            return m_impl->release();
        }

    }
}