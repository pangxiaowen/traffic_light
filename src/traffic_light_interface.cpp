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
            // 处理输入数据
            perception::camera::CameraFrame frame;
            frame.data_provider = input.image_data;
            frame.width = input.width;
            frame.height = input.height;
            frame.car_pose = {input.vehicle_info.x, input.vehicle_info.y, input.vehicle_info.z, input.vehicle_info.yaw};

            for (auto it : input.traffic_infos)
            {
                auto tl_info = std::make_shared<base::TrafficLight>();

                for (auto point : it.tl_3d_bbox)
                {
                    tl_info->region.points.push_back({point.x, point.y, point.z, 0});
                }

                tl_info->region.width = it.width;
                tl_info->region.height = it.height;
                tl_info->status.signal = static_cast<base::TLSignal>(it.turn_info);
            }

            // 红绿灯识别
            m_traffic_light->process(&frame);

            // 处理输出数据
            for (auto it : frame.traffic_lights)
            {
                TrafficLightInfo info;
                info.id = it->status.track_id;
                info.x = it->region.detection_bbox.x;
                info.y = it->region.detection_bbox.y;
                info.width = it->region.detection_bbox.width;
                info.height = it->region.detection_bbox.height;
                info.color = static_cast<int>(it->status.color);
                info.turn_info = static_cast<int>(it->status.signal);
                for (auto point : it->region.points)
                {
                    info.tl_3d_bbox.push_back({point.x, point.y, point.z});
                }
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