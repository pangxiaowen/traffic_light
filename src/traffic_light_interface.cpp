#pragma once

#include "traffic_light_interface.h"
#include "traffic_light.h"
#include "parse_configs.h"

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

            camera::config::TrafficLightConfig config;
            config.Parse(params.config_path);

            perception::camera::TrafficLightParameter tl_params;
            tl_params.preprocess_params = config.m_preprocess_params;
            tl_params.detector_params = config.m_detector_params;
            tl_params.postprocess_params = config.m_postprocess_params;

            m_traffic_light = std::make_shared<camera::TrafficLight>();
            m_traffic_light->init(tl_params);

            return true;
        }

        void TrafficLightInterfaceImpl::process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output)
        {
            // 处理输入数据
            std::shared_ptr<perception::camera::CameraFrame> frame = std::make_shared<perception::camera::CameraFrame>();
            frame->data_provider = input.image_data;
            frame->width = input.width;
            frame->height = input.height;

            frame->car_pose.time_stamp = 0;
            frame->car_pose.x = input.vehicle_info.x;
            frame->car_pose.y = input.vehicle_info.y;
            frame->car_pose.z = input.vehicle_info.z;
            frame->car_pose.yaw = input.vehicle_info.yaw;

            for (auto it : input.traffic_infos)
            {
                auto tl_info = std::make_shared<base::TrafficLight>();

                for (auto point : it.tl_3d_bbox)
                {
                    tl_info->region.points.push_back({point.x, point.y, point.z});
                }

                tl_info->region.width = it.tl_width;
                tl_info->region.height = it.tl_height;
                tl_info->status.type = static_cast<base::TLType>(it.type);
                frame->traffic_lights.push_back(tl_info);
            }

            // 红绿灯识别
            m_traffic_light->process(frame.get());

            // 处理输出数据
            for (auto it : frame->traffic_lights)
            {
                TrafficLightInfo info;
                info.id = it->status.track_id;
                info.x = it->region.detection_bbox.x;
                info.y = it->region.detection_bbox.y;
                info.width = it->region.detection_bbox.width;
                info.height = it->region.detection_bbox.height;
                info.confidence = it->status.confidence;
                info.color = static_cast<int>(it->status.color);
                info.type = static_cast<int>(it->status.type);
                for (auto point : it->region.points)
                {
                    info.tl_3d_bbox.push_back({point.x, point.y, point.z});
                }
                info.tl_width = it->region.width;
                info.tl_height = it->region.height;
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