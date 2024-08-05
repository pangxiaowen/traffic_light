#include "traffic_light_postprocessor.h"
#include <iostream>

namespace perception
{
    namespace camera
    {
        static std::map<int, base::TLColor> g_index2color = {
            {1, base::TLColor::TL_RED},
            {2, base::TLColor::TL_YELLOW},
            {3, base::TLColor::TL_GREEN},
            {4, base::TLColor::TL_BLACK},
        };

        bool TrafficLightPostProcess::init(const TrafficLightPostProcessParameter &params)
        {
            m_params = params;
            m_bytetracker = std::make_shared<BYTETracker>(m_params.track_frame_rate, m_params.track_buffer);
            m_select.Init(6, 6);
            return true;
        }

        void TrafficLightPostProcess::process(CameraFrame *frame)
        {
            // 检测框为空则跳过后处理
            if (frame->detected_bboxes.empty())
                return;
            // bytetrack
            track(frame);
            // 根据track id进行过滤, 连续检测到多次，再进行后续处理
            filter_trafficLights(frame);
            // 修正信号灯结果
            revise_trafficLights(frame);
            // 对过滤后的框, 和project bbox 进行匹配
            select_trafficLights(frame);
        }

        bool TrafficLightPostProcess::release()
        {
            return true;
        }

        void TrafficLightPostProcess::track(CameraFrame *frame)
        {
            std::vector<bytetrackObject> byte_objs;
            for (const auto &it : frame->detected_bboxes)
            {
                auto bbox = it->region.detection_bbox;

                bytetrackObject obj;
                obj.prob = it->status.confidence;
                obj.label = static_cast<int>(it->status.color);
                obj.rect = {bbox.x, bbox.y, bbox.width, bbox.height};
                byte_objs.push_back(obj);
            }

            auto byte_res = m_bytetracker->update(byte_objs);

            for (const auto &it : byte_res)
            {
                base::TrafficLightPtr obj = std::make_shared<base::TrafficLight>();
                obj->status.track_id = it.track_id;
                obj->status.confidence = it.obj.prob;
                obj->status.color = g_index2color[it.obj.label];
                obj->region.detection_bbox.x = it.obj.rect.x;
                obj->region.detection_bbox.y = it.obj.rect.y;
                obj->region.detection_bbox.width = it.obj.rect.width;
                obj->region.detection_bbox.height = it.obj.rect.height;
                frame->track_detected_bboxes.push_back(obj);
            }
        }

        void TrafficLightPostProcess::filter_trafficLights(CameraFrame *frame)
        {
            if (frame->track_detected_bboxes.empty())
                return;

            std::vector<base::TrafficLightPtr> filter_bbox;
            for (auto it : frame->track_detected_bboxes)
            {
                auto element = m_track_id_cache.find(it->status.track_id);
                if (element == m_track_id_cache.end())
                {
                    m_track_id_cache.insert({it->status.track_id, 1});
                }
                else
                {
                    element->second = element->second + 1;
                    if (element->second > m_params.min_number_of_track)
                        filter_bbox.push_back(it);
                }
            }

            frame->track_detected_bboxes = filter_bbox;
        }

        void TrafficLightPostProcess::select_trafficLights(CameraFrame *frame)
        {
            if (frame->track_detected_bboxes.empty() || frame->traffic_lights.empty())
                return;

            m_select.SelectTrafficLights(frame->track_detected_bboxes, &frame->traffic_lights);
        }

        void TrafficLightPostProcess::revise_trafficLights(CameraFrame *frame)
        {
            if (frame->track_detected_bboxes.empty())
                return;

            for (auto &it : frame->track_detected_bboxes)
            {
                auto element = m_histroy_color.find(it->status.track_id);
                if (element == m_histroy_color.end())
                {
                    m_histroy_color.insert({it->status.track_id, it->status.color});
                }
                else
                {
                    switch (it->status.color)
                    {
                    case base::TLColor::TL_RED:
                    case base::TLColor::TL_GREEN:
                    case base::TLColor::TL_YELLOW:
                        // 若上一状态为红色，当前状态为黄色，则将当前状态修改为红色
                        if (element->second == base::TLColor::TL_RED && it->status.color == base::TLColor::TL_YELLOW)
                            it->status.color = base::TLColor::TL_RED;
                        // 红 绿 黄 更新灯的状态
                        element->second = it->status.color;
                        break;
                    case base::TLColor::TL_BLACK:
                        // 黑色则读取历史状态
                        it->status.color = element->second;
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }
}
