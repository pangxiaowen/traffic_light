#pragma once
#include <vector>
#include "box.h"

namespace perception
{
    namespace base
    {
        enum class TLColor
        {
            TL_UNKNOWN_COLOR = 0,
            TL_RED = 1,
            TL_YELLOW = 2,
            TL_GREEN = 3,
            TL_BLACK = 4,
            TL_TOTAL_COLOR_NUM = 5
        };

        enum class TLSignal
        {
            TL_UNKNOWN_SIGNAL = 0,
            TL_STRAIGHT = 0,
            TL_TURN_LEFT = 1,
            TL_TURN_RIGHT = 2,
            TL_STRAIGHT_LEFT = 3,
            TL_STRAIGHT_RIGHT = 4,
        };

        struct LightRegion
        {
            Rect<int> projection_bbox;   // 投影框
            bool outside_image = false; // 是否再画面内
            Rect<int> detection_bbox;    // 匹配上的检测框

            // 3d polygon
            std::vector<base::PointXYZID> points; // 三维空间的边界点
        };

        struct LightStatus
        {
            // Traffic light color status.
            TLColor color = TLColor::TL_UNKNOWN_COLOR;
            // 该信号灯是左转,直行,右转
            TLSignal signal = TLSignal::TL_UNKNOWN_SIGNAL;
            // How confidence about the detected results, between 0 and 1.
            double confidence = 0.0;
            // track id
            int track_id;
            // Duration of the traffic light since detected.
            double tracking_time = 0.0;
        };

        struct TrafficLight
        {
            TrafficLight() = default;

            std::string id;
            LightRegion region; // Light region.
            LightStatus status; // Light Status.
        };

        typedef std::shared_ptr<TrafficLight> TrafficLightPtr;
        typedef std::vector<TrafficLightPtr> TrafficLightPtrs;
    }
}