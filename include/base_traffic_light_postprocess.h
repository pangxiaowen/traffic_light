#pragma once

#include "camera_frame.h"

namespace perception
{
    namespace camera
    {
        struct TrafficLightPostProcessParameter
        {
            int track_frame_rate = 30;
            int track_buffer = 15;
            int min_number_of_track = 5;
        };

        class BaseTrafficLightPostProcess
        {
        public:
            BaseTrafficLightPostProcess() = default;
            virtual ~BaseTrafficLightPostProcess() = default;

            virtual bool init(const TrafficLightPostProcessParameter &params) = 0;
            virtual void process(CameraFrame *frame) = 0;
            virtual bool release() = 0;
        };
    }
}