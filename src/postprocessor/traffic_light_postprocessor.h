#pragma once
#include "camera_frame.h"
#include "BYTETracker.h"
#include "select.h"
#include "base_traffic_light_postprocess.h"

namespace perception
{
    namespace camera
    {
        class TrafficLightPostProcess : public BaseTrafficLightPostProcess
        {
        public:
            bool init(const TrafficLightPostProcessParameter &params) override;
            void process(CameraFrame *frame) override;
            bool release() override;

        private:
            void track(CameraFrame *frame);
            void filter_trafficLights(CameraFrame *frame);
            void select_trafficLights(CameraFrame *frame);
            void revise_trafficLights(CameraFrame *frame);

        private:
            TrafficLightPostProcessParameter m_params;

            Select m_select;
            std::shared_ptr<BYTETracker> m_bytetracker;
            std::unordered_map<int, int> m_track_id_cache;
            std::unordered_map<int, base::TLColor> m_histroy_color;
        };
    }
}
