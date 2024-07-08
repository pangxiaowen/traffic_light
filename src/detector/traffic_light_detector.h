#pragma once

#include "camera_frame.h"
#include "base_traffic_light_detector.h"
#include "tensorrt/tensorrt.hpp"
#include "tensorrt/check.hpp"

namespace perception
{
    namespace camera
    {
        class TrafficLightDetection : public BaseTrafficLightDetector
        {
        public:
            bool init(const TrafficLightDetectionParameter &params) override;
            void process(CameraFrame *frame) override;
            bool release() override;

        private:
            void create_binding_memory();
            void inference();
            void postprocess(CameraFrame *frame);

        private:
            TrafficLightDetectionParameter m_params;

            // model params
            float m_class_bbox_thresh = 0.7;
            float m_nms_thresh = 0.45;

            // tensorrt
            cudaStream_t m_cuda_stream = nullptr;
            std::shared_ptr<TensorRT::Engine> m_trt_engine = nullptr;

            // only one input
            std::vector<void *> m_input_bindings;
            std::vector<std::vector<int>> m_input_bindshape;
            // only output input
            std::vector<void *> m_output_bindings;
            std::vector<std::vector<int>> m_output_bindshape;
        };
    }
}