#pragma once

#include <vector>
#include <string>
#include <memory>

namespace perception
{
    namespace interface
    {
        struct TrafficLightInfo
        {
            int id;
            int x;
            int y;
            int width;
            int height;
            int color; // UNKNOWN = 0; RED = 1;YELLOW = 2;GREEN = 3; BLACK = 4;
        };

        struct TrafficLightInterfaceParams
        {
            std::string model_path;
        };

        struct TrafficLightInterfaceInput
        {
            int width;
            int height;
            void *image_data;
        };

        struct TrafficLightInterfaceOuput
        {
            std::vector<TrafficLightInfo> traffic_infos;
        };

        class TrafficLightInterfaceImpl;

        class TrafficLightInterface
        {
        public:
            TrafficLightInterface() = default;
            ~TrafficLightInterface() = default;

            bool init(const TrafficLightInterfaceParams &params);
            void process(const TrafficLightInterfaceInput &input, TrafficLightInterfaceOuput &output);
            bool release();

        private:
            std::shared_ptr<TrafficLightInterfaceImpl> m_impl;
        };
    }
}