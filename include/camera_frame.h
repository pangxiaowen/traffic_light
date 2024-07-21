#pragma once

#include <memory>
#include <vector>
#include "base/traffic_light.h"

namespace perception
{
  namespace camera
  {
    struct CarPose
    {
      double x, y, z, yaw;
    };

    struct CameraFrame
    {
      // timestamp
      double timestamp = 0.0;

      // frame sequence id
      int frame_id = 0;

      // data provider
      int width, height;
      void *data_provider = nullptr; // 数据位于GPU端

      // ROI
      base::Rect<int> detection_roi;

      // detected traffic lights bbox
      std::vector<base::TrafficLightPtr> detected_bboxes;

      // track traffic lights bbox
      std::vector<base::TrafficLightPtr> track_detected_bboxes;

      // project traffic lights
      std::vector<base::TrafficLightPtr> traffic_lights;

      // Car Info
      CarPose car_pose;
    };

  } // namespace camera
} // namespace perception
