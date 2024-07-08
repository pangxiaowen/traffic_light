#pragma once

#include <memory>
#include <vector>
#include "base/traffic_light.h"

namespace perception
{
  namespace camera
  {
    struct CameraFrame
    {
      // timestamp
      double timestamp = 0.0;
      // frame sequence id
      int frame_id = 0;

      // data provider
      int width, height;
      void *data_provider = nullptr;

      // ROI
      base::Rect<int> detection_roi;

      // detected traffic lights bbox
      std::vector<base::TrafficLightPtr> detected_bboxes;

      // track traffic lights bbox
      std::vector<base::TrafficLightPtr> track_detected_bboxes;

      // project traffic lights
      std::vector<base::TrafficLightPtr> traffic_lights;

      // camera intrinsics
      // Eigen::Matrix3f camera_k_matrix = Eigen::Matrix3f::Identity();
      // // camera extrinsics
      // Eigen::Matrix4d camera_extrinsic = Eigen::Matrix4d::Identity();
    };

  } // namespace camera
} // namespace perception
