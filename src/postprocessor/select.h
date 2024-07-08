#pragma once

#include <utility>
#include <vector>

#include "base/point.h"
#include "base/traffic_light.h"
#include "common/hungarian_optimizer.h"

namespace perception
{
  namespace camera
  {

    class Select
    {
    public:
      Select() = default;

      bool Init(int rows, int cols);

      void SelectTrafficLights(const std::vector<base::TrafficLightPtr> &detect_bboxes, std::vector<base::TrafficLightPtr> *hdmap_bboxes);

      double Calc2dGaussianScore(base::Point2DI p1, base::Point2DI p2, float sigma1, float sigma2);

    private:
      common::HungarianOptimizer<float> m_munkres;
    };
  } // namespace camera
} // namespace perception
