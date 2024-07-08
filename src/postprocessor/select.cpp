#include "select.h"
#include <iostream>

namespace perception
{
  namespace camera
  {

    bool Select::Init(int rows, int cols)
    {
      if (rows < 0 || cols < 0)
      {
        return false;
      }

      m_munkres.costs()->Reserve(rows, cols);

      return true;
    }

    double Select::Calc2dGaussianScore(base::Point2DI p1, base::Point2DI p2,
                                       float sigma1, float sigma2)
    {
      return std::exp(-0.5 * (static_cast<float>((p1.x - p2.x) * (p1.x - p2.x)) /
                                  (sigma1 * sigma1) +
                              (static_cast<float>((p1.y - p2.y) * (p1.y - p2.y)) /
                               (sigma2 * sigma2))));
    }

    void Select::SelectTrafficLights(
        const std::vector<base::TrafficLightPtr> &detect_bboxes,
        std::vector<base::TrafficLightPtr> *hdmap_bboxes)
    {
      std::vector<std::pair<size_t, size_t>> assignments;

      m_munkres.costs()->Resize(hdmap_bboxes->size(), detect_bboxes.size());

      for (size_t row = 0; row < hdmap_bboxes->size(); ++row)
      {
        auto center_hd = (*hdmap_bboxes)[row]->region.projection_bbox.Center();

        // 如果某个投影框不在画面内部，则将其与其他检测框之间的代价设置为0
        if ((*hdmap_bboxes)[row]->region.outside_image)
        {
          for (size_t col = 0; col < detect_bboxes.size(); ++col)
          {
            (*m_munkres.costs())(row, col) = 0.0;
          }
          continue;
        }

        // 计算检测框与在画面内的投影框之间的距离
        for (size_t col = 0; col < detect_bboxes.size(); ++col)
        {
          float gaussian_score = 100.0f;
          auto center_refine = detect_bboxes[col]->region.detection_bbox.Center();
          // use gaussian score as metrics of distance and width
          double distance_score = Calc2dGaussianScore(
              center_hd, center_refine, gaussian_score, gaussian_score);

          double max_score = 0.9;
          auto detect_score = detect_bboxes[col]->status.confidence;
          double detection_score =
              detect_score > max_score ? max_score : detect_score;

          double distance_weight = 0.7;
          double detection_weight = 1 - distance_weight;
          (*m_munkres.costs())(row, col) =
              static_cast<float>(detection_weight * detection_score +
                                 distance_weight * distance_score);
        }
      }

      m_munkres.Maximize(&assignments);

      // 根据代价矩阵进行匹配
      for (size_t i = 0; i < assignments.size(); ++i)
      {
        if (static_cast<size_t>(assignments[i].first) >= hdmap_bboxes->size() ||
            static_cast<size_t>(assignments[i].second >= detect_bboxes.size()))
        {
        }
        else
        {
          auto &detect_bbox = detect_bboxes[assignments[i].second];
          auto &hdmap_bbox = (*hdmap_bboxes)[assignments[i].first];

          hdmap_bbox->id = detect_bbox->id;
          // 将颜色，检测框， 追踪ID， 赋值给投影框
          hdmap_bbox->status.color = detect_bbox->status.color;
          hdmap_bbox->status.confidence = detect_bbox->status.confidence;
          hdmap_bbox->status.track_id = detect_bbox->status.track_id;
          hdmap_bbox->region.detection_bbox = detect_bbox->region.detection_bbox;
        }
      }
    }

  } // namespace camera
} // namespace perception
