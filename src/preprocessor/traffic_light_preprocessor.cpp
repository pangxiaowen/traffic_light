#include "traffic_light_preprocessor.h"
#include <iostream>

namespace perception
{
    namespace camera
    {
        bool TrafficLightPreProcessor::init(const TrafficLightPreProcessParameter &params)
        {
            // 初始化相机内参
            if (params.camera_intrinsics.size() < 4 * 4)
            {
                std::cout << "The camera intrinsic matrix is not 4 x 4" << std::endl;
                exit(1);
            }
            else
            {
                m_camera_intrinsics = Eigen::Matrix4d(params.camera_intrinsics.data()).transpose();
                std::cout << "Camera Intrinsic Matrix: \n"
                          << m_camera_intrinsics << std::endl;
            }

            // 初始化相机外参
            if (params.camera2ego.size() < 4 * 4)
            {
                std::cout << "The camera2ego matrix is not 4 x 4" << std::endl;
                exit(1);
            }
            else
            {
                m_camera2ego = Eigen::Matrix4d(params.camera2ego.data()).transpose();
                std::cout << "Camera2Ego Matrix: \n"
                          << m_camera2ego << std::endl;
            }
            m_ego2image = m_camera_intrinsics * m_camera2ego.inverse();
            return true;
        }

        base::Point2D<double> TrafficLightPreProcessor::convert_ego2image(Eigen::Vector4d ego_point)
        {
            // EGO --> Image
            Eigen::Vector4d image_point = m_ego2image * ego_point;
            double x = image_point.data()[0] / image_point.data()[2];
            double y = image_point.data()[1] / image_point.data()[2];

            return base::Point2D<double>{x, y};
        }

        void TrafficLightPreProcessor::process(CameraFrame *frame)
        {
            Eigen::Vector3d cur_position = Eigen::Vector3d(frame->car_pose.x, frame->car_pose.y, frame->car_pose.z);
            auto cur_quaternion = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                                  Eigen::AngleAxisd(frame->car_pose.yaw, Eigen::Vector3d::UnitZ());

            // 计算红绿灯在图像上的投影框
            std::vector<base::RectI> tl_projection_bboxs;
            for (auto it : frame->traffic_lights)
            {
                auto tl_utm_positions = it->region.points;
                double width_offset = it->region.width / 2;
                double height_offset = it->region.height / 2;

                for (auto utm_position : tl_utm_positions)
                {
                    // 计算红绿灯中心点在自车坐标系下的位置
                    Eigen::Vector3d tl_utm_position = {utm_position.x, utm_position.y, utm_position.z};
                    Eigen::Vector3d tl_ego_position = cur_quaternion.inverse() * (tl_utm_position - cur_position);

                    // 如果信号灯在车后方，则跳过处理
                    if (tl_ego_position[0] < 0)
                        continue;

                    // 根据中心点以及宽高计算红绿灯左上和右下角点的位置, 并转为4维，方便后续计算
                    Eigen::Vector4d ego_top_left_point{tl_ego_position.x(), tl_ego_position.y() + width_offset, tl_ego_position.z() + height_offset, 1};
                    auto image_top_left_point = convert_ego2image(ego_top_left_point); // 左上

                    Eigen::Vector4d bottom_right_point{tl_ego_position.x(), tl_ego_position.y() - width_offset, tl_ego_position.z() - height_offset, 1};
                    auto image_bottom_right_point = convert_ego2image(bottom_right_point); // 右下

                    base::RectI projection_bbox{image_top_left_point.x, image_top_left_point.y,
                                                image_bottom_right_point.x - image_top_left_point.x,
                                                image_bottom_right_point.y - image_top_left_point.y};

                    // 判断中心点是否在画面内
                    auto center_point = projection_bbox.Center();
                    if (center_point.x < 0 || center_point.y < 0 || center_point.x > frame->width || center_point.y > frame->height)
                    {
                        it->region.outside_image = true;
                        continue;
                    }

                    // 保存投影框
                    it->region.projection_bbox = projection_bbox;
                    tl_projection_bboxs.push_back(projection_bbox);
                }
            }

            // 根据投影框计算ROI
            base::RectI detection_roi;
            if (!tl_projection_bboxs.empty())
            {
                // 计算一个能覆盖所有投影框的ROI区域
                detection_roi = tl_projection_bboxs[0];
                for (auto it : tl_projection_bboxs)
                {
                    detection_roi = detection_roi | it;
                }

                // 固定ROI区域 640x640
                detection_roi.x = detection_roi.Center().x - 480;
                detection_roi.x = detection_roi.x > 0 ? detection_roi.x : 0;
                detection_roi.y = detection_roi.Center().y - 480;
                detection_roi.y = detection_roi.y > 0 ? detection_roi.y : 0;
                detection_roi.width = 960;
                detection_roi.height = 960;

                // 自定义动态ROI TODO
                // detection_roi.x = detection_roi.x - detection_roi.width * 4;
                // detection_roi.y = detection_roi.y - detection_roi.height * 4;
                // detection_roi.width = detection_roi.width * 10 > 640 ? detection_roi.width * 10 : 640;
                // detection_roi.height = detection_roi.height * 10 > 640 ? detection_roi.height * 10 : 640;
            }
            else // 如果没有投影框，则选用默认的ROI区域
            {
                // detection_roi = perception::base::Rect<int>{960, 270, 1920, 1080};
                detection_roi = perception::base::Rect<int>{0, 0, 0, 0};
            }

            // 保证检测区域在画面内
            base::RectI src_image_size = {0, 0, frame->width, frame->height};
            detection_roi = detection_roi & src_image_size;
            frame->detection_roi = detection_roi;
        }

        bool TrafficLightPreProcessor::release()
        {
            return true;
        }
    }
}
