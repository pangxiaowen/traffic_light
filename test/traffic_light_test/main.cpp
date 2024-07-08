#include "traffic_light.h"
#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    perception::camera::TrafficLightParameter params;
    params.detector_params.model_path = "/home/pxw/project/traffic_light/model/s2tld_epoch100.trt";

    perception::camera::TrafficLight traffic_light;
    traffic_light.init(params);

    // 读取文件夹下所有图像
    std::vector<std::string> file_names;
    std::string pattern = "/home/pxw/project/data/tl/front_3/front_3";
    cv::glob(pattern, file_names, false);

    // 存为视频
    cv::VideoWriter videoWriter("./front_3.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(3840, 2160));

    // 投影框
    std::vector<cv::Rect> project_bbox = {{900, 150, 24, 54}, {880, 150, 24, 53}};

    for (auto it : file_names)
    {
        cv::Mat mat = cv::imread(it);

        std::cout << it << std::endl;
        if (mat.cols != 3840 || mat.rows != 2160)
            continue;

        perception::camera::CameraFrame frame;
        frame.data_provider = mat.ptr();
        frame.width = mat.cols;
        frame.height = mat.rows;
        frame.detection_roi = perception::base::Rect<int>{1150, 450, 960, 960};

        for (auto it : project_bbox)
        {
            perception::base::TrafficLightPtr project = std::make_shared<perception::base::TrafficLight>();
            project->region.projection_bbox = perception::base::RectI{it.x, it.y, it.width, it.height};
            frame.traffic_lights.push_back(project);
        }

        auto start = std::chrono::system_clock::now();

        traffic_light.process(&frame);

        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        cv::Mat res_image = mat.clone();
        for (auto it : frame.track_detected_bboxes)
        {
            auto res = it;

            cv::Rect project_roi{res->region.projection_bbox.x, res->region.projection_bbox.y,
                                 res->region.projection_bbox.width, res->region.projection_bbox.height};
            cv::Rect detect_roi{res->region.detection_bbox.x, res->region.detection_bbox.y,
                                res->region.detection_bbox.width, res->region.detection_bbox.height};

            std::cout << "ID: " << res->status.track_id
                      << " Color: " << static_cast<int>(res->status.color)
                      << " Detect Rect: " << detect_roi
                      << " Project Rect: " << project_roi << std::endl;

            cv::Scalar color;
            if (res->status.color == perception::base::TLColor::TL_RED)
            {
                color = cv::Scalar(0, 0, 255);
            }
            else if (res->status.color == perception::base::TLColor::TL_GREEN)
            {
                color = cv::Scalar(0, 255, 0);
            }
            else if (res->status.color == perception::base::TLColor::TL_YELLOW)
            {
                color = cv::Scalar(0, 255, 255);
            }
            else
            {
                color = cv::Scalar(0, 0, 0);
            }

            cv::rectangle(res_image, project_roi, color, 2);
            cv::rectangle(res_image, detect_roi, color, 2);

            std::stringstream ss;
            ss << res->status.track_id;
            cv::putText(res_image, ss.str(), cv::Point(project_roi.x, project_roi.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::putText(res_image, ss.str(), cv::Point(detect_roi.x, detect_roi.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        videoWriter.write(res_image);
    }

    videoWriter.release();
    traffic_light.release();

    return 0;
}