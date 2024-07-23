#include "traffic_light.h"
#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Eigen>

int main(int argc, char **argv)
{
    // 初始化红绿灯模块
    perception::camera::TrafficLightParameter params;
    params.detector_params.model_path = "/home/pxw/project/traffic_light/model/s2tld_epoch100.trt";

    perception::camera::TrafficLight traffic_light;
    traffic_light.init(params);

    // 读取文件夹下所有图像
    std::vector<std::string> file_names;
    std::string pattern = "/home/pxw/project/tl_data/test";
    cv::glob(pattern, file_names, false);

    // 存为视频
    cv::VideoWriter videoWriter("./front_jpg_2.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(3840, 2160));

    // 分配cuda内存
    void *image_data;
    cudaMallocManaged(&image_data, 3840 * 2160 * 4 * sizeof(uint8_t), cudaMemAttachGlobal);

    // 红绿灯位置信息
    perception::base::TrafficLightPtr tl_1 = std::make_shared<perception::base::TrafficLight>();
    tl_1->region.points.push_back({6120.224311316056, 8926.692219366063, 30.657984679563395});
    tl_1->region.width = 0.655106;
    tl_1->region.height = 1.61742;
    tl_1->status.type = perception::base::TLType::STRAIGHT;

    perception::base::TrafficLightPtr tl_2 = std::make_shared<perception::base::TrafficLight>();
    tl_2->region.points.push_back({6122.269526667129, 8926.921721728417, 30.640411978580715});
    tl_2->region.width = 0.822632;
    tl_2->region.height = 1.59562;
    tl_2->status.type = perception::base::TLType::STRAIGHT;

    // 自车的位姿信息
    perception::camera::CarPose car_pose;
    car_pose = {6121.2841387552035, 8901.53482125327, 26.670000000000208, 1.6885217519016429};

    for (auto it : file_names)
    {
        cv::Mat bgrImage = cv::imread(it);

        // BGR2ARGB
        cv::Mat argbImage;
        cv::cvtColor(bgrImage, argbImage, cv::COLOR_BGR2RGBA);

        // Host2Device
        cudaMemcpy(image_data, argbImage.ptr(), bgrImage.cols * bgrImage.rows * 4, cudaMemcpyHostToDevice);

        // 准备输入数据
        perception::camera::CameraFrame frame;
        frame.data_provider = image_data;
        frame.width = bgrImage.cols;
        frame.height = bgrImage.rows;

        frame.car_pose = car_pose;
        frame.traffic_lights.push_back(tl_1);
        frame.traffic_lights.push_back(tl_2);

        // 推理
        auto start = std::chrono::system_clock::now();
        traffic_light.process(&frame);
        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        // 解析输出数据
        for (auto it : frame.traffic_lights)
        {
            std::cout << "detection_roi: " << frame.detection_roi.ToStr() << std::endl;
            std::cout << "projection_bbox: " << it->region.projection_bbox.ToStr() << std::endl;
            std::cout << "detection_bbox: " << it->region.detection_bbox.ToStr() << std::endl;
            std::cout << "Color: " << static_cast<int>(it->status.color) << std::endl;
            std::cout << "Type: " << static_cast<int>(it->status.type) << std::endl;

            cv::Scalar color;
            switch (static_cast<int>(it->status.color))
            {
            case 1:
                color = cv::Scalar(0, 0, 255);
                break;
            case 2:
                color = cv::Scalar(0, 255, 255);
                break;
            case 3:
                color = cv::Scalar(0, 255, 0);
                break;
            default:
                color = cv::Scalar(0, 0, 0);
                break;
            }

            // 画框
            cv::Rect projection_bbox = {it->region.projection_bbox.x, it->region.projection_bbox.y, it->region.projection_bbox.width, it->region.projection_bbox.height};
            cv::Rect detection_bbox = {it->region.detection_bbox.x, it->region.detection_bbox.y, it->region.detection_bbox.width, it->region.detection_bbox.height};
            cv::rectangle(bgrImage, projection_bbox, color, 2);
            cv::rectangle(bgrImage, detection_bbox, color, 2);

            // 写上颜色和转向信息
            std::stringstream ss;
            ss << it->status.track_id << "_" << static_cast<int>(it->status.color) << "_" << static_cast<int>(it->status.type);
            cv::putText(bgrImage, ss.str(), cv::Point(projection_bbox.x, projection_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::putText(bgrImage, ss.str(), cv::Point(detection_bbox.x, detection_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        // 画出检测的ROI区域
        cv::rectangle(bgrImage, {frame.detection_roi.x, frame.detection_roi.y, frame.detection_roi.width, frame.detection_roi.height}, {0, 255, 0}, 2);

        cv::imwrite("test.png", bgrImage);
        videoWriter.write(bgrImage);
    }

    videoWriter.release();
    traffic_light.release();

    return 0;
}