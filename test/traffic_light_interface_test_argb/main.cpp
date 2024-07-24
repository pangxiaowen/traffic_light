#include "traffic_light_interface.h"
#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

cv::Mat converBGR2NV12(cv::Mat bgrImage)
{
    // BGR2NV12
    cv::Mat yuvImage(bgrImage.rows * 3 / 2, bgrImage.cols, CV_8UC1, cv::Scalar(0)); // Y分量
    cv::Mat uvImage(bgrImage.rows * 3 / 2, bgrImage.cols, CV_8UC1, cv::Scalar(0));  // UV分量 (NV12)
    cv::cvtColor(bgrImage, yuvImage, cv::COLOR_BGR2YUV_I420);

    memcpy(uvImage.data, yuvImage.data, bgrImage.cols * bgrImage.rows);

    int yLen = bgrImage.cols * bgrImage.rows;
    int uvLen = bgrImage.cols * bgrImage.rows / 4;
    // 将UV分量（U和V）从I420格式提取并排列为NV12格式
    for (int el = 0; el < uvLen; el++)
    {
        uvImage.data[yLen + 2 * el] = yuvImage.data[yLen + el];
        uvImage.data[yLen + 2 * el + 1] = yuvImage.data[yLen + el + uvLen];
    }

    return uvImage;
}

int main(int argc, char **argv)
{

    perception::interface::TrafficLightInterfaceParams params;
    params.config_path = "/home/pxw/project/traffic_light/configs/traffic_light_config.yaml";

    perception::interface::TrafficLightInterface traffic_light;
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
    perception::interface::TrafficLightInfo tl_1;
    tl_1.tl_3d_bbox.push_back({6120.224311316056, 8926.692219366063, 30.657984679563395});
    tl_1.tl_width = 0.655106;
    tl_1.tl_height = 1.61742;
    tl_1.type = 0;

    perception::interface::TrafficLightInfo tl_2;
    tl_2.tl_3d_bbox.push_back({6122.269526667129, 8926.921721728417, 30.640411978580715});
    tl_2.tl_width = 0.822632;
    tl_2.tl_height = 1.61742;
    tl_2.type = 0;

    // 自车姿态
    perception::interface::VehicleInfo vehicle_info{6121.2841387552035, 8901.53482125327, 26.670000000000208, 1.6885217519016429};

    for (auto it : file_names)
    {
        cv::Mat bgrImage = cv::imread(it);
        // if (bgrImage.cols != 3840 || bgrImage.rows != 2160)
        //     continue;

        // BGR2ARGB
        cv::Mat argbImage;
        cv::cvtColor(bgrImage, argbImage, cv::COLOR_BGR2RGBA);

        // Host2Device
        cudaMemcpy(image_data, argbImage.ptr(), bgrImage.cols * bgrImage.rows * 4, cudaMemcpyHostToDevice);

        perception::interface::TrafficLightInterfaceInput input;
        input.image_data = image_data;
        input.width = bgrImage.cols;
        input.height = bgrImage.rows;
        input.vehicle_info = vehicle_info;
        input.traffic_infos.push_back(tl_1);
        input.traffic_infos.push_back(tl_2);

        perception::interface::TrafficLightInterfaceOuput ouput;

        auto start = std::chrono::system_clock::now();

        traffic_light.process(input, ouput);

        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        for (auto it : ouput.traffic_infos)
        {
            auto res = it;
            cv::Rect detect_bbox{it.x, it.y, it.width, it.height};

            std::cout << "ID: " << it.id
                      << " Color: " << static_cast<int>(it.color)
                      << " Detect Rect: " << detect_bbox << std::endl;

            cv::Scalar color;
            switch (it.color)
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

            cv::rectangle(bgrImage, detect_bbox, color, 2);

            std::stringstream ss;
            ss << it.id << " " << it.type;
            cv::putText(bgrImage, ss.str(), cv::Point(detect_bbox.x, detect_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        videoWriter.write(bgrImage);
    }

    videoWriter.release();
    traffic_light.release();

    return 0;
}