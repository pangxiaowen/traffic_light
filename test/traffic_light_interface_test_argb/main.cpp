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
    params.model_path = "/home/pxw/project/traffic_light/model/s2tld_epoch100.trt";

    perception::interface::TrafficLightInterface traffic_light;
    traffic_light.init(params);

    // 读取文件夹下所有图像
    std::vector<std::string> file_names;
    std::string pattern = "/home/pxw/project/traffic_light/data";
    cv::glob(pattern, file_names, false);

    // 存为视频
    cv::VideoWriter videoWriter("./front_jpg_2.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(3840, 2160));

    // 分配cuda内存
    void *image_data;
    cudaMallocManaged(&image_data, 3840 * 2160 * 4 * sizeof(uint8_t), cudaMemAttachGlobal);

    for (auto it : file_names)
    {
        cv::Mat bgrImage = cv::imread(it);
        if (bgrImage.cols != 3840 || bgrImage.rows != 2160)
            continue;

        // BGR2ARGB
        cv::Mat argbImage;
        cv::cvtColor(bgrImage, argbImage, cv::COLOR_BGR2RGBA);

        // Host2Device
        cudaMemcpy(image_data, argbImage.ptr(), bgrImage.cols * bgrImage.rows * 4, cudaMemcpyHostToDevice);

        perception::interface::TrafficLightInterfaceInput input;
        input.image_data = image_data;
        input.width = bgrImage.cols;
        input.height = bgrImage.rows;

        perception::interface::TrafficLightInterfaceOuput ouput;

        auto start = std::chrono::system_clock::now();

        traffic_light.process(input, ouput);

        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        for (auto it : ouput.traffic_infos)
        {
            auto res = it;
            cv::Rect detect_roi{it.x, it.y, it.width, it.height};

            std::cout << "ID: " << it.id
                      << " Color: " << static_cast<int>(it.color)
                      << " Detect Rect: " << detect_roi << std::endl;

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

            cv::rectangle(bgrImage, detect_roi, color, 2);
            cv::rectangle(bgrImage, {960, 270, 1920, 1080}, {0, 0, 255}, 5);

            std::stringstream ss;
            ss << it.id;
            cv::putText(bgrImage, ss.str(), cv::Point(detect_roi.x, detect_roi.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        videoWriter.write(bgrImage);
    }

    videoWriter.release();
    traffic_light.release();

    return 0;
}