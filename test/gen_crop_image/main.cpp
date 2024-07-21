#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

std::vector<std::string> Stringsplit(std::string str, const char split)
{
    std::vector<std::string> split_str;
    std::istringstream iss(str);            // 输入流
    std::string token;                      // 接收缓冲区
    while (std::getline(iss, token, split)) // 以split为分隔符
    {
        split_str.push_back(token);
    }

    return split_str;
}

int main(int argc, char **argv)
{
    // 读取文件夹下所有图像
    std::vector<std::string> file_names;
    std::string pattern = "/media/pxw/PS2000/data/20240711data/front_jpg_4";
    cv::glob(pattern, file_names, false);

    std::string save_dir = "/home/pxw/project/traffic_light/data/";

    int count = 0;
    for (auto it : file_names)
    {
        count++;
        if (count % 5 != 0)
            continue;

        cv::Mat bgrImage = cv::imread(it);

        // Save Crop Image
        cv::Rect rect = {960, 270, 1920, 1080};

        auto split_str = Stringsplit(it, '/');
        std::string crop_image_path = save_dir + split_str[split_str.size() - 1];

        cv::Mat crop_image = bgrImage(rect);
        cv::imwrite(crop_image_path, crop_image);
    }

    return 0;
}