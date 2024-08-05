#include "traffic_light_interface.h"
#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>

std::vector<perception::interface::TrafficLightInfo> load_tl_info(const std::string &path)
{
    // 创建ptree对象
    boost::property_tree::ptree json_root;
    // 读取file文件，并将根节点存储赋值给json_root
    boost::property_tree::read_json<boost::property_tree::ptree>(path, json_root);

    std::vector<perception::interface::TrafficLightInfo> tl_infos;
    for (auto it : json_root)
    {
        perception::interface::TrafficLightInfo tl_info;
        double x, y, z;
        x = it.second.get<double>("x");
        y = it.second.get<double>("y");
        z = it.second.get<double>("z");

        tl_info.tl_3d_bbox.push_back({x, y, z});
        tl_info.tl_width = it.second.get<double>("width");
        tl_info.tl_height = it.second.get<double>("length");
        tl_info.type = 0;

        tl_infos.push_back(tl_info);
    }

    return tl_infos;
}

std::vector<perception::interface::TrafficLightInfo> select_tl_by_distance(std::vector<perception::interface::TrafficLightInfo> tl_infos, perception::interface::VehicleInfo vehicle_info)
{
    std::vector<perception::interface::TrafficLightInfo> select_tl_infos;

    for (auto it : tl_infos)
    {
        // 距离当前车辆80m以内的红绿灯
        double distance = std::pow(it.tl_3d_bbox[0].x - vehicle_info.x, 2) + std::pow(it.tl_3d_bbox[0].y - vehicle_info.y, 2);
        if (distance < 100 * 100 && distance > 5 * 5)
        {
            select_tl_infos.push_back(it);
        }
    }

    return select_tl_infos;
}

std::vector<perception::interface::VehicleInfo> load_vehicle_info(const std::string &path)
{
    // 创建ptree对象
    boost::property_tree::ptree json_root;
    // 读取file文件，并将根节点存储赋值给json_root
    boost::property_tree::read_json<boost::property_tree::ptree>(path, json_root);

    std::vector<perception::interface::VehicleInfo> vehicle_infos;
    for (auto it : json_root)
    {
        perception::interface::VehicleInfo vehicle_info;
        vehicle_info.time_stamp = std::stod(it.second.get<std::string>("timestep"));

        {
            std::vector<double> position;
            boost::property_tree::ptree pChild = it.second.get_child("pos");
            for (auto pos = pChild.begin(); pos != pChild.end(); ++pos)
            {
                // 迭代循环,将元素房补vector列表中
                double v = pos->second.get_value<double>();
                position.push_back(v);
            }
            vehicle_info.x = position[0];
            vehicle_info.y = position[1];
            vehicle_info.z = 24.074000000000208;
        }

        {
            std::vector<double> rotation;
            boost::property_tree::ptree pChild = it.second.get_child("yaw");
            for (auto pos = pChild.begin(); pos != pChild.end(); ++pos)
            {
                // 迭代循环,将元素房补vector列表中
                double v = pos->second.get_value<double>();
                rotation.push_back(v);
            }

            auto heading = rotation[2];
            heading = heading - 90;
            if (heading < -180)
            {
                heading += 360;
            }
            heading = -heading;

            vehicle_info.yaw = heading;
        }

        vehicle_infos.push_back(vehicle_info);
    }
    return vehicle_infos;
}

perception::interface::VehicleInfo select_vehicle_info_by_time(double cur_time_stamp, std::vector<perception::interface::VehicleInfo> vehicle_infos)
{
    perception::interface::VehicleInfo vehicle_info;

    for (auto it : vehicle_infos)
    {
        double diff_time = std::abs(it.time_stamp - cur_time_stamp);
        if (diff_time < 20000)
        {
            vehicle_info = it;
            break;
        }
    }

    return vehicle_info;
}

int main(int argc, char **argv)
{
    // 模型配置
    perception::interface::TrafficLightInterfaceParams params;
    params.config_path = "/home/pxw/project/traffic_light/configs/traffic_light_config_df.yaml";

    // 初始化模型
    perception::interface::TrafficLightInterface traffic_light;
    traffic_light.init(params);

    // 读取文件夹下所有图像
    std::vector<std::string> file_names;
    std::string pattern = "/home/pxw/project/data/0731_data_turnRightG1/CAM_FRONT";
    cv::glob(pattern, file_names, false);

    // 红绿灯位置信息
    std::string tl_json = "/home/pxw/project/traffic_light/data/traffic_light.json";
    auto all_tl_infos = load_tl_info(tl_json);

    // 车辆位置信息
    std::string vehicle_json = "/home/pxw/project/traffic_light/data/can_bus_dump.json";
    auto all_vehicle_infos = load_vehicle_info(vehicle_json);

    // 存为视频
    cv::VideoWriter videoWriter("./front_jpg_2.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(3840, 2160));

    // 分配cuda内存
    void *image_data;
    cudaMallocManaged(&image_data, 3840 * 2160 * 4 * sizeof(uint8_t), cudaMemAttachGlobal);

    for (auto it : file_names)
    {
        cv::Mat bgrImage = cv::imread(it);

        // BGR2ARGB
        cv::Mat argbImage;
        cv::cvtColor(bgrImage, argbImage, cv::COLOR_BGR2RGBA);

        // Host2Device
        cudaMemcpy(image_data, argbImage.ptr(), bgrImage.cols * bgrImage.rows * 4, cudaMemcpyHostToDevice);

        // 根据图像的时间戳，寻找对应的定位数据
        boost::filesystem::path image_path{it};
        std::vector<std::string> result;
        boost::split(result, image_path.filename().stem().c_str(), boost::is_any_of("_"));
        double cur_time_stamp = std::stod(result[2]);
        perception::interface::VehicleInfo cur_vehicle_info = select_vehicle_info_by_time(cur_time_stamp, all_vehicle_infos);

        // 根据定位数据,选择附近的信号灯
        auto cur_tl = select_tl_by_distance(all_tl_infos, cur_vehicle_info);

        if (cur_tl.empty())
            continue;

        perception::interface::TrafficLightInterfaceInput input;
        input.image_data = image_data;
        input.width = bgrImage.cols;
        input.height = bgrImage.rows;
        input.vehicle_info = cur_vehicle_info;
        input.traffic_infos = cur_tl;

        perception::interface::TrafficLightInterfaceOuput ouput;

        auto start = std::chrono::system_clock::now();

        traffic_light.process(input, ouput);

        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        for (auto it : ouput.traffic_infos)
        {
            auto res = it;
            cv::Rect detect_bbox{res.x, res.y, res.width, res.height};

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