#include "traffic_light.h"
#include <iostream>
#include <chrono>
#include <sys/types.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Eigen>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>

std::vector<perception::base::TrafficLightPtr> load_tl_info(const std::string &path)
{
    // 创建ptree对象
    boost::property_tree::ptree json_root;
    // 读取file文件，并将根节点存储赋值给json_root
    boost::property_tree::read_json<boost::property_tree::ptree>(path, json_root);

    std::vector<perception::base::TrafficLightPtr> tl_infos;
    for (auto it : json_root)
    {
        perception::base::TrafficLightPtr tl_info = std::make_shared<perception::base::TrafficLight>();
        double x, y, z;
        x = it.second.get<double>("x");
        y = it.second.get<double>("y");
        z = it.second.get<double>("z");

        tl_info->region.points.push_back({x, y, z});
        tl_info->region.width = it.second.get<double>("width");
        tl_info->region.height = it.second.get<double>("length");
        tl_info->status.type = perception::base::TLType::STRAIGHT;

        tl_infos.push_back(tl_info);
    }

    return tl_infos;
}

std::vector<perception::base::TrafficLightPtr> select_tl_by_distance(std::vector<perception::base::TrafficLightPtr> tl_infos, perception::camera::CarPose vehicle_info)
{
    std::vector<perception::base::TrafficLightPtr> select_tl_infos;

    for (auto it = tl_infos.begin(); it != tl_infos.end();)
    {
        double distance = std::pow(it->get()->region.points[0].x - vehicle_info.x, 2) + std::pow(it->get()->region.points[0].y - vehicle_info.y, 2);

        // 距离小于100 开始进行检测
        if (distance < 100 * 100)
        {
            perception::base::TrafficLightPtr tl_info = std::make_shared<perception::base::TrafficLight>();
            tl_info->id = (*it)->id;
            tl_info->region.points = (*it)->region.points;
            tl_info->region.width = (*it)->region.width;
            tl_info->region.height = (*it)->region.height;
            tl_info->status.type = (*it)->status.type;
            select_tl_infos.push_back(tl_info);
        }

        it++;
    }

    return select_tl_infos;
}

std::vector<perception::camera::CarPose> load_vehicle_info(const std::string &path)
{
    // 创建ptree对象
    boost::property_tree::ptree json_root;
    // 读取file文件，并将根节点存储赋值给json_root
    boost::property_tree::read_json<boost::property_tree::ptree>(path, json_root);

    std::vector<perception::camera::CarPose> vehicle_infos;
    for (auto it : json_root)
    {
        perception::camera::CarPose vehicle_info;
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

            vehicle_info.yaw = heading / 180 * M_PI;
        }

        vehicle_infos.push_back(vehicle_info);
    }
    return vehicle_infos;
}

perception::camera::CarPose select_vehicle_info_by_time(double cur_time_stamp, std::vector<perception::camera::CarPose> vehicle_infos)
{
    perception::camera::CarPose vehicle_info;

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
    // 初始化红绿灯模块
    perception::camera::TrafficLightParameter params;
    params.detector_params.model_path = "/home/pxw/project/traffic_light/model/build/engines/df_tl.engine";

    params.preprocess_params.camera_intrinsics = {2425.953125, 0.0, 1920.06396484375, 0,
                                                  0.0, 2425.725341796875, 1082.137939453125, 0,
                                                  0, 0, 1, 0,
                                                  0, 0, 0, 1};

    params.preprocess_params.camera2ego = {
        -0.01696487, -0.00873035,  0.99981797,  1.84701157,
        -0.99974738,  0.01489341, -0.01683363, -0.08812751,
        -0.01474373, -0.99985097, -0.00898081,  1.30374217,
        0, 0, 0, 1};

    // wd params
    // params.preprocess_params.camera2ego = {
    //     -0.05529270928380491, -0.08737339761267134, 0.9946395789567097, 1.4671017894905598,
    //     -0.9983053628761077, 0.022937879337663676, -0.05348151809360546, -0.056925327489644635,
    //     -0.018142107313755842, -0.9959111319958205, -0.08849368100414383, 1.3137434155143959,
    //     0, 0, 0, 1};

    perception::camera::TrafficLight traffic_light;
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

    cv::Mat map = cv::Mat::zeros(3000, 3000, CV_8UC3);

    for (auto it : all_vehicle_infos)
    {
        cv::Point2d point;
        point.x = it.x - 4500;
        point.y = it.y - 7500;

        cv::circle(map, point, 1, {255, 255, 255});
    }

    for (auto it : all_tl_infos)
    {
        cv::Point2d point;
        point.x = it->region.points[0].x - 4500;
        point.y = it->region.points[0].y - 7500;

        cv::circle(map, point, 3, {0, 0, 255});
    }
    cv::imwrite("map.png", map);

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
        perception::camera::CarPose cur_vehicle_info = select_vehicle_info_by_time(cur_time_stamp, all_vehicle_infos);

        // 根据定位数据,选择附近的信号灯
        std::vector<perception::base::TrafficLightPtr> cur_tl = select_tl_by_distance(all_tl_infos, cur_vehicle_info);

        if (cur_tl.empty())
            continue;

        std::cout << "Camera Path: " << it << std::endl;
        std::cout << std::fixed << cur_vehicle_info.time_stamp << std::endl;
        std::cout << "cur_vehicle_info" << std::endl;
        std::cout << std::fixed
                  << "can time_stamp: " << cur_vehicle_info.time_stamp << "\n"
                  << "cur_vehicle_info.x " << cur_vehicle_info.x << "\n"
                  << "cur_vehicle_info.y " << cur_vehicle_info.y << "\n"
                  << "cur_vehicle_info.z " << cur_vehicle_info.z << "\n"
                  << "cur_vehicle_info.yaw " << cur_vehicle_info.yaw << std::endl;
        std::cout << "cur_tl" << std::endl;
        for (auto &it : cur_tl)
        {
            std::cout << "tl_info.x " << it->region.points[0].x << "\n"
                      << "tl_info.y " << it->region.points[0].y << "\n"
                      << "tl_info.z " << it->region.points[0].z << std::endl;
        }

        // 准备输入数据
        std::shared_ptr<perception::camera::CameraFrame> frame = std::make_shared<perception::camera::CameraFrame>();
        frame->data_provider = image_data;
        frame->width = bgrImage.cols;
        frame->height = bgrImage.rows;
        frame->car_pose = cur_vehicle_info;
        frame->traffic_lights = cur_tl;

        // 推理
        auto start = std::chrono::system_clock::now();
        traffic_light.process(frame.get());
        auto end = std::chrono::system_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "tl time: " << inference_time << " ms" << std::endl;

        std::cout << "track_detected_bboxes size: " << frame->track_detected_bboxes.size() << std::endl;
        std::cout << "detected_bboxes size: " << frame->detected_bboxes.size() << std::endl;

        // 解析输出数据
        for (auto it : frame->traffic_lights)
        {
            std::cout << "detection_roi: " << frame->detection_roi.ToStr() << std::endl;
            if (frame->detection_roi.Area() < 40 * 40)
                continue;

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
            ss << it->status.track_id << "P_" << static_cast<int>(it->status.color) << "_" << static_cast<int>(it->status.type);
            cv::putText(bgrImage, ss.str(), cv::Point(projection_bbox.x, projection_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

            ss.str("");
            ss << it->status.track_id << "D_" << static_cast<int>(it->status.color) << "_" << static_cast<int>(it->status.type);
            cv::putText(bgrImage, ss.str(), cv::Point(detection_bbox.x, detection_bbox.y - 1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }

        // 画出检测的ROI区域
        cv::rectangle(bgrImage, {frame->detection_roi.x, frame->detection_roi.y, frame->detection_roi.width, frame->detection_roi.height}, {0, 255, 0}, 2);

        cv::imwrite("test.png", bgrImage);
        videoWriter.write(bgrImage);
    }

    videoWriter.release();
    traffic_light.release();

    return 0;
}