#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat Image = cv::imread("/home/pxw/project/traffic_light/data/cam_front_center_undistorted/000000.jpg");

    // 红绿灯位置信息
    std::vector<Eigen::Vector3d> tl_utm_positons;
    tl_utm_positons.push_back({6120.224311316056, 8926.692219366063, 30.657984679563395});
    tl_utm_positons.push_back({6122.269526667129, 8926.921721728417, 30.640411978580715});


    // 自车的位姿信息
    Eigen::Vector3d cur_utm_position = Eigen::Vector3d(6122.368579481466, 8840.882388177793, 26.72400000000021);
    auto cur_utm_quaternion = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()) *
                              Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                              Eigen::AngleAxisd(1.5564116023526404, Eigen::Vector3d::UnitZ());

    // 相机内外参数
    Eigen::Matrix4d cam2ego_matrix;
    cam2ego_matrix << -0.05529270928380491, -0.08737339761267134, 0.9946395789567097, 1.4671017894905598,
        -0.9983053628761077, 0.022937879337663676, -0.05348151809360546, -0.056925327489644635,
        -0.018142107313755842, -0.9959111319958205, -0.08849368100414383, 1.3137434155143959,
        0, 0, 0, 1;

    Eigen::Matrix4d cam_intrinsic;
    cam_intrinsic << 675.466713720245, 0, 647.387385423985, 0,
        0, 678.957501724718, 356.804618676128, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    for (auto it : tl_utm_positons)
    {
        // UTM --> EGO
        Eigen::Vector3d tl_ego_position = cur_utm_quaternion.inverse() * (it - cur_utm_position);

        // 计算四个角点在自车下的坐标，H: 1.6. W: 0.8 计算
        std::vector<Eigen::Vector4d> tl_ego_positions_4d;
        tl_ego_positions_4d.push_back({tl_ego_position.x(), tl_ego_position.y() - 0.4, tl_ego_position.z() - 0.8, 1}); // 左上
        tl_ego_positions_4d.push_back({tl_ego_position.x(), tl_ego_position.y() + 0.4, tl_ego_position.z() + 0.8, 1}); // 右下
        tl_ego_positions_4d.push_back({tl_ego_position.x(), tl_ego_position.y() - 0.4, tl_ego_position.z() + 0.8, 1}); // 左下
        tl_ego_positions_4d.push_back({tl_ego_position.x(), tl_ego_position.y() + 0.4, tl_ego_position.z() - 0.8, 1}); // 右上

        for (auto tl_ego : tl_ego_positions_4d)
        {
            // EGO --> Image
            Eigen::Vector4d image_position_4d = cam_intrinsic * cam2ego_matrix.inverse() * tl_ego;
            double x = image_position_4d.data()[0] / image_position_4d.data()[2];
            double y = image_position_4d.data()[1] / image_position_4d.data()[2];
            std::cout << x << " " << y << std::endl;

            // draw
            cv::circle(Image, {x, y}, 3, {0, 0, 255});
        }
    }
    
    cv::imwrite("test.jpg", Image);

    return 0;
}
