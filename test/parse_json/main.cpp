#include <boost/property_tree/json_parser.hpp>
#include <string>
#include <iostream>

int main()
{
    std::string file_name = "/home/pxw/project/traffic_light/data/traffic_light.json";
    // 创建ptree对象
    boost::property_tree::ptree json_root;
    // 读取file文件，并将根节点存储赋值给json_root
    boost::property_tree::read_json<boost::property_tree::ptree>(file_name, json_root);

    std::cout << json_root.size() << std::endl;

    for (auto it : json_root)
    {
          std::cout << it.second.get<double>("x") << std::endl;
    }


    return 0;
}