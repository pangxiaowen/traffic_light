#include "nppi.h"
#include <opencv2/opencv.hpp>
#include <string>

int main()
{
    std::string imagePath = "/home/pxw/project/traffic_light/test/nppi_nv12bgr/build/CAM_FRONT_1719753933823303.jpg";

    cv::Mat bgrImage = cv::imread(imagePath);

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

    void *device_nv12;
    cudaMallocManaged(&device_nv12, 3840 * 2160 * 3 / 2 * sizeof(uint8_t));
    cudaMemcpy(device_nv12, uvImage.data, 3840 * 2160 * 3 / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    void *device_bgr;
    cudaMallocManaged(&device_bgr, 3840 * 2160 * 3 * sizeof(uint8_t));

    // 在NV12转BGR的时候进行ROI操作， ROI区域为{1150, 450, 960. 960}
    cv::Rect roi{1150, 450, 960, 960};

    Npp8u *pSrc[2];
    pSrc[0] = static_cast<uint8_t *>(device_nv12) + roi.y * 3840 + roi.x;
    pSrc[1] = static_cast<uint8_t *>(device_nv12) + yLen + (roi.y * 3840 / 2 + roi.x);

    Npp8u *pDst;
    pDst = static_cast<uint8_t *>(device_bgr);

    NppiSize oSizeROI;
    oSizeROI.width = roi.width;
    oSizeROI.height = roi.height;
    auto status = nppiNV12ToBGR_8u_P2C3R(pSrc, 3840, pDst, 960 * 3, oSizeROI);
    std::cout << status << std::endl;

    cv::Mat nppi_bgrImage(roi.height, roi.width, CV_8UC3, device_bgr); // Y分量
    cv::imwrite("nppi_bgr.png", nppi_bgrImage);
    return 0;
}