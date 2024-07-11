#include "traffic_light_detector.h"
#include "trt_preprocess.hpp"
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/opencv.hpp>
#include <nppi.h>

constexpr int MAX_IMAGE_SIZE = 3840 * 2160 * 3;

namespace perception
{
    namespace camera
    {
        struct Object
        {
            float conf; // bbox_conf * cls_conf
            int class_id;
            base::BBox2D<int> bbox;
        };

        static std::map<int, base::TLColor> g_class2color = {
            {0, base::TLColor::TL_RED},
            {1, base::TLColor::TL_YELLOW},
            {2, base::TLColor::TL_GREEN},
            {3, base::TLColor::TL_BLACK}};

        bool TrafficLightDetection::init(const TrafficLightDetectionParameter &params)
        {
            // init params
            m_params = params;
            m_class_bbox_thresh = m_params.class_bbox_thresh;
            m_nms_thresh = m_params.nms_thresh;

            // load trt engine
            m_trt_engine = TensorRT::load(m_params.model_path);
            m_trt_engine->print();

            // 分配模型的输入输出内存
            create_binding_memory();

            // init cuda stream
            checkRuntime(cudaStreamCreate(&m_cuda_stream));

            // 分配用于图像处理的内存
            cudaMallocManaged(&m_nv12_device, MAX_IMAGE_SIZE * sizeof(uint8_t));
            cudaMallocManaged(&m_bgr_crop_device, MAX_IMAGE_SIZE * sizeof(uint8_t));

            return true;
        }

        void TrafficLightDetection::process(CameraFrame *frame)
        {
            // 预处理
            preprocess(frame);

            // 推理
            inference();

            // 后处理
            postprocess(frame);
        }

        bool TrafficLightDetection::release()
        {
            for (size_t i = 0; i < m_input_bindings.size(); ++i)
            {
                cudaFree(m_input_bindings[i]);
            }

            for (size_t i = 0; i < m_output_bindings.size(); ++i)
            {
                cudaFree(m_output_bindings[i]);
            }

            cudaFree(m_nv12_device);
            cudaFree(m_bgr_crop_device);
            m_trt_engine.reset();
            return true;
        }

        void TrafficLightDetection::create_binding_memory()
        {
            for (int ibinding = 0; ibinding < m_trt_engine->num_bindings(); ++ibinding)
            {
                auto shape = m_trt_engine->static_dims(ibinding);
                size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                void *pdata = nullptr;

                // zero copy
                checkRuntime(cudaMallocManaged(&pdata, volumn * sizeof(float), cudaMemAttachGlobal));
                cudaMemset(pdata, 0, volumn * sizeof(float));

                if (m_trt_engine->is_input(ibinding))
                {
                    m_input_bindings.push_back(pdata);
                    m_input_bindshape.push_back(shape);
                }
                else
                {
                    m_output_bindings.push_back(pdata);
                    m_output_bindshape.push_back(shape);
                }
            }
        }

        void TrafficLightDetection::preprocess(CameraFrame *frame)
        {
            // frame->data_provider --> NV12 Format
            cudaMemcpy(m_nv12_device, frame->data_provider, frame->width * frame->height * 3 / 2, cudaMemcpyHostToDevice);

            int src_width = frame->width;
            int src_height = frame->height;
            base::RectI roi = frame->detection_roi;

            Npp8u *pSrc[2];
            pSrc[0] = static_cast<uint8_t *>(m_nv12_device) + roi.y * src_width + roi.x;
            pSrc[1] = static_cast<uint8_t *>(m_nv12_device) + src_width * src_height + (roi.y * src_width / 2 + roi.x);

            Npp8u *pDst;
            pDst = static_cast<uint8_t *>(m_bgr_crop_device);

            NppiSize oSizeROI;
            oSizeROI.width = roi.width;
            oSizeROI.height = roi.height;

            // NV12 --> BGR and Crop
            NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, src_width, pDst, roi.width * 3, oSizeROI);

            // Debug NV12 --> BGR and Crop
            // std::cout << status << std::endl;
            // cudaDeviceSynchronize();
            // cv::Mat roi_image(roi.height, roi.width, CV_8UC3, m_bgr_crop_device);
            // cv::imwrite("roi.png", roi_image);

            // Resize & packed2plane & div 255
            int dist_width = m_input_bindshape[0][2];
            int dist_height = m_input_bindshape[0][3];
            preprocess::resize_bilinear_gpu(static_cast<float *>(m_input_bindings[0]), static_cast<uint8_t *>(m_bgr_crop_device),
                                            dist_width, dist_height, roi.width, roi.height,
                                            preprocess::tactics::GPU_BILINEAR_CENTER);
            cudaDeviceSynchronize();

            // Debug Resize & packed2plane & div 255
            // cv::Mat input_image(dist_height, dist_width, CV_32FC1, m_input_bindings[0]);
            // input_image = input_image * 255;
            // cv::imwrite("input_image.png", input_image);
            // exit(1);
        }

        void TrafficLightDetection::inference()
        {
            m_trt_engine->forward({m_input_bindings[0], m_output_bindings[0], m_output_bindings[1], m_output_bindings[2], m_output_bindings[3]}, m_cuda_stream);
            cudaStreamSynchronize(m_cuda_stream);
        }

        void TrafficLightDetection::postprocess(CameraFrame *frame)
        {
            int img_w = frame->detection_roi.width;
            int img_h = frame->detection_roi.height;

            float scale = std::min(m_input_bindshape[0][2] / (img_w * 1.0), m_input_bindshape[0][3] / (img_h * 1.0));
            float padw = std::round(img_w * scale);
            float padh = std::round(img_h * scale);
            float dw = (m_input_bindshape[0][2] - padw) / 2.0f;
            float dh = (m_input_bindshape[0][3] - padh) / 2.0f;

            int *detection_num = static_cast<int *>(m_output_bindings[0]);
            float *detection_boxes = static_cast<float *>(m_output_bindings[1]);
            float *detection_scores = static_cast<float *>(m_output_bindings[2]);
            int *detection_classes = static_cast<int *>(m_output_bindings[3]);

            // generate_yolo_proposals
            std::vector<std::shared_ptr<Object>> proposals;
            for (size_t i = 0; i < (*detection_num); ++i)
            {
                if (detection_scores[i] > m_class_bbox_thresh)
                {
                    std::shared_ptr<Object> obj_ptr = std::make_shared<Object>();
                    obj_ptr->conf = detection_scores[i];
                    obj_ptr->bbox.xmin = (detection_boxes[i * 4 + 0] - dw) / scale + frame->detection_roi.x;
                    obj_ptr->bbox.ymin = (detection_boxes[i * 4 + 1] - dh) / scale + frame->detection_roi.y;
                    obj_ptr->bbox.xmax = (detection_boxes[i * 4 + 2] - dw) / scale + frame->detection_roi.x;
                    obj_ptr->bbox.ymax = (detection_boxes[i * 4 + 3] - dh) / scale + frame->detection_roi.y;
                    obj_ptr->class_id = detection_classes[i];
                    proposals.push_back(obj_ptr);
                }
            }

            // Sort by confidence
            std::sort(proposals.begin(), proposals.end(), [](std::shared_ptr<Object> x, std::shared_ptr<Object> y)
                      { return x->conf > y->conf; });

            // NMS
            std::vector<int> picked;
            for (auto i = 0; i < proposals.size(); ++i)
            {
                auto a = proposals[i];
                auto a_area = a->bbox.Area();
                int keep = 1;
                for (int j = 0; j < picked.size(); ++j)
                {
                    // IOU
                    auto b = proposals[picked[j]];
                    auto x_inter_lt = std::max(a->bbox.xmin, b->bbox.xmin);
                    auto y_inter_lt = std::max(a->bbox.ymin, b->bbox.ymin);
                    auto x_inter_rb = std::min(a->bbox.xmax, b->bbox.xmax);
                    auto y_inter_rb = std::min(a->bbox.ymax, b->bbox.ymax);

                    float inter_area = 0;
                    if (x_inter_lt < x_inter_rb & y_inter_lt < y_inter_rb)
                    {
                        inter_area = (x_inter_rb - x_inter_lt) * (y_inter_rb - y_inter_lt);
                    }

                    float union_area = a_area + b->bbox.Area() - inter_area;
                    if (inter_area / union_area > m_nms_thresh)
                        keep = 0;
                }
                if (keep)
                    picked.push_back(i);
            }

            for (auto i = 0; i < picked.size(); ++i)
            {
                auto obj = proposals[picked[i]];

                base::TrafficLightPtr tf_obj = std::make_shared<base::TrafficLight>();
                tf_obj->id = i;
                tf_obj->status.confidence = obj->conf;
                tf_obj->region.detection_bbox = obj->bbox;
                tf_obj->status.color = g_class2color[obj->class_id];
                frame->detected_bboxes.push_back(tf_obj);
            }
        }
    }
}