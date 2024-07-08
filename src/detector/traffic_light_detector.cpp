#include "traffic_light_detector.h"
#include "preprocess.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/opencv.hpp>

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

            // 分别预处理内存
            cuda_preprocess_init(MAX_IMAGE_SIZE);
            return true;
        }

        void TrafficLightDetection::process(CameraFrame *frame)
        {
            // ROI
            cv::Mat src_image(frame->height, frame->width, CV_8UC3, frame->data_provider);
            cv::Rect roi_rect = {frame->detection_roi.x, frame->detection_roi.y, frame->detection_roi.width, frame->detection_roi.height};
            cv::Mat roi_image = src_image(roi_rect).clone();

            // 预处理
            cuda_preprocess(static_cast<uint8_t *>(roi_image.ptr()), roi_image.cols, roi_image.rows,
                            static_cast<float *>(m_input_bindings[0]), m_input_bindshape[0][2], m_input_bindshape[0][3],
                            m_cuda_stream);
            cudaStreamSynchronize(m_cuda_stream);

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