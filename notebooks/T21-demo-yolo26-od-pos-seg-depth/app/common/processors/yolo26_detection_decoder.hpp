/**
 * @file yolo26_detection_decoder.hpp
 * @brief YOLO26 anchor-free detection decoder.
 */
#ifndef YOLO26_DETECTION_DECODER_HPP
#define YOLO26_DETECTION_DECODER_HPP

#include <dxrt/dxrt_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common/utility/coco_labels.hpp"
#include "postprocess_utils.hpp"

// ============================================================================
// Result type
// ============================================================================
struct AnchorlessYOLOResult {
    std::vector<float> box{};
    float confidence{0.0f};
    int class_id{0};
    std::string class_name{};

    AnchorlessYOLOResult() = default;
    AnchorlessYOLOResult(std::vector<float> box_val, float conf, int cls_id,
                         const std::string& cls_name)
        : box(std::move(box_val)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }

    float iou(const AnchorlessYOLOResult& other) const {
        return postprocess_utils::compute_iou(box, other.box);
    }

    bool is_invalid(int w, int h) const {
        return box[0] < 0 || box[1] < 0 || box[2] > w || box[3] > h;
    }
};

// ============================================================================
// Postprocess class
// ============================================================================
class YOLO26PostProcess {
public:
    YOLO26PostProcess(int input_w, int input_h,
                       float score_threshold, float nms_threshold,
                       bool is_ort_configured)
        : input_width_(input_w), input_height_(input_h),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          is_ort_configured_(is_ort_configured) {}

    std::vector<AnchorlessYOLOResult> postprocess(const dxrt::TensorPtrs& outputs) {
        auto aligned = align_tensors(outputs);
        if (aligned.empty()) {
            std::ostringstream msg;
            msg << "[DXAPP] [ER] AnchorlessYOLOPostProcess - Aligned outputs are empty.\n"
                << "  Unexpected shape\n";
            msg << postprocess_utils::format_tensor_shapes(outputs);
            msg << "Please re-compile the model with the correct output configuration.\n";
            throw std::runtime_error(msg.str());
        }

        std::vector<AnchorlessYOLOResult> dets;
        if (is_ort_configured_) {
            dets = decoding_cpu_e2e(aligned);
        } else {
            dets = decoding_npu_outputs(aligned);
        }
        return apply_nms(dets);
    }

    dxrt::TensorPtrs align_tensors(const dxrt::TensorPtrs& outputs) const {
        dxrt::TensorPtrs aligned;
        if (is_ort_configured_) {
            for (const auto& o : outputs) {
                if (o->shape().size() == 3) { aligned.push_back(o); break; }
            }
            return aligned;
        }
        if (outputs.size() == 6) {
            std::vector<dxrt::TensorPtr> cls_out, reg_out;
            for (const auto& o : outputs) {
                if (o->shape().size() != 4) continue;
                if (o->shape()[1] == num_classes_) cls_out.push_back(o);
                else if (o->shape()[1] == 64) reg_out.push_back(o);
            }
            if (cls_out.size() == 3 && reg_out.size() == 3) {
                auto cmp = [](const dxrt::TensorPtr& a, const dxrt::TensorPtr& b) {
                    return a->shape()[2] > b->shape()[2];
                };
                std::sort(cls_out.begin(), cls_out.end(), cmp);
                std::sort(reg_out.begin(), reg_out.end(), cmp);
                for (size_t i = 0; i < 3; ++i) {
                    aligned.push_back(reg_out[i]);
                    aligned.push_back(cls_out[i]);
                }
                return aligned;
            }
        }
        return aligned;
    }

private:
    int input_width_;
    int input_height_;
    float score_threshold_;
    float nms_threshold_;
    enum { num_classes_ = 80 };
    bool is_ort_configured_;

    // Helper: softmax-weighted DFL distance for regression direction k at grid position sp.
    float compute_dfl_dist(const float* reg_data, int k, int num_grid, int sp) const {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int d = 0; d < 16; ++d) {
            float v = reg_data[(k * 16 + d) * num_grid + sp];
            if (v > max_val) max_val = v;
        }
        float exp_sum = 0.f, weighted_sum = 0.f;
        for (int d = 0; d < 16; ++d) {
            float e = std::exp(reg_data[(k * 16 + d) * num_grid + sp] - max_val);
            exp_sum     += e;
            weighted_sum += e * d;
        }
        return weighted_sum / exp_sum;
    }

    // Helper: find the best class at grid position sp.
    // Populates max_cls / max_conf; returns true when a class exceeds the threshold.
    bool find_best_class(const float* cls_data, int num_grid, int sp,
                         int& max_cls, float& max_conf) const {
        max_cls  = -1;
        max_conf = score_threshold_;
        for (int c = 0; c < num_classes_; ++c) {
            float conf = cls_data[c * num_grid + sp];
            if (conf > max_conf) { max_conf = conf; max_cls = c; }
        }
        return max_cls != -1;
    }

    // ---- NPU decode: per-stride grid decoding ----
    void decodeStrideGrid(const float* reg_data, const float* cls_data,
                          int H, int W, int stride,
                          std::vector<AnchorlessYOLOResult>& detections) const {
        int num_grid = H * W;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int sp = h * W + w;
                int   max_cls  = -1;
                float max_conf = 0.f;
                if (!find_best_class(cls_data, num_grid, sp, max_cls, max_conf)) continue;

                float dist[4];
                for (int k = 0; k < 4; ++k)
                    dist[k] = compute_dfl_dist(reg_data, k, num_grid, sp);

                float ax = w + 0.5f, ay = h + 0.5f;
                AnchorlessYOLOResult r;
                r.confidence = max_conf;
                r.class_id   = max_cls;
                r.class_name = dxapp::common::get_coco_class_name(max_cls);
                r.box = {(ax - dist[0]) * stride, (ay - dist[1]) * stride,
                         (ax + dist[2]) * stride, (ay + dist[3]) * stride};
                detections.push_back(std::move(r));
            }
        }
    }

    // ---- NPU decode for raw YOLO26 outputs ----
    std::vector<AnchorlessYOLOResult> decoding_npu_outputs(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorlessYOLOResult> detections;

        if (outputs.size() == 6) {
            for (size_t i = 0; i < 3; ++i) {
                const auto& reg_tensor = outputs[2 * i];
                const auto& cls_tensor = outputs[2 * i + 1];
                auto reg_data = static_cast<const float*>(reg_tensor->data());
                auto cls_data = static_cast<const float*>(cls_tensor->data());

                auto H = static_cast<int>(cls_tensor->shape()[2]);
                auto W = static_cast<int>(cls_tensor->shape()[3]);
                int stride = input_width_ / W;
                decodeStrideGrid(reg_data, cls_data, H, W, stride, detections);
            }
        }
        return detections;
    }

    // CPU output: [1, 300, 6] end-to-end layout
    std::vector<AnchorlessYOLOResult> decoding_cpu_e2e(const dxrt::TensorPtrs& outputs) const {
        std::vector<AnchorlessYOLOResult> detections;
        if (outputs.empty()) return detections;
        const auto& tensor = outputs[0];
        const float* data = static_cast<const float*>(tensor->data());
        const auto& shape = tensor->shape();
        if (shape.size() != 3) return detections;
        int N = static_cast<int>(shape[1]);
        int stride = static_cast<int>(shape[2]);
        for (int i = 0; i < N; ++i) {
            const float* det = data + i * stride;
            if (det[4] < score_threshold_) continue;
            int cls = static_cast<int>(det[5]);
            if (cls < 0 || cls >= num_classes_) continue;
            AnchorlessYOLOResult r;
            r.confidence = det[4];
            r.class_id = cls;
            r.class_name = dxapp::common::get_coco_class_name(cls);
            r.box = {det[0], det[1], det[2], det[3]};
            detections.push_back(std::move(r));
        }
        return detections;
    }

    std::vector<AnchorlessYOLOResult> apply_nms(const std::vector<AnchorlessYOLOResult>& dets) const {
        return postprocess_utils::apply_nms(dets, nms_threshold_);
    }
};

#endif  // YOLO26_DETECTION_DECODER_HPP
