/**
 * @file yolo26_segmentation_postprocessor.hpp
 * @brief YOLO26 instance segmentation postprocessor used by this tutorial.
 */

#ifndef YOLO26_SEGMENTATION_POSTPROCESSOR_HPP
#define YOLO26_SEGMENTATION_POSTPROCESSOR_HPP

#include <algorithm>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/yolo26_segmentation_decoder.hpp"

namespace dxapp {

class YOLO26SegPostprocessor : public IPostprocessor<InstanceSegmentationResult> {
public:
    YOLO26SegPostprocessor(int input_width = 640, int input_height = 640,
                           float score_threshold = 0.45f,
                           float nms_threshold = 0.4f,
                           bool is_ort_configured = false,
                           int num_classes = 80)
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured, num_classes) {}

    std::vector<InstanceSegmentationResult> process(
        const dxrt::TensorPtrs& outputs,
        const PreprocessContext& ctx) override {
        auto decoded = impl_.postprocess(outputs);
        std::vector<InstanceSegmentationResult> results;
        results.reserve(decoded.size());

        for (const auto& source : decoded) {
            InstanceSegmentationResult result;
            result.box = source.box;
            result.confidence = source.confidence;
            result.class_id = source.class_id;
            result.class_name = source.class_name;

            if (!source.mask.empty() &&
                source.mask_height > 0 && source.mask_width > 0) {
                cv::Mat mask_float(source.mask_height, source.mask_width, CV_32FC1,
                                   const_cast<float*>(source.mask.data()));
                mask_float.convertTo(result.mask, CV_8UC1, 255.0);
            }

            scaleBox(result.box, ctx);
            scaleMask(result.mask, ctx);
            results.push_back(std::move(result));
        }
        return results;
    }

    std::string getModelName() const override { return "YOLO26-Seg"; }

private:
    static void scaleBox(std::vector<float>& box, const PreprocessContext& ctx) {
        if (box.size() < 4) return;

        if (ctx.pad_x == 0 && ctx.pad_y == 0 &&
            ctx.scale_x > 0.0f && ctx.scale_y > 0.0f) {
            box[0] /= ctx.scale_x;
            box[1] /= ctx.scale_y;
            box[2] /= ctx.scale_x;
            box[3] /= ctx.scale_y;
        } else {
            box[0] = (box[0] - ctx.pad_x) / ctx.scale;
            box[1] = (box[1] - ctx.pad_y) / ctx.scale;
            box[2] = (box[2] - ctx.pad_x) / ctx.scale;
            box[3] = (box[3] - ctx.pad_y) / ctx.scale;
        }

        const float width = static_cast<float>(ctx.original_width);
        const float height = static_cast<float>(ctx.original_height);
        box[0] = std::clamp(box[0], 0.0f, width);
        box[1] = std::clamp(box[1], 0.0f, height);
        box[2] = std::clamp(box[2], 0.0f, width);
        box[3] = std::clamp(box[3], 0.0f, height);
    }

    static void scaleMask(cv::Mat& mask, const PreprocessContext& ctx) {
        if (mask.empty() || ctx.original_width <= 0 || ctx.original_height <= 0) {
            return;
        }

        cv::Mat cropped = mask;
        if (ctx.pad_x > 0 || ctx.pad_y > 0) {
            const int unpadded_width = mask.cols - 2 * ctx.pad_x;
            const int unpadded_height = mask.rows - 2 * ctx.pad_y;
            if (unpadded_width > 0 && unpadded_height > 0) {
                cropped = mask(cv::Rect(ctx.pad_x, ctx.pad_y,
                                        unpadded_width, unpadded_height)).clone();
            }
        }
        cv::resize(cropped, mask, cv::Size(ctx.original_width, ctx.original_height));
    }

    YOLOv8SegPostProcess impl_;
};

}  // namespace dxapp

#endif  // YOLO26_SEGMENTATION_POSTPROCESSOR_HPP
