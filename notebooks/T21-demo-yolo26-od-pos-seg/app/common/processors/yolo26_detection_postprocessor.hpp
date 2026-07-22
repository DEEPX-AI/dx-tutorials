/**
 * @file yolo26_detection_postprocessor.hpp
 * @brief YOLO26 object detection postprocessor used by this tutorial.
 */

#ifndef YOLO26_DETECTION_POSTPROCESSOR_HPP
#define YOLO26_DETECTION_POSTPROCESSOR_HPP

#include <algorithm>
#include <string>
#include <vector>

#include "common/base/i_processor.hpp"
#include "common/processors/yolo26_detection_decoder.hpp"

namespace dxapp {

class YOLOv26Postprocessor : public IPostprocessor<DetectionResult> {
public:
    YOLOv26Postprocessor(int input_width = 640, int input_height = 640,
                         float score_threshold = 0.3f,
                         float nms_threshold = 0.45f,
                         bool is_ort_configured = false)
        : impl_(input_width, input_height, score_threshold, nms_threshold,
                is_ort_configured) {}

    std::vector<DetectionResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        auto decoded = impl_.postprocess(outputs);
        std::vector<DetectionResult> results;
        results.reserve(decoded.size());

        for (const auto& source : decoded) {
            DetectionResult result(source.box, source.confidence,
                                   source.class_id, source.class_name);
            scaleBox(result.box, ctx);
            results.push_back(std::move(result));
        }
        return results;
    }

    std::string getModelName() const override { return "YOLOv26"; }

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

    YOLOv26PostProcess impl_;
};

}  // namespace dxapp

#endif  // YOLO26_DETECTION_POSTPROCESSOR_HPP
