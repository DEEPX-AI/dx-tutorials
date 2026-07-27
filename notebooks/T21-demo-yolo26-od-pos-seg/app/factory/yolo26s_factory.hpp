/**
 * @file yolo26s_factory.hpp
 * @brief Yolo26s Abstract Factory implementation
 */

#ifndef YOLO26S_FACTORY_HPP
#define YOLO26S_FACTORY_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo26_detection_postprocessor.hpp"

namespace dxapp {

class Yolo26sFactory {
public:
    Yolo26sFactory(float score_threshold = 0.3f,
                   float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) {
        return std::make_unique<YOLOv26Postprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_,
            is_ort_configured
        );
    }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLO26S_FACTORY_HPP
