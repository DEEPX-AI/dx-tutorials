/**
 * @file yolo26s_pose_factory.hpp
 * @brief Yolo26s_pose Abstract Factory implementation
 * 
 * Uses v3-native YOLOv8Pose postprocessor (anchor-free, transposed output).
 */

#ifndef YOLO26S_POSE_FACTORY_HPP
#define YOLO26S_POSE_FACTORY_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo26_pose_postprocessor.hpp"

namespace dxapp {

class Yolo26s_poseFactory {
public:
    Yolo26s_poseFactory(float score_threshold = 0.3f,
                      float nms_threshold = 0.45f)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) {
        (void)is_ort_configured;
        return std::make_unique<YOLO26PosePostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_
        );
    }

private:
    float score_threshold_;
    float nms_threshold_;
};

}  // namespace dxapp

#endif  // YOLO26S_POSE_FACTORY_HPP
