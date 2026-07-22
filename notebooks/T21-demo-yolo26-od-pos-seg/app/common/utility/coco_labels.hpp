/**
 * @file coco_labels.hpp
 * @brief COCO class names used by YOLO26 detection and segmentation.
 */

#ifndef YOLO26_COCO_LABELS_HPP
#define YOLO26_COCO_LABELS_HPP

#include <array>
#include <string>

namespace dxapp::common {

inline std::string get_coco_class_name(int class_id) {
    static const std::array<const char*, 80> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    };

    if (class_id < 0 || class_id >= static_cast<int>(class_names.size())) {
        return "class_" + std::to_string(class_id);
    }
    return class_names[static_cast<size_t>(class_id)];
}

}  // namespace dxapp::common

#endif  // YOLO26_COCO_LABELS_HPP
