/**
 * @file preprocessing.hpp
 * @brief Common preprocessing utilities (letterbox, color conversion, etc.)
 */

#ifndef DXAPP_PREPROCESSING_HPP
#define DXAPP_PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include "../base/i_processor.hpp"

namespace dxapp {

/**
 * @brief Calculate letterbox padding for maintaining aspect ratio
 * @param img_width Original image width
 * @param img_height Original image height
 * @param target_width Target model input width
 * @param target_height Target model input height
 * @param ctx Output preprocessing context with padding info
 */
inline void calculateLetterboxParams(int img_width, int img_height,
                                     int target_width, int target_height,
                                     PreprocessContext& ctx) {
    ctx.original_width = img_width;
    ctx.original_height = img_height;
    ctx.input_width = target_width;
    ctx.input_height = target_height;

    // Calculate scale to maintain aspect ratio
    ctx.scale = std::min(
        static_cast<float>(target_width) / img_width,
        static_cast<float>(target_height) / img_height
    );

    // Calculate padding
    int new_width = static_cast<int>(img_width * ctx.scale);
    int new_height = static_cast<int>(img_height * ctx.scale);
    
    ctx.pad_x = (target_width - new_width) / 2;
    ctx.pad_y = (target_height - new_height) / 2;
}

/**
 * @brief Apply letterbox transformation to an image
 * @param image Input image (BGR format)
 * @param output Output image (resized with padding)
 * @param target_width Target width
 * @param target_height Target height
 * @param color_conversion OpenCV color conversion code
 * @param ctx Output preprocessing context
 * @param pad_value Padding pixel value (default 114 for YOLO models)
 */
inline void makeLetterboxImage(const cv::Mat& image, cv::Mat& output,
                               int target_width, int target_height,
                               int color_conversion,
                               PreprocessContext& ctx,
                               int pad_value = 114) {
    calculateLetterboxParams(image.cols, image.rows, target_width, target_height, ctx);

    // Calculate new dimensions after scaling (must match calculateLetterboxParams)
    int new_width = static_cast<int>(image.cols * ctx.scale);
    int new_height = static_cast<int>(image.rows * ctx.scale);

    // Resize to new dimensions
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    // Convert color space
    cv::Mat converted;
    cv::cvtColor(resized, converted, color_conversion);

    // Asymmetric padding (matching original: remainder goes to bottom/right)
    int top = ctx.pad_y;
    int bottom = target_height - new_height - top;
    int left = ctx.pad_x;
    int right = target_width - new_width - left;

    // Add padding
    cv::copyMakeBorder(converted, output,
                       top, bottom,
                       left, right,
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(pad_value, pad_value, pad_value));
}

}  // namespace dxapp

#endif  // DXAPP_PREPROCESSING_HPP
