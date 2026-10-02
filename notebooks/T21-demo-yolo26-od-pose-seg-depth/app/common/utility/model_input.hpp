/**
 * @file model_input.hpp
 * @brief Model input-shape parsing and float-buffer conversion helpers.
 */

#ifndef YOLO26_MODEL_INPUT_HPP
#define YOLO26_MODEL_INPUT_HPP

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

inline bool isInputNHWC(const std::vector<int64_t>& shape) {
    return shape.size() >= 4 && shape[3] <= 4 && shape[1] > shape[3];
}

inline void parseInputShape(const std::vector<int64_t>& shape,
                            int& width, int& height) {
    if (shape.size() >= 4) {
        if (isInputNHWC(shape)) {
            height = static_cast<int>(shape[1]);
            width = static_cast<int>(shape[2]);
        } else {
            height = static_cast<int>(shape[2]);
            width = static_cast<int>(shape[3]);
        }
    } else if (shape.size() == 3) {
        height = static_cast<int>(shape[1]);
        width = static_cast<int>(shape[2]);
    } else if (shape.size() >= 2) {
        height = static_cast<int>(shape[0]);
        width = static_cast<int>(shape[1]);
    } else {
        height = 0;
        width = 0;
    }
}

inline std::vector<float> convertToFloatBuffer(const cv::Mat& image, bool nhwc) {
    const int height = image.rows;
    const int width = image.cols;
    const int channels = image.channels();
    std::vector<float> buffer(height * width * channels);

    if (channels == 1) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buffer[y * width + x] = image.at<uint8_t>(y, x) / 255.0f;
            }
        }
    } else if (nhwc) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int channel = 0; channel < channels; ++channel) {
                    buffer[y * width * channels + x * channels + channel] =
                        image.at<cv::Vec3b>(y, x)[channel] / 255.0f;
                }
            }
        }
    } else {
        for (int channel = 0; channel < channels; ++channel) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    buffer[channel * height * width + y * width + x] =
                        image.at<cv::Vec3b>(y, x)[channel] / 255.0f;
                }
            }
        }
    }
    return buffer;
}

#endif  // YOLO26_MODEL_INPUT_HPP
