/**
 * @file model_version.hpp
 * @brief DXRT and compiled-model version compatibility check.
 */

#ifndef YOLO26_MODEL_VERSION_HPP
#define YOLO26_MODEL_VERSION_HPP

#include <dxrt/dxrt_cxx_api.h>

#include <iostream>
#include <sstream>
#include <string>

namespace dxapp {

inline bool isVersionGreaterOrEqual(const std::string& lhs,
                                    const std::string& rhs) {
    std::istringstream left(lhs);
    std::istringstream right(rhs);
    int left_number = 0;
    int right_number = 0;
    char separator = 0;

    while (left.good() || right.good()) {
        if (left.good()) left >> left_number;
        if (right.good()) right >> right_number;

        if (left_number < right_number) return false;
        if (left_number > right_number) return true;

        left_number = 0;
        right_number = 0;
        if (left.good()) left >> separator;
        if (right.good()) right >> separator;
    }
    return true;
}

inline bool minversionforRTandCompiler(dxrt::InferenceEngine* engine) {
    if (!engine) return false;

    const std::string runtime_version =
        dxrt::Configuration::GetInstance().GetVersion();
    const std::string compiler_version = engine->GetModelVersion();

    if (!isVersionGreaterOrEqual(runtime_version, "3.0.0")) {
        std::cerr << "[DXAPP] [ER] DXRT library version is too low. (required: "
                     ">= 3.0.0, current: "
                  << runtime_version << ")" << std::endl;
        return false;
    }
    if (!isVersionGreaterOrEqual(compiler_version, "v7")) {
        std::cerr << "[DXAPP] [ER] Compiler version is too low. (required: "
                     ">= 7, current: "
                  << compiler_version << ")" << std::endl;
        return false;
    }
    return true;
}

}  // namespace dxapp

#endif  // YOLO26_MODEL_VERSION_HPP
