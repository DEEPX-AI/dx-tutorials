#include <dxrt/inference_option.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <climits>  // For INT_MAX, UINT_MAX, LLONG_MAX, and related limits
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <string.h>
#include <errno.h>
#include <cctype>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_FREETYPE
#include <opencv2/freetype.hpp>
#endif
#ifdef HAVE_FREETYPE_DIRECT
#include <ft2build.h>
#include FT_FREETYPE_H
#endif

#include "display.h"

#include "od.h"

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <utils/common_util.hpp>
#include <dxrt/device_info_status.h>

#define DISPLAY_WINDOW_NAME "Object Detection"
#define EXPAND_WINDOW_NAME "Expand Window"
#define EXPAND_WINDOW 1

/* Tuning point */
#define INPUT_CAPTURE_PERIOD_MS 33

static bool g_full_screan = false;

static void notifyLauncherReady()
{
    const char* readyPath = std::getenv("DX_LAUNCHER_READY_FILE");
    if (readyPath == nullptr || *readyPath == '\0')
    {
        return;
    }

    // Readiness notification must never prevent the demo from running.
    std::ofstream out(readyPath);
    if (out)
    {
        out << "ready\n";
    }
}

/**
 * @brief AppConfig Definition
 *      application_type : 0 (single), 1 (multi)
 */
struct AppConfig
{
    int application_type;
    std::string model_path;
    std::string model_name;
    std::vector<std::pair<std::string, std::string>> video_sources;
    std::vector<int> pre_saved_frame_count;
    std::string display_label;

    int input_capture_period_ms;

    int board_width;
    int board_height;

    int is_show_fps;
    int is_fill_blank;
    int is_expand_mode;
    int is_fullsize_mode;

    int grid_cols;   // 0 = auto (ceil(sqrt(N)))
    int grid_rows;   // 0 = auto
    int num_devices; // for sidebar display
    float sidebar_font_scale; // 1.0 = default, <1.0 smaller, >1.0 larger
    float fps_value_font_scale; // 0.5 = default
};

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640,
yolov7_512, yolov7_640, yolov8_640, yolox_s_512, yolov5s_face_640, yolov3_512, yolov4_416,
yolov9_640, yolov5s_512_ppu, scrfd_face_640_ppu;
std::vector<YoloParam> yoloParams = {
    yolov5s_320,
    yolov5s_512,
    yolov5s_640,
    yolov7_512,
    yolov7_640,
    yolov8_640,
    yolox_s_512,
    yolov5s_face_640,
    yolov3_512,
    yolov4_416,
    yolov9_640,
    yolov5s_512_ppu,
    scrfd_face_640_ppu
};

const char* usage =
"yolo demo\n"
"  -c, --config        use config json file for run application\n"
"                      e.g. sudo yolo_multi -c _multi_od_.json -a \n"
"      --window_size    FPS by average over the last {window_size} seconds (default: 60)\n"
"                      e.g. sudo yolo_multi -c _multi_od_.json --window_size 60\n"
"      --exit-btn       show a small exit button overlay in the top-right corner\n"
"  -h, --help          show help\n"
;

void help()
{
    std::cout << usage << std::endl;
}

int ApplicationJsonParser(std::string configPath, AppConfig* dst)
{
    std::ifstream ifs(configPath);
    DXRT_ASSERT(ifs.is_open(), "can't open " + configPath );
    std::string json((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    std::cout << buffer.GetString() << std::endl;
    if(doc.IsObject())
    {
        DXRT_ASSERT(doc.HasMember("usage"), "ERR. usage argument not placed");
        DXRT_ASSERT(doc["usage"].IsString(), "ERR. usage argument must be str");
        dst->application_type = std::string(doc["usage"].GetString()) == "multi" ? 1 : 0;

        DXRT_ASSERT(doc.HasMember("model_path"), "ERR. model_path argument not placed");
        DXRT_ASSERT(doc["model_path"].IsString(), "ERR. model_path argument must be str");
        dst->model_path = doc["model_path"].GetString();

        DXRT_ASSERT(doc.HasMember("model_name"), "ERR. model_name argument not placed");
        DXRT_ASSERT(doc["model_name"].IsString(), "ERR. model_name argument must be str");
        dst->model_name = doc["model_name"].GetString();

        DXRT_ASSERT(doc.HasMember("display_config"), "ERR. display_config argument not placed");
        DXRT_ASSERT(doc["display_config"].IsObject(), "ERR. display_config must be json object");

        const rapidjson::Value& displayConfig = doc["display_config"];

        DXRT_ASSERT(displayConfig.HasMember("display_label"), "ERR. display_label argument not placed");
        DXRT_ASSERT(displayConfig["display_label"].IsString(), "ERR. display_label must be str");
        dst->display_label = displayConfig["display_label"].GetString();

        if(!displayConfig.HasMember("output_width"))
        {
            g_full_screan = true;
#ifdef __linux__
            std::ifstream graphics_info_file("/sys/class/graphics/fb0/virtual_size");
            if(!graphics_info_file)
            {
                std::cout << "Failed to open framebuffer info, It will be set FHD size" << std::endl;
                dst->board_width = 1920;
                dst->board_height = 1080;
            }
            else
            {
                int graphics_info_w, graphics_info_h;
                char comma;
                graphics_info_file >> graphics_info_w >> comma >> graphics_info_h;
                dst->board_width = graphics_info_w;
                dst->board_height = graphics_info_h;
            }
#elif _WIN32
            dst->board_width = GetSystemMetrics(SM_CXSCREEN);
            dst->board_height = GetSystemMetrics(SM_CYSCREEN);
#endif
        }
        else
        {
            DXRT_ASSERT(displayConfig["output_width"].IsInt(), "ERR. output_width must be integer");
            dst->board_width = displayConfig["output_width"].GetInt();

            DXRT_ASSERT(displayConfig.HasMember("output_height"), "ERR. output_height argument not placed");
            DXRT_ASSERT(displayConfig["output_height"].IsInt(), "ERR. output_height must be integer");
            dst->board_height = displayConfig["output_height"].GetInt();
        }
        if(displayConfig.HasMember("capture_period"))
        {
            DXRT_ASSERT(displayConfig["capture_period"].IsInt(), "ERR. capture_period must be integer");
            dst->input_capture_period_ms = displayConfig["capture_period"].GetInt();
        }
        else
        {
            dst->input_capture_period_ms = INPUT_CAPTURE_PERIOD_MS;
        }

        if(displayConfig.HasMember("show_fps"))
        {
            DXRT_ASSERT(displayConfig["show_fps"].IsBool(), "ERR. show_fps must be boolean");
            dst->is_show_fps = displayConfig["show_fps"].GetBool();
        }
        else
        {
            dst->is_show_fps = true;
        }

        if(displayConfig.HasMember("fill_blank"))
        {
            DXRT_ASSERT(displayConfig["fill_blank"].IsBool(), "ERR. fill_blank must be boolean");
            dst->is_fill_blank = displayConfig["fill_blank"].GetBool();
        }
        else
        {
            dst->is_fill_blank = true;
        }

        if(displayConfig.HasMember("expand_mode"))
        {
            DXRT_ASSERT(displayConfig["expand_mode"].IsBool(), "ERR. expand_mode must be boolean");
            dst->is_expand_mode = displayConfig["expand_mode"].GetBool();
        }
        else
        {
            dst->is_expand_mode = false;
        }
        if(displayConfig.HasMember("dynamic_window_mode"))
        {
            DXRT_ASSERT(displayConfig["dynamic_window_mode"].IsBool(), "ERR. dynamic_window_mode must be boolean");
            dst->is_fullsize_mode = !displayConfig["dynamic_window_mode"].GetBool();
        }
        else
        {
            dst->is_fullsize_mode = true;
        }

        dst->grid_cols = 0;
        dst->grid_rows = 0;
        if(displayConfig.HasMember("grid_cols"))
        {
            DXRT_ASSERT(displayConfig["grid_cols"].IsInt(), "ERR. grid_cols must be integer");
            dst->grid_cols = displayConfig["grid_cols"].GetInt();
        }
        if(displayConfig.HasMember("grid_rows"))
        {
            DXRT_ASSERT(displayConfig["grid_rows"].IsInt(), "ERR. grid_rows must be integer");
            dst->grid_rows = displayConfig["grid_rows"].GetInt();
        }

        dst->num_devices = 1;
        if(doc.HasMember("num_devices"))
        {
            DXRT_ASSERT(doc["num_devices"].IsInt(), "ERR. num_devices must be integer");
            dst->num_devices = doc["num_devices"].GetInt();
        }

        dst->sidebar_font_scale = 1.0f;
        if(displayConfig.HasMember("sidebar_font_scale"))
        {
            DXRT_ASSERT(displayConfig["sidebar_font_scale"].IsNumber(), "ERR. sidebar_font_scale must be number");
            dst->sidebar_font_scale = (float)displayConfig["sidebar_font_scale"].GetDouble();
        }

        dst->fps_value_font_scale = 0.5f;
        if(displayConfig.HasMember("fps_value_font_scale"))
        {
            DXRT_ASSERT(displayConfig["fps_value_font_scale"].IsNumber(), "ERR. fps_value_font_scale must be number");
            dst->fps_value_font_scale = (float)displayConfig["fps_value_font_scale"].GetDouble();
        }

        DXRT_ASSERT(doc.HasMember("video_sources"), "ERR. video_sources argument not placed");
        DXRT_ASSERT(doc["video_sources"].IsArray(), "ERR. video_sources must be array");
        const rapidjson::Value& videoSources = doc["video_sources"];
        for(rapidjson::SizeType i = 0; i < videoSources.Size(); i++){
            const rapidjson::Value& videoSource = videoSources[i];
            std::pair<std::string, std::string> videoSourceInfo(std::pair<std::string, std::string>(videoSource[0].GetString(), videoSource[1].GetString()));
#if __riscv
            if(std::string(videoSource[1].GetString()) == "isp"){
                dst->video_sources.clear();
                dst->pre_saved_frame_count.clear();
                dst->video_sources.emplace_back(videoSourceInfo);
                dst->pre_saved_frame_count.emplace_back(-1);
                return 1;
            }
#endif
            if(std::string(videoSource[1].GetString()) == "offline")
            {
                if(videoSource.Size() == 2)
                {
                    dst->pre_saved_frame_count.emplace_back(0);
                }
                else if(videoSource.Size() == 3)
                {
                    dst->pre_saved_frame_count.emplace_back(videoSource[2].GetInt());
                }
            }else{
                dst->pre_saved_frame_count.emplace_back(-1);
            }
            dst->video_sources.emplace_back(videoSourceInfo);
        }
    }else{
        return -1;
    }
    return 1;
}

YoloParam getYoloParameter(std::string model_name){
    if(model_name == "yolov5s_320")
        return yolov5s_320;
    else if(model_name == "yolov5s_512")
        return yolov5s_512;
    else if(model_name == "yolov5s_640")
        return yolov5s_640;
    else if(model_name == "yolox_s_512")
        return yolox_s_512;
    else if(model_name == "yolov7_640")
        return yolov7_640;
    else if(model_name == "yolov7_512")
        return yolov7_512;
    else if(model_name == "yolov8_640")
        return yolov8_640;
    else if(model_name == "yolov5s_face_640")
        return yolov5s_face_640;
    else if(model_name == "yolov3_512")
        return yolov3_512;
    else if(model_name == "yolov4_416")
        return yolov4_416;
    else if(model_name == "yolov9_640")
        return yolov9_640;
    else if(model_name == "yolov5s_512_ppu")
        return yolov5s_512_ppu;
    else if(model_name == "scrfd_face_640_ppu")
        return scrfd_face_640_ppu;
    return yolov5s_512;
}
YoloParam yoloParam;

// --- CPU load measurement (Linux) ---
#ifdef __linux
static float getCpuLoad()
{
    static uint64_t lastIdle = 0, lastTotal = 0;
    std::ifstream stat("/proc/stat");
    if(!stat.is_open()) return 0.f;
    std::string cpu;
    uint64_t user, nice, system, idle, iowait = 0, irq = 0, softirq = 0;
    stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    uint64_t total = user + nice + system + idle + iowait + irq + softirq;
    uint64_t deltaIdle  = idle  - lastIdle;
    uint64_t deltaTotal = total - lastTotal;
    lastIdle  = idle;
    lastTotal = total;
    if(deltaTotal == 0) return 0.f;
    return (1.0f - (float)deltaIdle / deltaTotal) * 100.f;
}
#else
static float getCpuLoad() { return 0.f; }
#endif

// --- NPU average temperature ---
static float getNpuAvgTemp(int numDevices)
{
    if(numDevices <= 0) return 0.f;
    int sum = 0, count = 0;
    for(int i = 0; i < numDevices; i++) {
        try {
            auto status = dxrt::DeviceStatus::GetCurrentStatus(i);
            int temp = status.Temperature(0);
            if(temp > 0) { sum += temp; count++; }
        } catch(...) {}
    }
    return (count > 0) ? (float)sum / count : 0.f;
}

// Helper for grid layout (unused after Task 3; may be used in future)
static int devideBoard(int numImages)
{
    return (int)ceil(sqrt(numImages));
}

// Suppress unused function warnings for CPU/NPU helpers removed by Task 3
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
static void suppressUnusedTaskThreeHelpers() {
    (void)getCpuLoad;
    (void)getNpuAvgTemp;
    (void)devideBoard;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// --- EXIT button overlay ---
static bool g_exitRequested = false;
static bool g_showExitBtn = false;
static cv::Rect g_exitBtnRect;

static void drawExitButtonOverlay(cv::Mat& frame)
{
    if(!g_showExitBtn || frame.empty()) return;

    int btnSize = std::max(18, std::min(24, frame.cols / 80));
    int margin = std::max(4, frame.cols / 200);
    int x = frame.cols - btnSize - margin;
    int y = margin;
    g_exitBtnRect = cv::Rect(x, y, btnSize, btnSize);

    const cv::Scalar bgColor(38, 38, 38);
    const cv::Scalar xColor(58, 58, 58);
    cv::rectangle(frame, g_exitBtnRect, bgColor, cv::FILLED, cv::LINE_AA);

    int pad = btnSize / 4;
    cv::line(frame, cv::Point(x + pad, y + pad),
             cv::Point(x + btnSize - pad, y + btnSize - pad), xColor, 1, cv::LINE_AA);
    cv::line(frame, cv::Point(x + btnSize - pad, y + pad),
             cv::Point(x + pad, y + btnSize - pad), xColor, 1, cv::LINE_AA);
}

// --- DEEPX device display names ---
// Create user-friendly names from the "Device N: <variant>" output of dxrt-cli -s.
// Example: an M1 chip on an M.2 board becomes "DX-M1" and an H1 board becomes
// "DX-H1 Quattro". Multiple devices with the same name are grouped as "DX-M1 * 3".
static std::vector<std::string> getDeepxDeviceNames(int numDevices)
{
    std::vector<std::string> order;
    std::map<std::string, int> counts;
    for(int i = 0; i < numDevices; i++) {
        std::string name;
        try {
            auto status = dxrt::DeviceStatus::GetCurrentStatus(i);
            std::string board = status.BoardTypeStr();
            std::string variant = status.DeviceVariantStr();
            if(board == "H1") name = "DX-H1 Quattro";
            else if(!variant.empty()) name = "DX-" + variant;
            else name = "DEEPX DEVICE";
        } catch(...) {
            name = "DEEPX DEVICE";
        }
        if(counts.find(name) == counts.end()) order.push_back(name);
        counts[name]++;
    }
    // One H1 Quattro board is reported as four chips, so divide the count by four.
    auto it = counts.find("DX-H1 Quattro");
    if(it != counts.end() && it->second >= 4) it->second /= 4;

    std::vector<std::string> result;
    for(const auto& n : order) {
        int c = counts[n];
        if(c <= 1) result.push_back(n);
        else result.push_back(n + " * " + std::to_string(c));
    }
    return result;
}

static void onMouseCallback(int event, int x, int y, int /*flags*/, void* /*userdata*/)
{
    if(event == cv::EVENT_LBUTTONDOWN && g_showExitBtn && g_exitBtnRect.contains(cv::Point(x, y)))
    {
        g_exitRequested = true;
    }
}

// --- HeaderUI: weighted Montserrat font loader ---
enum class HeaderFontWeight { Regular = 0, SemiBold, Bold, ExtraBold, Count };

static std::string fontFileName(HeaderFontWeight weight)
{
    switch(weight) {
        case HeaderFontWeight::Regular:   return "Montserrat-Regular.ttf";
        case HeaderFontWeight::SemiBold:  return "Montserrat-SemiBold.ttf";
        case HeaderFontWeight::Bold:      return "Montserrat-Bold.ttf";
        case HeaderFontWeight::ExtraBold: return "Montserrat-ExtraBold.ttf";
        default: return "Montserrat-Regular.ttf";
    }
}

static std::vector<std::string> fontCandidates(HeaderFontWeight weight)
{
    std::string fileName = fontFileName(weight);
    std::vector<std::string> candidates;

#ifdef PROJECT_ROOT_DIR
    candidates.push_back(std::string(PROJECT_ROOT_DIR) + "/sample/fonts/" + fileName);
#endif
    candidates.push_back("./sample/fonts/" + fileName);
    candidates.push_back("../sample/fonts/" + fileName);

    // Fallback to system DejaVu fonts
    candidates.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf");
    candidates.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

    return candidates;
}

#ifdef HAVE_OPENCV_FREETYPE
static cv::Ptr<cv::freetype::FreeType2> g_headerFonts[static_cast<int>(HeaderFontWeight::Count)];
static bool g_headerFonts_initialized[static_cast<int>(HeaderFontWeight::Count)] = { false, false, false, false };

static cv::Ptr<cv::freetype::FreeType2> getHeaderFont(HeaderFontWeight weight)
{
    int idx = static_cast<int>(weight);
    if(!g_headerFonts_initialized[idx])
    {
        g_headerFonts_initialized[idx] = true;
        auto candidates = fontCandidates(weight);
        for(const auto& path : candidates)
        {
            try {
                auto ft = cv::freetype::createFreeType2();
                ft->loadFontData(path, 0);
                g_headerFonts[idx] = ft;
                std::cout << "[HeaderUI] Loaded font: " << path << std::endl;
                break;
            } catch(...) {
                // Try next candidate
            }
        }
    }
    return g_headerFonts[idx];
}
#endif

#ifdef HAVE_FREETYPE_DIRECT
static FT_Library directFreeTypeLibrary()
{
    static FT_Library library = nullptr;
    static bool initialized = false;

    if(!initialized) {
        initialized = true;
        if(FT_Init_FreeType(&library) != 0) {
            library = nullptr;
        }
    }

    return library;
}

static FT_Face getDirectHeaderFace(HeaderFontWeight weight)
{
    static FT_Face faces[static_cast<int>(HeaderFontWeight::Count)] = { nullptr, nullptr, nullptr, nullptr };
    static bool initialized[static_cast<int>(HeaderFontWeight::Count)] = { false, false, false, false };

    int idx = static_cast<int>(weight);
    if(!initialized[idx]) {
        initialized[idx] = true;
        FT_Library library = directFreeTypeLibrary();
        if(library != nullptr) {
            auto candidates = fontCandidates(weight);
            for(const auto& path : candidates) {
                FT_Face face = nullptr;
                if(FT_New_Face(library, path.c_str(), 0, &face) == 0) {
                    faces[idx] = face;
                    std::cout << "[HeaderUI] Loaded font with FreeType fallback: " << path << std::endl;
                    break;
                }
            }
        }
    }

    return faces[idx];
}

static cv::Size headerTextSizeWithFreeType(const std::string& text, int fontH, HeaderFontWeight weight)
{
    FT_Face face = getDirectHeaderFace(weight);
    if(face == nullptr || fontH <= 0) return cv::Size(0, 0);
    if(FT_Set_Pixel_Sizes(face, 0, fontH) != 0) return cv::Size(0, 0);

    FT_Pos penX = 0;
    FT_UInt prevGlyph = 0;
    int maxRight = 0;
    int maxTop = 0;
    int maxBottom = 0;
    bool useKerning = FT_HAS_KERNING(face);

    for(unsigned char ch : text) {
        FT_UInt glyphIndex = FT_Get_Char_Index(face, ch);
        if(useKerning && prevGlyph != 0 && glyphIndex != 0) {
            FT_Vector delta;
            if(FT_Get_Kerning(face, prevGlyph, glyphIndex, FT_KERNING_DEFAULT, &delta) == 0) {
                penX += delta.x;
            }
        }
        if(FT_Load_Glyph(face, glyphIndex, FT_LOAD_DEFAULT) != 0) {
            continue;
        }

        FT_GlyphSlot glyph = face->glyph;
        int glyphLeft = (int)((penX + glyph->metrics.horiBearingX) >> 6);
        int glyphRight = (int)((penX + glyph->metrics.horiBearingX + glyph->metrics.width + 63) >> 6);
        int glyphTop = (int)(glyph->metrics.horiBearingY >> 6);
        int glyphBottom = (int)((glyph->metrics.height - glyph->metrics.horiBearingY + 63) >> 6);

        maxRight = std::max(maxRight, std::max(glyphRight, glyphLeft));
        maxTop = std::max(maxTop, glyphTop);
        maxBottom = std::max(maxBottom, glyphBottom);
        penX += glyph->advance.x;
        prevGlyph = glyphIndex;
    }

    int advanceWidth = (int)((penX + 63) >> 6);
    int width = std::max(maxRight, advanceWidth);
    int height = std::max(1, maxTop + maxBottom);
    return cv::Size(std::max(1, width), height);
}

static bool drawHeaderTextWithFreeType(cv::Mat& img, const std::string& text, cv::Point org,
                                       int fontH, HeaderFontWeight weight,
                                       const cv::Scalar& color, int thickness)
{
    (void)thickness;
    FT_Face face = getDirectHeaderFace(weight);
    if(face == nullptr || fontH <= 0 || img.empty() || img.channels() != 3) return false;
    if(FT_Set_Pixel_Sizes(face, 0, fontH) != 0) return false;

    FT_Pos penX = ((FT_Pos)org.x) << 6;
    int baselineY = org.y;
    FT_UInt prevGlyph = 0;
    bool useKerning = FT_HAS_KERNING(face);

    for(unsigned char ch : text) {
        FT_UInt glyphIndex = FT_Get_Char_Index(face, ch);
        if(useKerning && prevGlyph != 0 && glyphIndex != 0) {
            FT_Vector delta;
            if(FT_Get_Kerning(face, prevGlyph, glyphIndex, FT_KERNING_DEFAULT, &delta) == 0) {
                penX += delta.x;
            }
        }
        if(FT_Load_Glyph(face, glyphIndex, FT_LOAD_DEFAULT) != 0) {
            continue;
        }
        if(FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL) != 0) {
            continue;
        }

        FT_GlyphSlot glyph = face->glyph;
        FT_Bitmap& bitmap = glyph->bitmap;
        int startX = (int)(penX >> 6) + glyph->bitmap_left;
        int startY = baselineY - glyph->bitmap_top;

        for(int row = 0; row < (int)bitmap.rows; row++) {
            int dstY = startY + row;
            if(dstY < 0 || dstY >= img.rows) continue;

            const unsigned char* srcRow = bitmap.buffer + row * bitmap.pitch;
            for(int col = 0; col < (int)bitmap.width; col++) {
                int dstX = startX + col;
                if(dstX < 0 || dstX >= img.cols) continue;

                double alpha = srcRow[col] / 255.0;
                if(alpha <= 0.0) continue;

                cv::Vec3b& pixel = img.at<cv::Vec3b>(dstY, dstX);
                pixel[0] = (uchar)(pixel[0] * (1.0 - alpha) + color[0] * alpha);
                pixel[1] = (uchar)(pixel[1] * (1.0 - alpha) + color[1] * alpha);
                pixel[2] = (uchar)(pixel[2] * (1.0 - alpha) + color[2] * alpha);
            }
        }

        penX += glyph->advance.x;
        prevGlyph = glyphIndex;
    }

    return true;
}
#endif

// --- Header HUD rendering helpers ---
static float roundedRectCoverage(float px, float py, int x0, int y0, int x1, int y1, float r)
{
    float nearestX = std::max((float)x0 + r, std::min(px, (float)x1 - r));
    float nearestY = std::max((float)y0 + r, std::min(py, (float)y1 - r));
    float dx = px - nearestX;
    float dy = py - nearestY;
    float dist = std::sqrt(dx * dx + dy * dy) - r;
    return std::max(0.f, std::min(1.f, -dist));
}

static void drawRoundedPrimitive(cv::Mat& img, const cv::Rect& rect, int radius, const cv::Scalar& color, float alpha = 1.f)
{
    if(rect.width <= 0 || rect.height <= 0 || alpha <= 0.f) return;

    int r = std::min(radius, std::min(rect.width / 2, rect.height / 2));
    if(r <= 0) {
        cv::rectangle(img, rect, color, cv::FILLED, cv::LINE_AA);
        return;
    }

    int x0 = rect.x;
    int y0 = rect.y;
    int x1 = rect.x + rect.width;
    int y1 = rect.y + rect.height;
    const float fr = (float)r;

    int clipX0 = std::max(x0, 0);
    int clipY0 = std::max(y0, 0);
    int clipX1 = std::min(x1, img.cols);
    int clipY1 = std::min(y1, img.rows);

    for(int y = clipY0; y < clipY1; ++y) {
        cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
        float py = (float)y + 0.5f;
        for(int x = clipX0; x < clipX1; ++x) {
            float coverage = roundedRectCoverage((float)x + 0.5f, py, x0, y0, x1, y1, fr);
            if(coverage <= 0.f) continue;

            float blendA = coverage * alpha;
            cv::Vec3b& pixel = row[x];
            pixel[0] = (uchar)(pixel[0] * (1.f - blendA) + color[0] * blendA);
            pixel[1] = (uchar)(pixel[1] * (1.f - blendA) + color[1] * blendA);
            pixel[2] = (uchar)(pixel[2] * (1.f - blendA) + color[2] * blendA);
        }
    }
}

static void drawFilledRoundedRect(cv::Mat& img, const cv::Rect& rect, int radius, const cv::Scalar& color, double alpha = 0.95)
{
    cv::Rect clipped = rect & cv::Rect(0, 0, img.cols, img.rows);
    if(clipped.width <= 0 || clipped.height <= 0) return;

    drawRoundedPrimitive(img, clipped, radius, color, (float)alpha);
}

static cv::Size headerTextSize(const std::string& text, int fontH, HeaderFontWeight weight, int thickness = 1)
{
#ifdef HAVE_OPENCV_FREETYPE
    auto ft = getHeaderFont(weight);
    if(!ft.empty()) {
        int baseline = 0;
        return ft->getTextSize(text, fontH, thickness, &baseline);
    }
#elif defined(HAVE_FREETYPE_DIRECT)
    auto size = headerTextSizeWithFreeType(text, fontH, weight);
    if(size.width > 0 && size.height > 0) return size;
#else
    (void)weight;  // unused in fallback builds
#endif
    double scale = std::max(0.1, fontH / 28.0);
    int baseline = 0;
    return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, std::max(1, thickness), &baseline);
}

static int headerTextBaselineYForCenter(const std::string& text, int fontH, HeaderFontWeight weight, int centerY)
{
    auto sz = headerTextSize(text, fontH, weight);
    return centerY + sz.height / 2;
}

static void drawHeaderText(cv::Mat& img, const std::string& text, cv::Point org, int fontH,
                           HeaderFontWeight weight, const cv::Scalar& color, int thickness = -1)
{
#ifdef HAVE_OPENCV_FREETYPE
    auto ft = getHeaderFont(weight);
    if(!ft.empty()) {
        ft->putText(img, text, org, fontH, color, thickness, cv::LINE_AA, true);
        return;
    }
#elif defined(HAVE_FREETYPE_DIRECT)
    if(drawHeaderTextWithFreeType(img, text, org, fontH, weight, color, thickness)) {
        return;
    }
#else
    (void)weight;  // unused in fallback builds
#endif
    double scale = std::max(0.1, fontH / 28.0);
    int thick = (thickness > 0) ? thickness : std::max(1, (int)(scale * 2));
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
}

static int fitHeaderFontHeight(const std::string& text, int fontH, int maxWidth, HeaderFontWeight weight)
{
    auto sz = headerTextSize(text, fontH, weight);
    if(sz.width > maxWidth && sz.width > 0)
        return std::max(10, fontH * maxWidth / sz.width);
    return fontH;
}

// --- Text formatting helpers ---
static bool isNumericToken(const std::string& token)
{
    if(token.empty()) return false;
    for(char ch : token) {
        if(!std::isdigit((unsigned char)ch)) return false;
    }
    return true;
}

static std::string upperToken(const std::string& token)
{
    std::string result;
    for(char ch : token) {
        result += std::toupper((unsigned char)ch);
    }
    return result;
}

static std::string formatModelToken(const std::string& token)
{
    std::string lower = token;
    for(auto& ch : lower) ch = std::tolower((unsigned char)ch);

    // Special formatting for known tokens
    if(lower == "yolov5s") return "YOLOv5S";
    if(lower == "yolov7") return "YOLOv7";
    if(lower == "yolov8") return "YOLOv8";
    if(lower == "yolov9") return "YOLOv9";
    if(lower == "yolox") return "YOLOX";
    if(lower == "yolov3") return "YOLOv3";
    if(lower == "yolov4") return "YOLOv4";
    if(lower == "scrfd") return "SCRFD";
    if(lower == "ppu") return "PPU";
    if(lower == "face") return "Face";
    if(lower.find("yolo") == 0 && lower.size() > 4) {
        // Generic YOLOvX handling
        return "YOLO" + upperToken(lower.substr(4));
    }

    // Default: uppercase
    return upperToken(token);
}

static std::string formatHeaderModelName(const std::string& modelName)
{
    std::vector<std::string> tokens;
    std::string token;
    for(size_t i = 0; i <= modelName.size(); i++) {
        if(i == modelName.size() || modelName[i] == '_') {
            if(!token.empty() && !isNumericToken(token)) {
                tokens.push_back(formatModelToken(token));
            }
            token.clear();
        } else {
            token += modelName[i];
        }
    }

    if(tokens.empty()) return upperToken(modelName);

    std::string result;
    for(size_t i = 0; i < tokens.size(); i++) {
        if(i > 0) result += " ";
        result += tokens[i];
    }
    return result;
}

static std::string headerDeviceLabel(int numDevices)
{
    auto deviceNames = getDeepxDeviceNames(numDevices);
    if(deviceNames.empty()) return "DEEPX DEVICE";

    std::string result;
    for(size_t i = 0; i < deviceNames.size(); i++) {
        if(i > 0) result += ", ";
        result += deviceNames[i];
    }
    return result;
}

static std::string headerHardwareLabel(int numDevices)
{
    std::string deviceLabel = headerDeviceLabel(numDevices);
    if(deviceLabel.find("DEEPX") == 0) return deviceLabel;
    return "DEEPX " + deviceLabel;
}

static std::string fpsValueText(float fps)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << fps;
    return oss.str();
}

static cv::Mat headerDeepxLogo()
{
    static cv::Mat g_deepxLogo;
    static bool g_deepxLogoLoadAttempted = false;

    if(g_deepxLogoLoadAttempted) {
        return g_deepxLogo;
    }

    g_deepxLogoLoadAttempted = true;

    std::vector<std::string> searchPaths;

#ifdef PROJECT_ROOT_DIR
    searchPaths.push_back(std::string(PROJECT_ROOT_DIR) + "/sample/header/deepx_logo.png");
#endif
    searchPaths.push_back("./sample/header/deepx_logo.png");
    searchPaths.push_back("../sample/header/deepx_logo.png");

    for(const auto& path : searchPaths) {
        cv::Mat logo = cv::imread(path, cv::IMREAD_UNCHANGED);
        if(!logo.empty()) {
            g_deepxLogo = logo;
            std::cout << "[HeaderUI] Loaded logo: " << path << std::endl;
            return g_deepxLogo;
        }
    }

    std::cout << "[HeaderUI] Failed to load deepx logo from any search path" << std::endl;
    return cv::Mat();
}

static cv::Mat headerLiveImage()
{
    static cv::Mat g_liveImage;
    static bool g_liveImageLoadAttempted = false;

    if(g_liveImageLoadAttempted) {
        return g_liveImage;
    }

    g_liveImageLoadAttempted = true;

    std::vector<std::string> searchPaths;

#ifdef PROJECT_ROOT_DIR
    searchPaths.push_back(std::string(PROJECT_ROOT_DIR) + "/sample/header/live.png");
#endif
    searchPaths.push_back("./sample/header/live.png");
    searchPaths.push_back("../sample/header/live.png");

    for(const auto& path : searchPaths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
        if(!image.empty()) {
            g_liveImage = image;
            std::cout << "[HeaderUI] Loaded live badge: " << path << std::endl;
            return g_liveImage;
        }
    }

    std::cout << "[HeaderUI] Failed to load live badge from any search path" << std::endl;
    return cv::Mat();
}

static cv::Size headerLogoTargetSize(const cv::Mat& logo, int maxW, int maxH)
{
    static constexpr int MIN_LOGO_DIMENSION = 10;
    if(logo.empty() || logo.cols < MIN_LOGO_DIMENSION || logo.rows < MIN_LOGO_DIMENSION || maxW <= 0 || maxH <= 0) {
        return cv::Size(0, 0);
    }

    double scaleW = (double)maxW / logo.cols;
    double scaleH = (double)maxH / logo.rows;
    double scale = std::min(scaleW, scaleH);

    int targetW = (int)(logo.cols * scale);
    int targetH = (int)(logo.rows * scale);

    if(targetW < MIN_LOGO_DIMENSION || targetH < MIN_LOGO_DIMENSION) {
        return cv::Size(0, 0);
    }

    return cv::Size(targetW, targetH);
}

static cv::Size drawHeaderImage(cv::Mat& frame, cv::Point org, int maxW, int maxH,
                                const cv::Mat& image, int cacheSlot)
{
    if(frame.empty() || frame.cols <= 0 || frame.rows <= 0 || maxW <= 0 || maxH <= 0) {
        return cv::Size(0, 0);
    }

    if(frame.type() != CV_8UC3) {
        return cv::Size(0, 0);
    }

    cv::Size imageSize = headerLogoTargetSize(image, maxW, maxH);

    if(imageSize.width <= 0 || imageSize.height <= 0) {
        return cv::Size(0, 0);
    }

    if(org.x < 0 || org.y < 0 ||
       org.x + imageSize.width > frame.cols ||
       org.y + imageSize.height > frame.rows) {
        return cv::Size(0, 0);
    }

    static cv::Mat cachedResizedImages[2];
    static cv::Size cachedImageSizes[2];
    if(cacheSlot < 0 || cacheSlot >= 2) {
        cacheSlot = 0;
    }

    if(cachedResizedImages[cacheSlot].empty() || cachedImageSizes[cacheSlot] != imageSize) {
        cv::resize(image, cachedResizedImages[cacheSlot], imageSize, 0, 0, cv::INTER_AREA);
        cachedImageSizes[cacheSlot] = imageSize;
    }

    const cv::Mat& resizedImage = cachedResizedImages[cacheSlot];
    if(resizedImage.empty()) {
        return cv::Size(0, 0);
    }

    cv::Rect roi(org.x, org.y, imageSize.width, imageSize.height);

    if(resizedImage.channels() == 4) {
        for(int y = 0; y < resizedImage.rows; y++) {
            for(int x = 0; x < resizedImage.cols; x++) {
                cv::Vec4b pixel = resizedImage.at<cv::Vec4b>(y, x);
                double alpha = pixel[3] / 255.0;

                cv::Vec3b& framePixel = frame.at<cv::Vec3b>(org.y + y, org.x + x);
                framePixel[0] = (uchar)(framePixel[0] * (1.0 - alpha) + pixel[0] * alpha);
                framePixel[1] = (uchar)(framePixel[1] * (1.0 - alpha) + pixel[1] * alpha);
                framePixel[2] = (uchar)(framePixel[2] * (1.0 - alpha) + pixel[2] * alpha);
            }
        }
    } else if(resizedImage.channels() == 3) {
        resizedImage.copyTo(frame(roi));
    } else {
        return cv::Size(0, 0);
    }

    return imageSize;
}

static cv::Size drawHeaderLogo(cv::Mat& frame, cv::Point org, int maxW, int maxH)
{
    return drawHeaderImage(frame, org, maxW, maxH, headerDeepxLogo(), 0);
}

// --- Unified Header HUD renderer (cached static layer + per-frame FPS update) ---
static void renderHeaderHud(cv::Mat& frame,
                           int boardW, int titleH,
                           int activeStreams, int numDevices,
                           float totalFps,
                           bool showFps, bool calcFps,
                           const std::string& modelName,
                           float fpsValueFontScale = 0.5f)
{
    if(frame.empty() || boardW <= 0 || titleH <= 0) return;

    // --- Static cache for the HUD layer (rebuilt only when parameters change) ---
    static cv::Mat s_hudCache;
    static int s_boardW = 0, s_titleH = 0, s_streams = 0, s_devices = 0;
    static std::string s_model;
    static float s_fpsScale = -1.0f;
    // Cached positions for per-frame FPS value drawing
    static int s_totalValueSlotX = 0, s_avgValueSlotX = 0;
    static int s_totalValueSlotW = 0, s_avgValueSlotW = 0;
    static int s_metricsBaselineY = 0, s_valueFontH = 0;

    bool needRebuild = s_hudCache.empty() ||
                       s_boardW != boardW || s_titleH != titleH ||
                       s_streams != activeStreams || s_devices != numDevices ||
                       s_model != modelName || s_fpsScale != fpsValueFontScale;

    if(needRebuild) {
        s_hudCache = cv::Mat::zeros(titleH, boardW, CV_8UC3);
        s_boardW = boardW; s_titleH = titleH;
        s_streams = activeStreams; s_devices = numDevices;
        s_model = modelName; s_fpsScale = fpsValueFontScale;

        // Colors (BGR)
        const cv::Scalar COLOR_HUD_BG   (26,  26,  26);
        const cv::Scalar COLOR_DEEPX    (255, 85,  47);
        const cv::Scalar COLOR_TEXT_PRI (245, 245, 245);
        const cv::Scalar COLOR_TEXT_SEC (208, 208, 208);

        // Layout proportions
        int cardH   = titleH;
        int cardTop = 0;
        int marginX = 0;
        int gap     = std::max(4, (int)(boardW * 0.008));
        int metricsW = std::max(280, (int)(boardW * 0.285));
        int mainW   = boardW - marginX * 2 - gap - metricsW;
        int mainCardRadius = std::max(16, (int)(cardH * 0.18));

        if(mainW < 320) {
            metricsW = std::max(160, boardW - marginX * 2 - gap - 320);
            if(metricsW < 160) metricsW = 160;
            mainW = boardW - marginX * 2 - gap - metricsW;
            if(mainW < 1) mainW = 1;
        }

        int mainX    = marginX;
        int metricsX = marginX + mainW + gap;

        // Draw cards (opaque — no alpha blend needed for cache)
        drawFilledRoundedRect(s_hudCache, cv::Rect(mainX, cardTop, mainW, cardH), mainCardRadius, COLOR_HUD_BG, 1.0);
        drawFilledRoundedRect(s_hudCache, cv::Rect(metricsX, cardTop, metricsW, cardH), mainCardRadius, COLOR_HUD_BG, 1.0);

        // --- Main card content ---
        int contentX = mainX + std::max(20, (int)(mainW * 0.035));
        int contentY = cardTop + cardH / 2;

        // DEEPX logo
        cv::Size logoSize = headerLogoTargetSize(headerDeepxLogo(), mainW / 4, (int)(cardH * 0.45));
        int afterDeepx;

        if(logoSize.width > 0 && logoSize.height > 0) {
            int logoY = contentY - logoSize.height / 2;
            logoSize = drawHeaderLogo(s_hudCache, cv::Point(contentX, logoY), mainW / 4, (int)(cardH * 0.45));
        }

        if(logoSize.width > 0 && logoSize.height > 0) {
            afterDeepx = contentX + logoSize.width + std::max(16, (int)(mainW * 0.028));
        } else {
            int deepxFontH = std::max(28, (int)(cardH * 0.45));
            deepxFontH = fitHeaderFontHeight("DEEPX", deepxFontH, mainW / 4, HeaderFontWeight::ExtraBold);
            auto deepxSz = headerTextSize("DEEPX", deepxFontH, HeaderFontWeight::ExtraBold);
            drawHeaderText(s_hudCache, "DEEPX", cv::Point(contentX, contentY - deepxSz.height / 6),
                           deepxFontH, HeaderFontWeight::ExtraBold, COLOR_DEEPX);
            afterDeepx = contentX + deepxSz.width + std::max(16, (int)(mainW * 0.028));
        }

        std::string displayModel = formatHeaderModelName(modelName);

        // Title + LIVE badge layout
        int badgeH = std::max(20, (int)(cardH * 0.40));
        int liveMaxW = std::max(80, (int)(mainW * 0.30));
        cv::Size liveSize = headerLogoTargetSize(headerLiveImage(), liveMaxW, badgeH);
        if(liveSize.width <= 0 || liveSize.height <= 0) {
            liveSize = cv::Size(std::max(80, (int)(badgeH * 2.8)), badgeH);
        }
        int badgeW = liveSize.width;
        int rightPad = std::max(14, (int)(mainW * 0.044));
        int separatorGap = std::max(14, (int)(mainW * 0.018));
        int subAreaW = std::max(230, (int)(mainW * 0.28));
        subAreaW = std::min(subAreaW, std::max(180, (int)(mainW * 0.34)));
        int separatorX = mainX + mainW - rightPad - subAreaW - separatorGap;
        int titleBadgeGap = std::max(10, (int)(mainW * 0.012));
        int titleMaxW = separatorX - afterDeepx - badgeW - titleBadgeGap * 2;
        titleMaxW = std::max(80, titleMaxW);

        std::string title = std::to_string(activeStreams) + " CH. Real-time Processing";
        int titleFontH = std::max(16, (int)(cardH * 0.36));
        int titleTextMaxW = std::max(80, titleMaxW);
        titleFontH = fitHeaderFontHeight(title, titleFontH, titleTextMaxW, HeaderFontWeight::Bold);
        auto titleTextSz = headerTextSize(title, titleFontH, HeaderFontWeight::Bold);
        int titleBadgeNarrowGap = std::max(8, (int)(mainW * 0.02));
        int titleAreaInset = std::max(6, (int)(titleMaxW * 0.02));
        int titleX = afterDeepx + titleAreaInset;
        int titleY = headerTextBaselineYForCenter(title, titleFontH, HeaderFontWeight::Bold, contentY);

        drawHeaderText(s_hudCache, title, cv::Point(titleX, titleY - (titleTextSz.height * 0.05)),
                       titleFontH, HeaderFontWeight::Bold, COLOR_TEXT_PRI);

        // LIVE badge image
        int badgeX = titleX + titleTextSz.width + titleBadgeNarrowGap;
        badgeX = std::min(badgeX, separatorX - badgeW);
        badgeX = std::max(afterDeepx, badgeX);
        int badgeY = cardTop + (cardH - liveSize.height) / 2;
        drawHeaderImage(s_hudCache, cv::Point(badgeX, badgeY), liveMaxW, badgeH, headerLiveImage(), 1);

        // Separator
        int separatorTop = cardTop + std::max(10, (int)(cardH * 0.38));
        int separatorBottom = cardTop + cardH - std::max(10, (int)(cardH * 0.38));
        cv::line(s_hudCache, cv::Point(separatorX, separatorTop),
                 cv::Point(separatorX, separatorBottom), COLOR_TEXT_SEC, 1, cv::LINE_AA);

        // AI Model / Hardware info
        int subAreaX = separatorX + separatorGap;
        int subAreaMaxW = mainX + mainW - rightPad - subAreaX;
        std::string aiModelText = "AI Model: Object Detection (" + displayModel + ")";
        std::string hardwareText = "Hardware: " + headerHardwareLabel(numDevices);
        int subFontH = std::max(12, (int)(cardH * 0.24));
        subFontH = std::min(fitHeaderFontHeight(aiModelText, subFontH, subAreaMaxW, HeaderFontWeight::SemiBold),
                            fitHeaderFontHeight(hardwareText, subFontH, subAreaMaxW, HeaderFontWeight::SemiBold));
        auto aiModelSz = headerTextSize(aiModelText, subFontH, HeaderFontWeight::SemiBold);
        auto hardwareSz = headerTextSize(hardwareText, subFontH, HeaderFontWeight::SemiBold);
        int subLineGap = std::max(5, (int)(cardH * 0.14));
        int subGroupH = aiModelSz.height + subLineGap + hardwareSz.height;
        int aiModelY = contentY - subGroupH / 2 + aiModelSz.height;
        int hardwareY = aiModelY + subLineGap + hardwareSz.height;

        drawHeaderText(s_hudCache, aiModelText, cv::Point(subAreaX, aiModelY),
                       subFontH, HeaderFontWeight::SemiBold, COLOR_TEXT_SEC);
        drawHeaderText(s_hudCache, hardwareText, cv::Point(subAreaX, hardwareY),
                       subFontH, HeaderFontWeight::SemiBold, COLOR_TEXT_SEC);

        // --- Metrics card: labels only (values drawn per-frame) ---
        int mContentX = metricsX + std::max(16, (int)(metricsW * 0.06));
        int mContentW = metricsW - std::max(32, (int)(metricsW * 0.12));

        int labelFontH = std::max(13, (int)(cardH * 0.27));
        int valueFontH = std::max(labelFontH + 4, (int)(cardH * 0.36 * fpsValueFontScale));
        int metricLabelValueGap = std::max(6, (int)(metricsW * 0.018));
        int metricItemGap = std::max(18, (int)(metricsW * 0.055));
        std::string totalValueSlotText = "9999.9";
        std::string avgValueSlotText = "99.9";
        auto totalLabelSz = headerTextSize("Total FPS", labelFontH, HeaderFontWeight::Bold);
        auto totalValueSlotSz = headerTextSize(totalValueSlotText, valueFontH, HeaderFontWeight::Bold);
        auto avgLabelSz = headerTextSize("AVG FPS", labelFontH, HeaderFontWeight::Bold);
        auto avgValueSlotSz = headerTextSize(avgValueSlotText, valueFontH, HeaderFontWeight::Bold);
        int totalValueSlotW = totalValueSlotSz.width;
        int avgValueSlotW = avgValueSlotSz.width;
        int metricsGroupW = totalLabelSz.width + metricLabelValueGap + totalValueSlotW + metricItemGap + avgLabelSz.width + metricLabelValueGap + avgValueSlotW;
        if(metricsGroupW > mContentW && metricsGroupW > 0) {
            double shrink = std::max(0.72, (double)mContentW / metricsGroupW);
            labelFontH = std::max(8, (int)(labelFontH * shrink));
            valueFontH = std::max(labelFontH + 2, (int)(valueFontH * shrink));
            totalLabelSz = headerTextSize("Total FPS", labelFontH, HeaderFontWeight::Bold);
            totalValueSlotSz = headerTextSize(totalValueSlotText, valueFontH, HeaderFontWeight::Bold);
            avgLabelSz = headerTextSize("AVG FPS", labelFontH, HeaderFontWeight::Bold);
            avgValueSlotSz = headerTextSize(avgValueSlotText, valueFontH, HeaderFontWeight::Bold);
            totalValueSlotW = totalValueSlotSz.width;
            avgValueSlotW = avgValueSlotSz.width;
            metricsGroupW = totalLabelSz.width + metricLabelValueGap + totalValueSlotW + metricItemGap + avgLabelSz.width + metricLabelValueGap + avgValueSlotW;
        }

        int metricX = mContentX + std::max(0, (mContentW - metricsGroupW) / 2);
        int metricsBaselineY = cardTop + (cardH + valueFontH) / 2 - std::max(1, (int)(cardH * 0.03));
        int labelBaselineY = metricsBaselineY - totalValueSlotSz.height + totalLabelSz.height;

        // Draw static labels into cache (top-aligned with FPS value text)
        drawHeaderText(s_hudCache, "Total FPS", cv::Point(metricX, labelBaselineY),
                       labelFontH, HeaderFontWeight::Bold, COLOR_TEXT_SEC);
        int totalSlotStartX = metricX + totalLabelSz.width + metricLabelValueGap;
        int avgLabelX = totalSlotStartX + totalValueSlotW + metricItemGap;
        drawHeaderText(s_hudCache, "AVG FPS", cv::Point(avgLabelX, labelBaselineY),
                       labelFontH, HeaderFontWeight::Bold, COLOR_TEXT_SEC);
        int avgSlotStartX = avgLabelX + avgLabelSz.width + metricLabelValueGap;

        // Save positions for per-frame value rendering
        s_totalValueSlotX = totalSlotStartX;
        s_avgValueSlotX = avgSlotStartX;
        s_totalValueSlotW = totalValueSlotW;
        s_avgValueSlotW = avgValueSlotW;
        s_metricsBaselineY = metricsBaselineY;
        s_valueFontH = valueFontH;
    }

    // --- Per-frame: copy cached HUD and draw dynamic FPS values ---
    if(boardW <= frame.cols && titleH <= frame.rows) {
        cv::Mat hudRoi = frame(cv::Rect(0, 0, boardW, titleH));
        s_hudCache.copyTo(hudRoi);
    }

    // Draw FPS values (throttled; static labels live in s_hudCache)
    static constexpr long long kFpsDisplayIntervalMs = 200;
    static std::string s_displayTotalText;
    static std::string s_displayAvgText;
    static int s_totalValueX = 0;
    static int s_avgValueX = 0;
    static bool s_haveDisplayFps = false;
    static auto s_lastFpsDisplayUpdate = std::chrono::steady_clock::time_point{};

    const cv::Scalar COLOR_FPS(20, 255, 32);
    bool hasFps = showFps && calcFps && activeStreams > 0;
    if(hasFps) {
        const auto nowDisp = std::chrono::steady_clock::now();
        const long long dtDisp = s_haveDisplayFps
            ? std::chrono::duration_cast<std::chrono::milliseconds>(nowDisp - s_lastFpsDisplayUpdate).count()
            : kFpsDisplayIntervalMs;
        if(!s_haveDisplayFps || dtDisp >= kFpsDisplayIntervalMs) {
            s_displayTotalText = fpsValueText(totalFps);
            s_displayAvgText = fpsValueText(totalFps / activeStreams);
            auto totalValueSz = headerTextSize(s_displayTotalText, s_valueFontH, HeaderFontWeight::Bold);
            auto avgValueSz = headerTextSize(s_displayAvgText, s_valueFontH, HeaderFontWeight::Bold);
            s_totalValueX = s_totalValueSlotX + s_totalValueSlotW - totalValueSz.width;
            s_avgValueX = s_avgValueSlotX + s_avgValueSlotW - avgValueSz.width;
            s_lastFpsDisplayUpdate = nowDisp;
            s_haveDisplayFps = true;
        }

        drawHeaderText(frame, s_displayTotalText, cv::Point(s_totalValueX, s_metricsBaselineY),
                       s_valueFontH, HeaderFontWeight::Bold, COLOR_FPS);
        drawHeaderText(frame, s_displayAvgText, cv::Point(s_avgValueX, s_metricsBaselineY),
                       s_valueFontH, HeaderFontWeight::Bold, COLOR_FPS);
    } else {
        s_haveDisplayFps = false;
    }
}


int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    std::string configPath = "";
    double frameCount = 0.0, window_size = 60.0;
    bool loggingVersion = false;
    bool showExitBtn = false;

    AppConfig appConfig;

    cxxopts::Options options("yolo_multi", "yolo multi channels application usage ");
    options.add_options()
        ("c, config", "(* required) use config json file for run application", cxxopts::value<std::string>(configPath))
        ("t, test", "test mode", cxxopts::value<bool>(loggingVersion)->default_value("false"))
        ("window_size", "FPS by average over the last {window_size} seconds (default: 60)", cxxopts::value<double>(window_size)->default_value("60"))
        ("exit-btn", "show a small exit button overlay in the top-right corner", cxxopts::value<bool>(showExitBtn)->default_value("false"))
        ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    g_showExitBtn = showExitBtn;
    if(cmd.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    if(configPath.empty())
    {
        std::cout << "error: no config json file arguments." << std::endl;
        std::cout << "Use -h or --help for usage information." << std::endl;
        exit(0);
    }

    if(ApplicationJsonParser(configPath, &appConfig) < 0)
    {
        std::cout << "error: failed to parse config json file." << std::endl;
        std::cout << "Use -h or --help for usage information." << std::endl;
        exit(0);
    }

    LOG_VALUE(configPath);

    const int BOARD_WIDTH = appConfig.board_width;
    const int BOARD_HEIGHT = appConfig.board_height;

    // Top title-bar height for the integrated HUD header
    const int HEADER_MIN_HEIGHT = 72;
    const int TITLE_HEIGHT = std::max(HEADER_MIN_HEIGHT, (int)(BOARD_HEIGHT * 0.084));
    const int GRID_WIDTH = BOARD_WIDTH;

    // Determine the grid size from JSON, or calculate it automatically using sqrt
    int grid_cols, grid_rows;
    if(appConfig.grid_cols > 0 && appConfig.grid_rows > 0)
    {
        grid_cols = appConfig.grid_cols;
        grid_rows = appConfig.grid_rows;
    }
    else
    {
        int n = (int)appConfig.video_sources.size();
        grid_cols = (int)ceil(sqrt((double)n));
        grid_rows = (grid_cols > 0) ? (int)ceil((double)n / grid_cols) : 1;
    }
    int divWidth  = GRID_WIDTH  / grid_cols;
    int divHeight = (BOARD_HEIGHT - TITLE_HEIGHT) / grid_rows;
    if(appConfig.is_expand_mode && appConfig.video_sources.size()!=33 && appConfig.video_sources.size()!=73 && appConfig.video_sources.size()!= 61 && appConfig.video_sources.size()!=41) {
        appConfig.is_expand_mode = false;
    }

    // Separator thickness in pixels between stream cells. Shrink each cell by GAP
    // and offset it by GAP/2 to expose the gray background (the initial outFrame
    // color) between adjacent cells. This is not redrawn every frame, so it adds
    // no per-frame runtime cost.
    const int SEPARATOR_GAP = std::max(2, BOARD_HEIGHT / 360);
    const int CELL_OFFSET   = SEPARATOR_GAP / 2;
    const int HEADER_PREVIEW_GAP = SEPARATOR_GAP;  // title bar ↔ preview (2× CELL_OFFSET)
    const cv::Scalar SEPARATOR_COLOR(0, 0, 0);

    cv::Mat outFrame = cv::Mat(cv::Size(BOARD_WIDTH, BOARD_HEIGHT), CV_8UC3, SEPARATOR_COLOR);
    // Leave the sidebar area unchanged because the sidebar fills it every frame.
    auto io = dxrt::InferenceOption();
    io.useORT = false;
    auto ie = std::make_shared<dxrt::InferenceEngine>(appConfig.model_path, io);
    if(!(dxdemo::common::minversionforRTandCompiler(ie.get()) || ie.get()->IsPPU()))
    {
        std::cerr << "[DXDEMO] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
        return -1;
    }
    yoloParam = getYoloParameter(appConfig.model_name);
    Yolo yolo = Yolo(yoloParam);
    std::vector<std::shared_ptr<ObjectDetection>> apps;
    uint64_t allFrameCount = 0;  // Use 64 bits to prevent overflow
    bool calcFps = false;

    // ---- Camera-highlight layout (independent of is_expand_mode) ----
    // If camera input exists, expand the first camera in the center area and show
    // a yellow border with a LIVE badge. Apply the border and badge to every camera.
    const int CAMERA_BORDER       = std::max(4, BOARD_HEIGHT / 240);  // Yellow border thickness
    const cv::Scalar CAMERA_COLOR(0, 255, 255);                        // BGR Yellow

    int cameraExpandIdx = -1;
    int cameraScale = 1;
    int cameraOriginCol = 0, cameraOriginRow = 0;
    std::vector<bool> cameraReservedCell(grid_cols * grid_rows, false);
    if(!appConfig.is_expand_mode)
    {
        for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
        {
            const std::string& srcType = appConfig.video_sources[i].second;
            if(srcType == "camera"
               || srcType == "camera_image"
               || srcType == "camera_video")
            {
                cameraExpandIdx = i;
                break;
            }
        }
        if(cameraExpandIdx >= 0)
        {
            // Keep the camera cell within 20% of the total grid area
            //   scale ≤ floor( sqrt(0.20 × cols × rows) )
            // Examples: 5x5 -> 2 (16%), 8x8 -> 3 (14%), 10x10 -> 4 (16%),
            // 16x16 -> 7 (19%), 30x30 -> 13 (18%), 100x100 -> 44 (19%),
            // and 4x4 -> 1 (no expansion)
            cameraScale = (int)std::floor(std::sqrt(0.20 * (double)grid_cols * grid_rows));
            if(cameraScale < 1) cameraScale = 1;

            int totalCells   = grid_cols * grid_rows;
            int othersCount  = (int)appConfig.video_sources.size() - 1;
            // Reduce the camera area until enough cells remain for the other streams
            while(cameraScale > 1
                  && othersCount + cameraScale * cameraScale > totalCells)
            {
                cameraScale--;
            }

            cameraOriginCol = (grid_cols - cameraScale) / 2;
            cameraOriginRow = (grid_rows - cameraScale) / 2;
            for(int r = 0; r < cameraScale; r++)
                for(int c = 0; c < cameraScale; c++)
                    cameraReservedCell[(cameraOriginRow + r) * grid_cols
                                        + (cameraOriginCol + c)] = true;
        }
    }

    if(appConfig.is_expand_mode)
    {
	    int position_index=0;
        int Window_scale = 2;
        if(appConfig.video_sources.size() == 41 || appConfig.video_sources.size() == 73){
            Window_scale = 3;
        }
        for(int i=0;i<(int)appConfig.video_sources.size(); i++)
        {
		if(appConfig.video_sources.size() == 33){
            if(i < 14){
				position_index = i;
			} else if(i < 18){
				position_index = i+2;
			} else if (i < 32) {
				position_index = i+4;
			} else {
				position_index = 14;
			}
        }else if(appConfig.video_sources.size() == 41){
            if(i < 16){
				position_index = i;
			} else if(i < 20){
				position_index = i+3;
			} else if (i < 24) {
				position_index = i+6;
			} else if (i < 40) {
				position_index = i+9;
			} else {
				position_index = 16;
			}
        }else if(appConfig.video_sources.size() == 73){
            if(i < 30){
				position_index = i;
			} else if(i < 36){
				position_index = i+3;
			} else if (i < 42) {
				position_index = i+6;
			} else if (i < 72) {
				position_index = i+9;
			} else {
				position_index = 30;
			}
        }else if(appConfig.video_sources.size() == 61){
		    if(i < 27){
                position_index = i;
            } else if(i < 33){
                    position_index = i+2;
            } else if (i < 60) {
                    position_index = i+4;
            } else {
                    position_index = 27;
            }
	}

	   if( i == (int)appConfig.video_sources.size() - 1){
                apps.emplace_back(
                    std::make_shared<ObjectDetection>(
                        ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                        divWidth*Window_scale - SEPARATOR_GAP, divHeight*Window_scale - SEPARATOR_GAP,
                        divWidth*(position_index%grid_cols) + CELL_OFFSET,
                        TITLE_HEIGHT + divHeight*(position_index/grid_cols) + HEADER_PREVIEW_GAP,
                        appConfig.pre_saved_frame_count[i]
                    )
                );
	   } else {
                apps.emplace_back(
                    std::make_shared<ObjectDetection>(
                        ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                        divWidth - SEPARATOR_GAP, divHeight - SEPARATOR_GAP,
                        divWidth*(position_index%grid_cols) + CELL_OFFSET,
                        TITLE_HEIGHT + divHeight*(position_index/grid_cols) + HEADER_PREVIEW_GAP,
                        appConfig.pre_saved_frame_count[i]
                    )
                );
	    }

            std::cout << *apps.back() << std::endl;
        }
    }else
    {
        // Standard layout, including camera highlighting
        auto cellRectFor = [&](int gridIdx, bool isCameraExpanded) {
            int cols = isCameraExpanded ? cameraScale : 1;
            int rows = isCameraExpanded ? cameraScale : 1;
            int col  = isCameraExpanded ? cameraOriginCol : (gridIdx % grid_cols);
            int row  = isCameraExpanded ? cameraOriginRow : (gridIdx / grid_cols);
            cv::Rect r;
            r.x = divWidth  * col + CELL_OFFSET;
            r.y = TITLE_HEIGHT + divHeight * row + HEADER_PREVIEW_GAP;
            r.width  = divWidth  * cols - SEPARATOR_GAP;
            r.height = divHeight * rows - SEPARATOR_GAP;
            return r;
        };

        // Index of the next unoccupied cell for a standard input
        int nextCellIdx = 0;
        auto advanceToFreeCell = [&]() {
            while(nextCellIdx < grid_cols * grid_rows
                  && cameraReservedCell[nextCellIdx])
                nextCellIdx++;
        };

        for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
        {
            const std::string& srcType  = appConfig.video_sources[i].second;
            const bool isCamera         = (srcType == "camera"
                                           || srcType == "camera_image"
                                           || srcType == "camera_video");
            const bool isCameraExpanded = (i == cameraExpandIdx);

            cv::Rect cell;
            if(isCameraExpanded)
            {
                cell = cellRectFor(0, true);
            }
            else
            {
                advanceToFreeCell();
                if(nextCellIdx >= grid_cols * grid_rows) break;
                cell = cellRectFor(nextCellIdx, false);
                nextCellIdx++;
            }

            // Inset camera input by the width of the yellow border
            int border = isCamera ? CAMERA_BORDER : 0;
            int destW = std::max(1, cell.width  - 2 * border);
            int destH = std::max(1, cell.height - 2 * border);
            int posX  = cell.x + border;
            int posY  = cell.y + border;

            // Yellow border: fill the entire cell area in outFrame with yellow once.
            // The stream ROI is copied inside it, leaving yellow only around the edge.
            if(isCamera)
            {
                cv::rectangle(outFrame, cell, CAMERA_COLOR, cv::FILLED);
            }

            apps.emplace_back(
                std::make_shared<ObjectDetection>(
                    ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                    destW, destH, posX, posY,
                    appConfig.pre_saved_frame_count[i]
                )
            );
            if(isCamera) apps.back()->SetLive(true);
            std::cout << *apps.back() << std::endl;
        }
        if(appConfig.is_fill_blank && !appConfig.is_expand_mode)
        {
            for(int i = (int)appConfig.video_sources.size();
                i < grid_cols * grid_rows; i++)
            {
                advanceToFreeCell();
                if(nextCellIdx >= grid_cols * grid_rows) break;
                cv::Rect cell = cellRectFor(nextCellIdx, false);
                nextCellIdx++;
                apps.emplace_back(
                    std::make_shared<ObjectDetection>(
                        ie, i,
                        cell.width, cell.height,
                        cell.x, cell.y
                    )
                );
            }
        }
    }


    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
        [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void* arg)
        {
            ObjectDetection *app = (ObjectDetection *)arg;
            app->PostProc(outputs);
            return 0;
        };
    ie->RegisterCallback(postProcCallBack);

#if !__riscv
    cv::namedWindow(DISPLAY_WINDOW_NAME, cv::WINDOW_NORMAL);
    if(appConfig.is_fullsize_mode)
    {
        cv::setWindowProperty(DISPLAY_WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    }
    else
    {
        cv::resizeWindow(DISPLAY_WINDOW_NAME, BOARD_WIDTH, BOARD_HEIGHT);
    }
    cv::moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
    cv::setMouseCallback(DISPLAY_WINDOW_NAME, onMouseCallback, nullptr);
#endif

    for(auto &app:apps)
    {
        app->Run(appConfig.input_capture_period_ms);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    /* Debugging */
    std::vector<cv::Rect> dstPoint = std::vector<cv::Rect>(apps.size(), cv::Rect(0, 0, 0, 0));
    for(int i = 0; i < (int)apps.size(); i++)
    {
        dstPoint[i].x = apps[i]->Position().first;
        dstPoint[i].y = apps[i]->Position().second;
        dstPoint[i].width = apps[i]->Resolution().first;
        dstPoint[i].height = apps[i]->Resolution().second;
    }
    dxdemo::common::StatusLog sl;
    sl.period = 500, sl.threadStatus.store(0);
    std::thread log_thread = std::thread(&dxdemo::common::logThreadFunction, &sl);
    auto start = std::chrono::high_resolution_clock::now();
    long long duration = 0;
    long long passTime = 0;
    std::deque<std::pair<std::chrono::time_point<std::chrono::high_resolution_clock>, uint64_t>> timestampedCounts;
    bool calcStarted = false;
    std::vector<uint64_t> lastProcessedCounts;

    // Toggle the CPU/NPU system monitor with the 'm' key. Disabled by default.
    bool showSysStats = false;

    while(true)
    {
        frameCount = 0.1;
        float resultFps = 0.f;

        for(int i = 0; i < (int)apps.size(); i++)
        {
            cv::Mat roi = outFrame(dstPoint[i]);
            apps[i]->ResultFrame().copyTo(roi);
        }

        allFrameCount++;

        if(calcFps)
        {
            uint64_t checkSum = 0;  // Use uint64_t instead of int to prevent overflow
            for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
            {
                uint64_t currentCount = apps[i]->GetPostProcessCount();
                if(calcStarted && i < (int)lastProcessedCounts.size())
                {
                    // Calculate only the difference from the previous measurement
                    uint64_t delta = (currentCount > lastProcessedCounts[i]) ?
                                    (currentCount - lastProcessedCounts[i]) : 0;
                    // Check for overflow
                    if(checkSum > UINT64_MAX - delta) {
                        std::cerr << "Warning: checkSum overflow detected, resetting..." << std::endl;
                        checkSum = delta;  // Use only the current delta on overflow
                    } else {
                        checkSum += delta;
                    }
                }
                // Store the current count for the next measurement
                if(i >= (int)lastProcessedCounts.size())
                    lastProcessedCounts.resize(i + 1);
                lastProcessedCounts[i] = currentCount;
            }

            // Store the frame count with the current timestamp
            auto now = std::chrono::high_resolution_clock::now();
            timestampedCounts.push_back({now, checkSum});

            // Remove data older than window_size seconds to prevent overflow
            if(window_size > 0 && window_size < LLONG_MAX / 1000) {
                auto cutoff = now - std::chrono::milliseconds(static_cast<long long>(window_size * 1000));
                while(!timestampedCounts.empty() && timestampedCounts.front().first < cutoff)
                {
                    timestampedCounts.pop_front();
                }
            }
        }


        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        if(passTime != -1) passTime = duration;
        if(passTime > 1000 && calcFps == false)
        {
            calcFps = true;
            calcStarted = true;
            passTime = 0;
            start = std::chrono::high_resolution_clock::now();
            // Store the initial count values
            lastProcessedCounts.resize(appConfig.video_sources.size());
            for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
            {
                lastProcessedCounts[i] = apps[i]->GetPostProcessCount();
            }
        }

        // Aggregate frame counts
        frameCount = 0.0;
        for(const auto& entry : timestampedCounts)
        {
            frameCount += entry.second;
        }

        if(calcFps && calcStarted)
        {
            if(!timestampedCounts.empty())
            {
                if(timestampedCounts.size() > 1)
                {
                    // Calculate the actual interval between the first and last timestamps
                    auto timeSpan = std::chrono::duration_cast<std::chrono::milliseconds>(
                        timestampedCounts.back().first - timestampedCounts.front().first).count();

                    if(timeSpan > 0)
                    {
                        // Calculate FPS using the actual time interval
                        resultFps = (frameCount * 1000.0) / timeSpan;
                    }
                    else
                    {
                        // Use only the current frame count when the interval is zero
                        resultFps = frameCount;
                    }
                }
                else
                {
                    // Handle a single sample
                    resultFps = frameCount;
                }
            }
            else
            {
                resultFps = 0.0f;
            }
        }

        renderHeaderHud(outFrame,
                        BOARD_WIDTH, TITLE_HEIGHT,
                        (int)appConfig.video_sources.size(),
                        appConfig.num_devices,
                        resultFps,
                        appConfig.is_show_fps,
                        calcFps,
                        appConfig.model_name,
                        appConfig.fps_value_font_scale);
        drawExitButtonOverlay(outFrame);
        sl.frameNumber = std::min(allFrameCount, (uint64_t)UINT_MAX);  // Prevent overflow
        sl.runningTime = duration;
        if (loggingVersion)
            sl.threadStatus.store(2);
        else
            sl.threadStatus.store(1);

        sl.statusCheckCV.notify_one();

#if __riscv
        std::cout << "press 'q' and enter to exit. " << std::endl;
        static bool launcherReadyNotified = false;
        if (!launcherReadyNotified)
        {
            notifyLauncherReady();
            launcherReadyNotified = true;
        }
        int key = getchar();
#else
        cv::imshow(DISPLAY_WINDOW_NAME, outFrame);

        int key = cv::waitKey(1);
        static bool launcherReadyNotified = false;
        if (!launcherReadyNotified)
        {
            notifyLauncherReady();
            launcherReadyNotified = true;
        }
#endif
        if(key == 0x1B || key == 0x71 || g_exitRequested) //'ESC' or 'q' or EXIT button
        {
            sl.threadStatus.store(-1);
            for(auto &app:apps)
            {
                app->Stop();
            }
            log_thread.join();
            break;
        }
        else if(key == 0x74) // 't'
        {
            for(auto &app:apps)
            {
                app->Toggle();
            }
        }
        else if(key == 0x6D) // 'm': toggle CPU LOAD / NPU TEMP display
        {
            showSysStats = !showSysStats;
        }

    }
#ifdef __linux__
    sleep(1);
#elif _WIN32
    Sleep(1000);
#endif
DXRT_TRY_CATCH_END
    return 0;
}
