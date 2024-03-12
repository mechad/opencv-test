#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// g++ -o templ template.cpp `pkg-config --libs --static opencv4`

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./template_matching_video template image2" << std::endl;
        return -1;
    }

    // cv::Mat templ = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    // 检查图像尺寸并调整大小
    {
        if (templ.size().area() > img2.size().area())
        {
            resize(templ, templ, img2.size());
        }
    }

    cv::Mat result;

    auto start = std::chrono::high_resolution_clock::now();
    // 匹配的准确度，全彩图匹配成功率略高
    cv::matchTemplate(img2, templ, result, cv::TM_CCOEFF_NORMED);

    auto end = std::chrono::high_resolution_clock::now();
    // 计算执行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "函数执行时间: " << duration.count() << " 秒" << std::endl;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point matchLoc = maxLoc;

    // Convert maxVal to string
    std::stringstream ss;
    ss << "Similarity: (max)" << maxVal;
    ss << "  (min) " << minVal;
    std::cout << ss.str() << std::endl;

    return 0;
}

