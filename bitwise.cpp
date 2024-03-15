#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace std;
using namespace cv;

// 图片融合
void blendImages(cv::Mat& baseImg, cv::Mat& maskImg)
{
    if (baseImg.size() != maskImg.size()) return;

    if (maskImg.channels() > 1) {
        cv::threshold(maskImg, maskImg, 200, 255, cv::THRESH_BINARY);
    }

    for (int y = 0; y < baseImg.rows; ++y)
    {
        uchar* maskPtr = maskImg.ptr<uchar>(y);
        cv::Vec3b* basePtr = baseImg.ptr<cv::Vec3b>(y);

        for (int x = 0; x < baseImg.cols; ++x)
        {
            uchar pixel = maskPtr[x];

            // 检查像素值是否为黑色
            if (pixel == 0)
            {
                basePtr[x] = cv::Vec3b(255, 255, 255); // 将对应位置设为白色
            }
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./bitwise_and_example image1.jpg image2.jpg" << std::endl;
        return -1;
    }

    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);

    if (image1.empty() || image2.empty()) {
        std::cout << "Error loading images." << std::endl;
        return -1;
    }

    cv::Mat result;
    cv::bitwise_not(image2, image2);

    // 1
    auto start = std::chrono::high_resolution_clock::now();
    cv::add(image1, image2, result);
    auto end = std::chrono::high_resolution_clock::now();

    // 计算执行时间
    std::chrono::duration<double> duration = end - start;
    std::cout << "函数执行时间: " << duration.count() << " 秒" << std::endl;

    // 2
    cv::bitwise_not(image2, image2);
    cv::cvtColor(image2, image2, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像
    start = std::chrono::high_resolution_clock::now();
    blendImages(image1, image2);
    end = std::chrono::high_resolution_clock::now();

    /*
    *   cv::add     执行时间 0.0021
    *   blendImages 执行时间 0.0044
    */
    // 计算执行时间
    duration = end - start;
    std::cout << "函数执行时间: " << duration.count() << " 秒" << std::endl;

    cv::imshow("Result", result);
    cv::imshow("image1", image1);
    cv::waitKey(0);

    return 0;
}
