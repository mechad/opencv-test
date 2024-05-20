#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <stdexcept>
#include <cstdlib>

double StringToDouble(const std::string& str) {
    try {
        return std::stod(str);
    } catch (const std::invalid_argument& ex) {
        std::cerr << "Invalid argument: " << ex.what() << std::endl;
    } catch (const std::out_of_range& ex) {
        std::cerr << "Out of range: " << ex.what() << std::endl;
    }
    return 0.0; // 或者你可以选择其他方式处理错误
}

int main(int argc, char* argv[]) {
    // 检查是否提供了图像路径
    if (argc < 3) {
        std::cout << "Usage: DisplayImage <Image_Path> <scale>" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // 读取图像为灰度图

    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // 将图像缩放到原来的十分之一大小
    cv::Mat smallImage;
    double scale = StringToDouble(argv[2]); // 缩放比例
    cv::resize(image, smallImage, cv::Size(), scale, scale, cv::INTER_LINEAR);

    // 创建结构元素
    cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 应用顶帽操作
    cv::Mat topHat;
    cv::morphologyEx(smallImage, topHat, cv::MORPH_TOPHAT, se);

    // 显示缩放后的原图和顶帽结果
    cv::imshow("Resized Original Image", smallImage);
    cv::imshow("Top Hat Image", topHat);

    // 等待用户按键，再退出
    cv::waitKey(0);
    return 0;
}
