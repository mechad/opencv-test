#include <opencv2/opencv.hpp>
#include <iostream>

// g++ -o hist cv_hist.cpp `pkg-config --libs --static opencv4`
//
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./histogram_equalization <image_path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR); // 从命令行参数获取图像路径并读取彩色图像

    if (img.empty()) {
        std::cerr << "Error: Unable to read image" << std::endl;
        return -1;
    }

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY); // 转换为灰度图像

    cv::Mat blurred_img;
    cv::GaussianBlur(img_gray, blurred_img, cv::Size(5, 5), 0); // 进行高斯模糊，指定卷积核大小为 5x5

    cv::Mat binary_img;
    cv::adaptiveThreshold(img_gray, binary_img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, 5);

//------------------------------------------------------------------------
    // 定义不同形状和大小的结构元素
    cv::Mat element_rect = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(4, 4));
    cv::Mat element_ellipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // 先进行侵蚀操作
    cv::Mat eroded_img_rect, eroded_img_ellipse;
    cv::erode(binary_img, eroded_img_rect, element_rect);
    cv::erode(binary_img, eroded_img_ellipse, element_ellipse);

    // 再进行膨胀操作
    cv::Mat dilated_img_rect, dilated_img_ellipse;
    cv::dilate(eroded_img_rect, dilated_img_rect, element_rect);
    cv::dilate(eroded_img_ellipse, dilated_img_ellipse, element_ellipse);
//--------------------------------------------------------------------------

    cv::Mat equalized_img;
    cv::equalizeHist(img_gray, equalized_img); // 应用直方图均衡化

    cv::Mat combined_img;
    cv::vconcat(img_gray, blurred_img, combined_img); // 将两个图像水平拼接在一起
    cv::Mat combined_img2;
    cv::vconcat(binary_img, dilated_img_rect, combined_img2); // 将两个图像水平拼接在一起
    cv::hconcat(combined_img, combined_img2, combined_img); // 将两个图像水平拼接在一起
    // cv::hconcat(img_gray, equalized_img, combined_img); // 将两个图像水平拼接在一起

    // 调整显示图像框大小
    cv::resize(combined_img, combined_img, cv::Size(combined_img.cols/2, combined_img.rows/2));
    cv::imshow("Original Image vs Equalized Image", combined_img);
    cv::waitKey(0);

    return 0;
}

