#include <opencv2/opencv.hpp>

bool isFlatRegion(const cv::Mat& image)
{
    cv::Rect roi(image.cols / 3, image.rows / 3, image.cols / 3, image.rows / 3); // 定义中心三分之一尺寸大小的区域
    cv::Mat croppedImage = image(roi); // 获取中心区域图像

    cv::Mat grayImage;
    cv::cvtColor(croppedImage, grayImage, cv::COLOR_BGR2GRAY); // 将中心区域图像转换为灰度图像

    cv::Mat edges;
    cv::Canny(grayImage, edges, 100, 200); // 使用Canny边缘检测算法

    cv::namedWindow("edges", cv::WINDOW_AUTOSIZE);
    cv::imshow("edges", edges);
    cv::waitKey(0);

    int nonZeroCount = cv::countNonZero(edges); // 统计非零像素的个数
    std::cout << "nonZeroCount: " << nonZeroCount << std::endl;

    // 如果非零像素个数很少，则认为中心区域是平坦区域
    return nonZeroCount < 100; // 这里可以根据实际情况调整阈值
}

int main( int argc, char** argv )
{
    if (argc < 2) {
        std::cerr << "Usage: ./" << argv[0] << " image1" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]); // 读取图像

    if (image.empty())
    {
        std::cout << "Error: Could not read the image file." << std::endl;
        return -1;
    }

    if (isFlatRegion(image))
    {
        std::cout << "The center region of the image is a flat region." << std::endl;
    }
    else
    {
        std::cout << "The center region of the image is not a flat region." << std::endl;
    }

    return 0;
}

