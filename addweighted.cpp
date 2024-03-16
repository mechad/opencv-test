#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./program_name image1_path image2_path" << std::endl;
        return -1;
    }

    // 从命令行参数获取图片路径
    std::string image1_path = argv[1];
    std::string image2_path = argv[2];

    // 读取两张图片
    cv::Mat image1 = cv::imread(image1_path);
    cv::Mat image2 = cv::imread(image2_path);

    if (image1.empty() || image2.empty()) {
        std::cout << "Error: Unable to read one or both images." << std::endl;
        return -1;
    }

    // 确保两张图片大小相同
    cv::resize(image2, image2, cv::Size(image1.cols, image1.rows));

    // 将两张图片叠加在一起
    cv::Mat result;
    cv::addWeighted(image1, 0.9, image2, 0.3, 0, result);

    // 显示叠加后的结果
    cv::imshow("Result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

