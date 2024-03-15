#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// 将小图片融合到大图片中
void blendImages(Mat& largeImage, Mat& smallImage, int startX, int startY)
{
    for (int y = 0; y < largeImage.rows; ++y)
    {
        for (int x = 0; x < largeImage.cols; ++x)
        {
            // Vec3b pixel = smallImage.at<Vec3b>(y, x); // 读取 RGB 像素值
            uchar pixel = smallImage.at<uchar>(y, x);

            // 检查像素值是否为黑色
            if (pixel == 0)
            //if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
            {
                largeImage.at<Vec3b>(startY + y, startX + x) = Vec3b(0, 0, 0); // 将对应位置设为黑色
            }
        }
    }
}


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
    cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);

    // 确保两张图片大小相同
    if (image1.size() != image2.size()) {
        cv::resize(image2, image2, cv::Size(image1.cols, image1.rows));
    }
        cv::resize(image1, image1, cv::Size(640, 480));
        cv::resize(image2, image2, cv::Size(640, 480));
    // // 遍历图像的每个像素，并将每个像素设置为白色
    // for (int y = 0; y < image1.rows; ++y) {
    //     for (int x = 0; x < image1.cols; ++x) {
    //         image1.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255); // 设置像素为白色
    //     }
    // }

    if (image1.empty() || image2.empty()) {
        std::cout << "Error: Unable to read one or both images." << std::endl;
        return -1;
    }

    // cv::imshow("temp", image2);
    // 将二值图像转换为掩码
    cv::Mat mask_white, mask_black;
    cv::threshold(image2, mask_white, 200, 255, cv::THRESH_BINARY); // 白色部分作为掩码

    // 确保两张图片大小相同
    // cv::resize(image2, image2, cv::Size(image1.cols, image1.rows));

    // 将两张图片叠加在一起
    cv::Mat result;
    // cv::addWeighted(image1, 0.3, image2, 0.7, 0, result);
    blendImages(image1, mask_white, 0, 0);
    // cv::imshow("mask_white", mask_white);

    // 显示叠加后的结果
    cv::imshow("Result", image1);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

