#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, "{@input | | input image}");
    string inputImage = parser.get<string>("@input");

    // 检查图像路径是否已经提供
    if (inputImage.empty()) {
        cerr << "Error: No input image provided." << endl;
        return -1;
    }

    // 读取输入图像
    Mat src, equ, clahe;
    src = imread(inputImage, IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Error: Could not open or find the image." << endl;
        return -1;
    }

    // 自适应直方图均衡化（AHE）应用于亮度通道
    equalizeHist(src, equ);

    // 限制对比度自适应直方图均衡（CLAHE）应用于亮度通道
    Ptr<CLAHE> clahePtr = createCLAHE();
    clahePtr->setClipLimit(4.0); // 设置对比度限制
    clahePtr->setTilesGridSize(Size(8, 8)); // 设置tile grid size
    clahePtr->apply(src, clahe); // 注意这里我们使用equYChannel作为CLAHE的输出


    // 显示原始图像、AHE图像和CLAHE图像
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", src);

    namedWindow("AHE", WINDOW_AUTOSIZE);
    imshow("AHE", equ);

    namedWindow("CLAHE", WINDOW_AUTOSIZE);
    imshow("CLAHE", clahe);

    // 等待用户按键
    waitKey(0);

    return 0;
}
