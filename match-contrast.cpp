#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// g++ -o match-contrast match-contrast.cpp `pkg-config --libs --static opencv4`

using namespace std;
using namespace cv;

Scalar CalcMSSIM(Mat  inputimage1, Mat inputimage2)
{
    // 检查图像尺寸并调整大小
    if (inputimage1.size() != inputimage2.size())
    {
        if (inputimage1.size().area() > inputimage2.size().area())
        {
            resize(inputimage1, inputimage1, inputimage2.size());
        }
        else
        {
            resize(inputimage2, inputimage2, inputimage1.size());
        }
    }
    Mat i1 = inputimage1;
    Mat i2 = inputimage2;
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);
    return mssim;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./template_matching_video <template_image_path> <video_path> [phash|hist|ssim]" << std::endl;
        return -1;
    }

    // cv::Mat templ = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::Mat mask;
    std::string algorithm = "ssim";
    if (argc == 4) {
      algorithm = std::string(argv[4]);
    }
    std::cout << "algorithm is :" << algorithm << std::endl;

    if (templ.empty()) {
        std::cerr << "Error: Unable to read template image" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat result;

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        // 使用灰度图进行模板匹配时，耗时0.07 秒，同样的数据采用彩图进行匹配需要花费0.21s,相差了3倍
        cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像
        cv::matchTemplate(frame, templ, result, cv::TM_CCOEFF_NORMED);
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
	      ss << "  (min) " << minVal << std::endl;
        if (maxVal > 0.4)
        {
          cv::rectangle(frame, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);
        }

        cv::Rect rectangle(matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ));
        cv::Mat matchedImage = frame(rectangle);
        Scalar result = CalcMSSIM(templ, matchedImage);
        std::stringstream ss1;
        ss1 << "SSIM: " << result << std::endl;
        cv::putText(frame, ss1.str(), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // 调整显示图像框大小
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

	      std::cout << ss.str() << std::endl;

        // Print maxVal on the image
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Result", frame);

        if (cv::waitKey(30) == 27) {  // Press ESC to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

