#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// g++ -o match-contrast match-contrast.cpp `pkg-config --libs --static opencv4`

using namespace std;
using namespace cv;

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

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat hist_1, hist_2;

    calcHist(&templ, 1, 0, Mat(), hist_1, 1, &histSize, &histRange, uniform, accumulate);
    normalize(hist_1, hist_1, 0, 1, NORM_MINMAX, -1, Mat());
    // 绘制直方图
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage1(hist_h, hist_w, CV_8UC1, Scalar(255));
    normalize(hist_1, hist_1, 0, histImage1.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage1, Point(bin_w * (i - 1), hist_h - cvRound(hist_1.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(hist_1.at<float>(i))),
             Scalar(0), 2, 8, 0);
    }

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
        
/****************************************/

        calcHist(&matchedImage, 1, 0, Mat(), hist_2, 1, &histSize, &histRange, uniform, accumulate);
        normalize(hist_2, hist_2, 0, 1, NORM_MINMAX, -1, Mat());

        // 比较直方图
        double result = compareHist(hist_1, hist_2, 0);
        std::cout << "Method [0] result = " << result << std::endl;

        Mat histImage2(hist_h, hist_w, CV_8UC1, Scalar(255));
        normalize(hist_2, hist_2, 0, histImage2.rows, NORM_MINMAX, -1, Mat());

        for (int i = 1; i < histSize; i++)
        {
            line(histImage2, Point(bin_w * (i - 1), hist_h - cvRound(hist_2.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(hist_2.at<float>(i))),
                 Scalar(0), 2, 8, 0);
        }

        cv::Mat combined_img;
        cv::vconcat(histImage2, histImage1, combined_img); // 将两个图像水平拼接在一起

        // 显示直方图
        namedWindow("Histogram", WINDOW_AUTOSIZE);
        imshow("Histogram", combined_img);
   
/****************************************/
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

