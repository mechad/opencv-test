#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    if (argc < 3) {
        std::cerr << "Usage: ./" << argv[0] << " image1 image2 " << std::endl;
        return -1;
    }

    Mat src_1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat src_2 = imread(argv[2], IMREAD_GRAYSCALE);

    if (src_1.size() != src_2.size())
    {
        if (src_1.size().area() > src_2.size().area())
        {
            resize(src_1, src_1, src_2.size());
        }
        else
        {
            resize(src_2, src_2, src_1.size());
        }
    }

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    Mat hist_1, hist_2;

    calcHist(&src_1, 1, 0, Mat(), hist_1, 1, &histSize, &histRange, uniform, accumulate);
    normalize(hist_1, hist_1, 0, 1, NORM_MINMAX, -1, Mat());

    calcHist(&src_2, 1, 0, Mat(), hist_2, 1, &histSize, &histRange, uniform, accumulate);
    normalize(hist_2, hist_2, 0, 1, NORM_MINMAX, -1, Mat());

    // 比较直方图
    for (int i = 0; i < 4; i++)
    {
        int compare_method = i;
        double result = compareHist(hist_1, hist_2, compare_method);
        printf("Method [%d] result = %f \n", i, result);
    }

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
    waitKey(0);

    return 0;
}

