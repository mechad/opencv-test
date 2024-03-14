#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// g++ -o match-contrast match-contrast.cpp `pkg-config --libs --static opencv4`

using namespace std;
using namespace cv;

// 将小图片融合到大图片中
void blendImages(Mat& largeImage, Mat& smallImage, int startX, int startY)
{
    // 检查小图片尺寸是否超过大图片尺寸
    if (smallImage.cols > largeImage.cols - startX || smallImage.rows > largeImage.rows - startY)
    {
        // 计算缩放比例
        double scale = min((double)(largeImage.cols - startX) / smallImage.cols, (double)(largeImage.rows - startY) / smallImage.rows);

        // 等比例缩放小图片
        resize(smallImage, smallImage, Size(), scale, scale);
    }

    // 创建感兴趣区域（ROI）并将小图片放置在该区域内
    Mat roi = largeImage(Rect(startX, startY, smallImage.cols, smallImage.rows));
/*
		// 定义融合权重
    // 使用融合函数，背景颜色不会透明
		double alpha = 0.5; // 第一张图片的权重
		double beta = 0.5; // 第二张图片的权重
		double gamma = 0.0; // 亮度调整参数

	  // 将两张图片融合
		Mat blendedImage;
		addWeighted(roi, alpha, smallImage, beta, gamma, roi);
*/
		// 将小图片融合到大图片中（带有透明度通道）
    // 黑色背景/白色画面，将黑色背景设置为透明效果
    for (int y = 0; y < smallImage.rows; ++y)
    {
        for (int x = 0; x < smallImage.cols; ++x)
        {
						uchar pixel = smallImage.at<uchar>(y, x);
            if (pixel > 0) // 检查像素值是否大于0
            {
                roi.at<uchar>(y, x) = pixel;
            }
        }
    }
}

pair<Mat, Mat> plotHistogram(const Mat& image, int histSize, const float* histRange)
{
    Mat hist;
    calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
             Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
             Scalar(255), 2, 8, 0);
    }

    return make_pair(histImage, hist);
}

void mythreshold(cv::Mat &frame, int threshold)
{
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (frame.at<uchar>(i, j) > threshold)
            {
                frame.at<uchar>(i, j) = 255;
            }
        }
    }
}

void mythreshold2(cv::Mat &frame, int threshold)
{
    int neighborhood_size = 3; // 可以设置为3或5，表示相邻像素的邻域大小
    int diff_threshold = 50; // 设定的阈值差异

    cv::Mat binary_frame = cv::Mat::zeros(frame.size(), CV_8UC1);

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            int count = 0;
            int sum = 0;

            for (int k = i - neighborhood_size/2; k <= i + neighborhood_size/2; k++)
            {
                for (int l = j - neighborhood_size/2; l <= j + neighborhood_size/2; l++)
                {
                    if (k >= 0 && k < frame.rows && l >= 0 && l < frame.cols)
                    {
                        sum += frame.at<uchar>(k, l);
                        count++;
                    }
                }
            }

            int avg = sum / count;

            if (abs(threshold - avg) < diff_threshold)
            {
                binary_frame.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::imshow("bin", binary_frame);
    // 将原图与二值图进行与操作
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (binary_frame.at<uchar>(i, j) == 255 ) frame.at<uchar>(i, j) = 255;
        }
    }
}

void mythreshold3(cv::Mat &frame, int threshold)
{
    cv::Mat edges;
    cv::Canny(frame, edges, threshold, threshold * 2); // 使用Canny边缘检测算法

    cv::Mat binary_frame = cv::Mat::zeros(frame.size(), CV_8UC1);

    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (edges.at<uchar>(i, j) == 0) // 边缘检测结果为0表示平坦区域
            {
                binary_frame.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::imshow("bin", edges);
    // 将原图与二值图进行与操作
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (binary_frame.at<uchar>(i, j) == 255 && (abs(threshold - frame.at<uchar>(i, j)) < 50))
            {
                frame.at<uchar>(i, j) = 255;
            }
        }
    }
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

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    pair<Mat, Mat> result1 = plotHistogram(templ, histSize, histRange);

    Mat histImage1 = result1.first;
    Mat histData1 = result1.second;

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        // 使用灰度图进行模板匹配时，耗时0.07 秒，同样的数据采用彩图进行匹配需要花费0.21s,相差了3倍
        cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像
        // equalizeHist(frame, frame);
        // mythreshold3(frame, 188);
        cv::Mat edges;
        cv::Canny(frame, edges, 188, 188 * 2); // 使用Canny边缘检测算法
        int non_zero_pixels = cv::countNonZero(edges); // 统计非0像素的个数
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

        pair<Mat, Mat> result2 = plotHistogram(matchedImage, histSize, histRange);

				Mat histImage2 = result2.first;
				Mat histData2 = result2.second;

        // 比较直方图
        double ret = compareHist(histData1, histData2, 0);
        std::cout << "Method [0] result = " << ret << std::endl;

        cv::Mat combined_img;
        cv::vconcat(histImage1, histImage2, combined_img); // 将两个图像水平拼接在一起

				// 定义融合位置
				int startX = 100; // 融合起始位置的x坐标
				int startY = 100; // 融合起始位置的y坐标
				blendImages(frame, combined_img, startX, startY);

/****************************************/
        // 调整显示图像框大小
        // cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

	    std::cout << ss.str() << std::endl;
        std::cout << "Number of non-zero pixels: " << non_zero_pixels << std::endl;
        // Print maxVal on the image
        // cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Result", frame);

        if (cv::waitKey(30) == 27) {  // Press ESC to exit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

