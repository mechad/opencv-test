#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <zbar.h>

// g++ -o match-contrast match-contrast.cpp `pkg-config --libs --static opencv4`

using namespace std;
using namespace cv;
using namespace zbar;

ImageScanner scanner;

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


std::vector<std::string> barcodeRecognition(const cv::Mat &image)
{
    std::vector<std::string> barcodeData; // 存储条码内容的向量
    // 创建缓冲区
    uchar* buffer = new uchar[image.cols * image.rows];

    if (image.channels() > 1) cv::cvtColor(image, image, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像

    // 复制ROI区域的数据到缓冲区
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            buffer[i * image.cols + j] = image.at<uchar>(i, j);
        }
    }
    // 将图像数据传递给ZBar
    Image zbar_image(image.cols, image.rows, "Y800", buffer, image.cols * image.rows);

    // 扫描图像中的条码
    scanner.scan(zbar_image);
    // 遍历识别到的条码
    for (Image::SymbolIterator symbol = zbar_image.symbol_begin(); symbol != zbar_image.symbol_end(); ++symbol)
    {
        // 获取条码类型和内容
        // std::string barcodeType = symbol->get_type_name();
        std::string barcodeContent = symbol->get_data();
        // 存储条码内容
        barcodeData.push_back(barcodeContent);
    }
    // 释放ZBar图像
    zbar_image.set_data(NULL, 0);

    delete[] buffer;
    // 返回条码内容的向量
    return barcodeData;
}

void addText(cv::Mat& image, const std::string& text, const cv::Point& org) {
    cv::Scalar fontColor(255, 255, 255);
    int thickness = 2;
    cv::putText(image, text, org, cv::FONT_HERSHEY_SIMPLEX, 1.0, fontColor, thickness);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./template_matching_video <template_image_path> <video_path>" << std::endl;
        return -1;
    }

    // cv::Mat templ = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
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
    int total = 0;
    int barcnt = 0;

    // 创建ZBar解码器
    int barcodeTypes = ZBAR_CODE128; // 只识别一个即可 ZBAR_EAN13 | ZBAR_CODE39 | ZBAR_CODE128;

    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, barcodeTypes);
    while (cap.read(frame)) {
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像
        cv::matchTemplate(grayFrame, templ, result, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        cv::Point matchLoc = maxLoc;

        // Convert maxVal to string
        std::stringstream ss;
        ss << "Similarity: (max)" << maxVal;
	    ss << "  (min) " << minVal << std::endl;
        std::cout << ss.str() ;

        if (maxVal > 0.4)
        {
            cv::Rect rectangle(matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ));
            cv::Mat matchedImage = grayFrame(rectangle);

            cv::rectangle(frame, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);
            cv::Rect barcodeRect;
            barcodeRect.x = 680;
            barcodeRect.y = 200;
            barcodeRect.width = 100;
            barcodeRect.height = 500;
            
            cv::Point matchStart(matchLoc.x + barcodeRect.x, matchLoc.y + barcodeRect.y);
            cv::Point matchEnd(matchStart.x + barcodeRect.width, matchStart.y + barcodeRect.height);
            cv::rectangle( frame, matchStart, matchEnd, cv::Scalar(0,255,0), 2, 8, 0 );
            // 旋转图像
            cv::Mat rotatedImg = matchedImage(barcodeRect);
            cv::rotate(rotatedImg, rotatedImg, cv::ROTATE_90_CLOCKWISE);
            
            auto start = std::chrono::high_resolution_clock::now();

            mythreshold3(rotatedImg, 200); // 识别率99.52%, 耗时：0.0040s
            // mythreshold2(rotatedImg, 200); // 识别率79.90%
            // mythreshold(rotatedImg, 150); // 识别率99.52%, 耗时：0.0026s
            // cv::threshold(rotatedImg, rotatedImg, 140, 255, cv::THRESH_BINARY);

            std::vector<std::string> barcodeResults = barcodeRecognition(rotatedImg);
            
            auto end = std::chrono::high_resolution_clock::now();

            // 计算执行时间
            std::chrono::duration<double> duration = end - start;
            std::cout << "函数执行时间: " << duration.count() << " 秒" << std::endl;
            cv::imshow("barcode", rotatedImg);
            // 遍历打印结果向量中的条码内容
            for (const std::string& barcode : barcodeResults) {
                std::cout << "条码内容：" << barcode << std::endl;
                addText(frame, barcode ,matchLoc);
                barcnt ++;
            }
        }
        total ++;

        cv::Rect rectangle(matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ));
        cv::Mat matchedImage = frame(rectangle);

        // 调整显示图像框大小
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));

        cv::imshow("Result", frame);

        if (cv::waitKey(30) == 27) {  // Press ESC to exit
            break;
        }
    }

    double recognitionRate = static_cast<double>(barcnt) / total * 100;

    std::cout << "条码识别率： " << barcnt << "/" << total << " = " << std::fixed << std::setprecision(2) << recognitionRate << std::endl;
    cap.release();
    cv::destroyAllWindows();

    return 0;
}

