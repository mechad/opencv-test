#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// g++ -o templ template.cpp `pkg-config --libs --static opencv4`

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./template_matching_video <template_image_path> <video_path> [mask_image_path]" << std::endl;
        return -1;
    }

    // cv::Mat templ = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    cv::Mat mask;
    if (argc == 4) {
       mask = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
       if (mask.rows != templ.rows || mask.cols != templ.cols) {
         std::cout << "mask.size = " << mask.rows << "x" << mask.cols << std::endl;
         std::cout << "templ.size = " << templ.rows << "x" << templ.cols << std::endl;
	 return 1;
       }
       cv::cvtColor(templ, templ, cv::COLOR_RGB2GRAY); //将原图转换为灰度图像
    }

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
	if (argc == 4) {
	    // 使用mask 时，图像格式必须一致，也就是都要灰度图
	    // 只有 TM_CCORR_NORMED 模式下使用 mask 掩码模板成功
	    cv::matchTemplate(frame, templ, result, cv::TM_CCORR_NORMED, mask);
	} else {
	    cv::matchTemplate(frame, templ, result, cv::TM_CCOEFF_NORMED);
	}
	auto end = std::chrono::high_resolution_clock::now();
	// 计算执行时间
	std::chrono::duration<double> duration = end - start;
	std::cout << "函数执行时间: " << duration.count() << " 秒" << std::endl;

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        cv::Point matchLoc = maxLoc;

	if (maxVal > 0.4)
        cv::rectangle(frame, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);

        // 调整显示图像框大小
        cv::resize(frame, frame, cv::Size(frame.cols/2, frame.rows/2));
        // Convert maxVal to string
        std::stringstream ss;
        ss << "Similarity: " << maxVal;
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

