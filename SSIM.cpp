
// 结构性相似度 SSIM
// 计算两幅图像之间的结构相似性指数（SSIM）。SSIM是一种用于衡量两幅图像相似程度的指标，它考虑了亮度、对比度和结构三个方面的差异。以下是您提供的代码的简要说明：
// CalcMSSIM函数：该函数用于计算两幅输入图像之间的结构相似性指数。它首先对输入图像进行预处理，包括转换数据类型、计算均值、方差等。然后根据SSIM的公式计算结构相似性指数，并返回结果。

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

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

int main( int argc, char** argv )
{
  if (argc < 3) {
      std::cerr << "Usage: ./" << argv[0] << " image1 image2 " << std::endl;
      return -1;
  }

  Mat src_1 = imread(argv[1]);
  Mat src_2 = imread(argv[2]);
  Scalar result = CalcMSSIM(src_1,src_2);
  cout<<"the r g b channles similarity is :"<<result<<endl;
}
