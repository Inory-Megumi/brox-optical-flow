//
// Created by karthik on 06/12/20.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "opticalFlow.h"

using namespace cv;

#define DEFAULT_ALPHA 18
#define DEFAULT_GAMMA 7
#define DEFAULT_PYRAMID_LEVEL 10
#define DEFAULT_PYRAMID_FACTOR 0.5
#define TOLERANCE 0.0001

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "usage: broxflow <image 1 path> <image  path>\n";
        return -1;
    }

    Mat uchar_image1, uchar_image2;
    uchar_image1 = imread(argv[1], IMREAD_GRAYSCALE);
    uchar_image2 = imread(argv[2], IMREAD_GRAYSCALE);
    if (!uchar_image1.data) {
        std::cout << "Image 1 has no data!\n";
        return -1;
    }
    if (!uchar_image2.data) {
        std::cout << "Image 2 has no data!\n";
        return -1;
    }
    if (uchar_image1.rows != uchar_image2.rows || uchar_image1.cols != uchar_image2.cols) {
        std::cout << "Image sizes do not match!\n";
        return -1;
    }

    Mat_<float> I1, I2;
    uchar_image1.convertTo(I1, CV_32F);
    uchar_image2.convertTo(I2, CV_32F);

    float alpha = DEFAULT_ALPHA;
    float gamma = DEFAULT_GAMMA;
    int pyramidLevel = DEFAULT_PYRAMID_LEVEL;
    float pyramidFactor = DEFAULT_PYRAMID_FACTOR;
    float tolerance = TOLERANCE;

    int N = 1 + log(std::min(I1.cols, I1.rows) / 16.0f) / log(1.0f / pyramidFactor);
    if (N < pyramidLevel)pyramidLevel = N;
//    fimage.at<float>(10,50) -= 4.1;
//    std::cout<<fimage.at<float>(10,50)<<"\n";
//    std::cout<<(int)image.at<uchar>(10,50)<<"\n";
//    std::cout<<fimage.cols<<" "<<fimage.rows<<"\n";
//    std::cout<<image.cols<<" "<<image.rows<<"\n";

    Mat_<float> u = Mat::zeros(I1.rows, I1.cols, CV_32F);
    Mat_<float> v = Mat::zeros(I1.rows, I1.cols, CV_32F);

    calculateOpticalFlow(I1, I2, u, v, alpha, gamma, pyramidLevel, pyramidFactor, tolerance);

//    1u.convertTo(uu, CV_8U);
//    namedWindow("Display image", WINDOW_AUTOSIZE);
//    imshow("Display Image", u);
//
//    waitKey(0);

    return 0;
}
