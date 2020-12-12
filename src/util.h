//
// Created by karthik on 06/12/20.
//
#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

//for(int i = 1; i < ny-1; i++)
//{
//for(int j = 1; j < nx-1; j++)
//{
//const int k = i * nx + j;
//dx[k] = 0.5*(input[k+1] - input[k-1]);
//dy[k] = 0.5*(input[k+nx] - input[k-nx]);
//}
//}

void calculateGradients(Mat_<float> &I, Mat_<float> &Ix, Mat_<float> &Iy) {
//    Sobel(I, Ix, CV_32F, 1, 0, 3, 1, 0);
//    Sobel(I, Iy, CV_32F, 0, 1, 3, 1, 0);

    Ix = Mat::zeros(I.rows, I.cols, CV_32F);
    Iy = Mat::zeros(I.rows, I.cols, CV_32F);

    // center of image
    for (int i = 1; i < I.rows - 1; ++i) {
        for (int j = 1; j < I.cols - 1; ++j) {
            Ix.at<float>(i, j) = 0.5f * (I.at<float>(i, j + 1) - I.at<float>(i, j - 1));
            Iy.at<float>(i, j) = 0.5f * (I.at<float>(i + 1, j) - I.at<float>(i - 1, j));
        }
    }

    // first and last rows
    for (int i = 1; i < I.cols - 1; ++i) {
        Ix.at<float>(0, i) = 0.5f * (I.at<float>(0, i + 1) - I.at<float>(0, i - 1));
        Iy.at<float>(0, i) = 0.5f * (I.at<float>(1, i) - I.at<float>(0, i));

        Ix.at<float>(I.rows - 1, i) = 0.5f * (I.at<float>(I.rows - 1, i + 1) - I.at<float>(I.rows - 1, i - 1));
        Iy.at<float>(I.rows - 1, i) = 0.5f * (I.at<float>(I.rows - 1, i) - I.at<float>(I.rows - 2, i));
    }

    // first and last columns
    for (int i = 1; i < I.rows - 1; ++i) {
        Ix.at<float>(i, 0) = 0.5f * (I.at<float>(i, 1) - I.at<float>(i, 0));
        Iy.at<float>(i, 0) = 0.5f * (I.at<float>(i + 1, 0) - I.at<float>(i - 1, 0));

        Ix.at<float>(i, I.cols - 1) = 0.5f * (I.at<float>(i, I.cols - 1) - I.at<float>(i, I.cols - 2));
        Iy.at<float>(i, I.cols - 1) = 0.5f * (I.at<float>(i + 1, I.cols - 1) - I.at<float>(i - 1, I.cols - 1));
    }

    // top-left corner
    Ix.at<float>(0, 0) = 0.5f * (I.at<float>(0, 1) - I.at<float>(0, 0));
    Iy.at<float>(0, 0) = 0.5f * (I.at<float>(1, 0) - I.at<float>(1, 0));

    // top-right corner
    Ix.at<float>(0, I.cols - 1) = 0.5f * (I.at<float>(0, I.cols - 1) - I.at<float>(0, I.cols - 2));
    Iy.at<float>(0, I.cols - 1) = 0.5f * (I.at<float>(1, I.cols - 1) - I.at<float>(0, I.cols - 1));

    // bottom-left corner
    Ix.at<float>(I.rows - 1, 0) = 0.5f * (I.at<float>(I.rows - 1, 1) - I.at<float>(I.rows - 1, 0));
    Iy.at<float>(I.rows - 1, 0) = 0.5f * (I.at<float>(I.rows - 1, 0) - I.at<float>(I.rows - 2, 0));

    // bottom-right corner
    Ix.at<float>(I.rows - 1, I.cols - 1) =
            0.5f * (I.at<float>(I.rows - 1, I.cols - 1) - I.at<float>(I.rows - 1, I.cols - 2));
    Iy.at<float>(I.rows - 1, I.cols - 1) =
            0.5f * (I.at<float>(I.rows - 1, I.cols - 1) - I.at<float>(I.rows - 2, I.cols - 1));
}

void calculateSecondOrderGradients(Mat_<float> &I, Mat_<float> &Ixx, Mat_<float> &Iyy, Mat_<float> &Ixy) {
    Mat_<float> kxx = Mat::zeros(3, 3, CV_32F);
    kxx.at<float>(1, 0) = 1;
    kxx.at<float>(1, 2) = 1;
    kxx.at<float>(1, 1) = -2;

    Mat_<float> kyy = kxx.t();

    Mat_<float> kxy = Mat::zeros(3, 3, CV_32F);
    kxy.at<float>(0, 0) = 1;
    kxy.at<float>(0, 2) = -1;
    kxy.at<float>(2, 0) = -1;
    kxy.at<float>(2, 2) = 1;

    filter2D(I, Ixx, CV_32F, kxx);
    filter2D(I, Iyy, CV_32F, kyy);
    filter2D(I, Ixy, CV_32F, kxy);
}
