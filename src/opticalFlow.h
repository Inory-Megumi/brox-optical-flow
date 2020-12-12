//
// Created by karthik on 06/12/20.
//
#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

void
calculateLevelOpticalFlow(Mat_<float> &I1, Mat_<float> &I2, Mat_<float> &u, Mat_<float> &v, float alpha, float gamma,
                          float tolerance);

void calculateOpticalFlow(Mat_<float> &I1, Mat_<float> &I2, Mat_<float> &u, Mat_<float> &v, float alpha, float gamma,
                          int pyramidLevel, float pyramidFactor, float tolerance);