//
// Created by karthik on 06/12/20.
//
#include "opticalFlow.h"
#include "util.h"

#include <vector>
#include <opencv2/imgproc.hpp>

#define OUTER_LOOP_COUNT 1
#define INNER_LOOP_COUNT 1
#define EPSILON 0.001
#define MAX_SOR_ITER 1
#define SOR_PARAM 1.9

void calculate_psi_smooth(Mat_<float> &ux, Mat_<float> &uy, Mat_<float> &vx, Mat_<float> &vy, Mat_<float> &psi_smooth) {
    psi_smooth = Mat::zeros(ux.rows, ux.cols, CV_32F);
    for (int i = 0; i < ux.rows; ++i) {
        for (int j = 0; j < ux.cols; ++j) {
            float d2 = ux.at<float>(i, j) * ux.at<float>(i, j) + uy.at<float>(i, j) * uy.at<float>(i, j) +
                       vx.at<float>(i, j) * vx.at<float>(i, j) + vy.at<float>(i, j) * vy.at<float>(i, j);;
            psi_smooth.at<float>(i, j) = 1.0 / sqrt(d2 + EPSILON * EPSILON);
        }
    }
}

void calculate_psi_divergence_coeff(Mat_<float> &psi, Mat_<float> &psi1, Mat_<float> &psi2, Mat_<float> &psi3,
                                    Mat_<float> &psi4) {
    psi1 = Mat::zeros(psi.rows, psi.cols, CV_32F);
    psi2 = Mat::zeros(psi.rows, psi.cols, CV_32F);
    psi3 = Mat::zeros(psi.rows, psi.cols, CV_32F);
    psi4 = Mat::zeros(psi.rows, psi.cols, CV_32F);

    // (Ψ's(i+1,j) + Ψ's(i,j))/2 ; (Ψ's(i-1,j) + Ψ's(i,j))/2
    // (Ψ's(i,j+1) + Ψ's(i,j))/2 ; (Ψ's(i,j-1) + Ψ's(i,j))/2

    // at center of the image
    for (int i = 1; i < psi.rows - 1; ++i) {
        for (int j = 1; j < psi.cols - 1; ++j) {
            psi1.at<float>(i, j) = 0.5f * (psi.at<float>(i + 1, j) + psi.at<float>(i, j));
            psi2.at<float>(i, j) = 0.5f * (psi.at<float>(i - 1, j) + psi.at<float>(i, j));
            psi3.at<float>(i, j) = 0.5f * (psi.at<float>(i, j + 1) + psi.at<float>(i, j));
            psi4.at<float>(i, j) = 0.5f * (psi.at<float>(i, j - 1) + psi.at<float>(i, j));
        }
    }

    // first & last rows
    for (int i = 1; i < psi.cols - 1; ++i) {
        psi1.at<float>(0, i) = 0.5f * (psi.at<float>(1, i) + psi.at<float>(0, i));
        psi2.at<float>(0, i) = 0;
        psi3.at<float>(0, i) = 0.5f * (psi.at<float>(0, i + 1) + psi.at<float>(0, i));
        psi4.at<float>(0, i) = 0.5f * (psi.at<float>(0, i - 1) + psi.at<float>(0, i));

        psi1.at<float>(psi.rows - 1, i) = 0;
        psi2.at<float>(psi.rows - 1, i) = 0.5f * (psi.at<float>(psi.rows - 2, i) + psi.at<float>(psi.rows - 1, i));
        psi3.at<float>(psi.rows - 1, i) = 0.5f * (psi.at<float>(psi.rows - 1, i + 1) + psi.at<float>(psi.rows - 1, i));
        psi4.at<float>(psi.rows - 1, i) = 0.5f * (psi.at<float>(psi.rows - 1, i - 1) + psi.at<float>(psi.rows - 1, i));
    }

    // first & last columns
    for (int i = 1; i < psi.rows - 1; ++i) {
        psi1.at<float>(i, 0) = 0.5f * (psi.at<float>(i + 1, 0) + psi.at<float>(i, 0));
        psi2.at<float>(i, 0) = 0.5f * (psi.at<float>(i - 1, 0) + psi.at<float>(i, 0));
        psi3.at<float>(i, 0) = 0.5f * (psi.at<float>(i, 1) + psi.at<float>(i, 0));
        psi4.at<float>(i, 0) = 0;

        psi1.at<float>(i, psi.cols - 1) = 0.5f * (psi.at<float>(i + 1, psi.cols - 1) + psi.at<float>(i, psi.cols - 1));
        psi2.at<float>(i, psi.cols - 1) = 0.5f * (psi.at<float>(i - 1, psi.cols - 1) + psi.at<float>(i, psi.cols - 1));
        psi3.at<float>(i, psi.cols - 1) = 0;
        psi4.at<float>(i, psi.cols - 1) = 0.5f * (psi.at<float>(i, psi.cols - 2) + psi.at<float>(i, psi.cols - 1));
    }

    // top-left corner
    psi1.at<float>(0, 0) = 0.5f * (psi.at<float>(1, 0) + psi.at<float>(0, 0));
    psi2.at<float>(0, 0) = 0;
    psi3.at<float>(0, 0) = 0.5f * (psi.at<float>(0, 1) + psi.at<float>(0, 0));
    psi4.at<float>(0, 0) = 0;

    // top-right corner
    psi1.at<float>(0, psi.cols - 1) = 0.5f * (psi.at<float>(1, psi.cols - 1) + psi.at<float>(0, psi.cols - 1));
    psi2.at<float>(0, psi.cols - 1) = 0;
    psi3.at<float>(0, psi.cols - 1) = 0;
    psi4.at<float>(0, psi.cols - 1) = 0.5f * (psi.at<float>(0, psi.cols - 2) + psi.at<float>(0, psi.cols - 1));

    // bottom-left corner
    psi1.at<float>(psi.rows - 1, 0) = 0;
    psi2.at<float>(psi.rows - 1, 0) = 0.5f * (psi.at<float>(psi.rows - 2, 0) + psi.at<float>(psi.rows - 1, 0));
    psi3.at<float>(psi.rows - 1, 0) = 0.5f * (psi.at<float>(psi.rows - 1, 1) + psi.at<float>(psi.rows - 1, 0));
    psi4.at<float>(psi.rows - 1, 0) = 0;

    // bottom-right corner
    psi1.at<float>(psi.rows - 1, psi.cols - 1) = 0;
    psi2.at<float>(psi.rows - 1, psi.cols - 1) =
            0.5f * (psi.at<float>(psi.rows - 2, psi.cols - 1) + psi.at<float>(psi.rows - 1, psi.cols - 1));
    psi3.at<float>(psi.rows - 1, psi.cols - 1) = 0;
    psi4.at<float>(psi.rows - 1, psi.cols - 1) =
            0.5f * (psi.at<float>(psi.rows - 1, psi.cols - 2) + psi.at<float>(psi.rows - 1, psi.cols - 1));
}

void calculate_divergence(Mat_<float> &u, Mat_<float> &v, Mat_<float> &psi1, Mat_<float> &psi2, Mat_<float> &psi3,
                          Mat_<float> &psi4, Mat_<float> &div_u, Mat_<float> &div_v) {

    div_u = Mat::zeros(u.rows, u.cols, CV_32F);
    div_v = Mat::zeros(u.rows, u.cols, CV_32F);

    for (int i = 1; i < u.rows - 1; ++i) {
        for (int j = 1; j < u.cols - 1; ++j) {
            div_u.at<float>(i, j) = psi1.at<float>(i, j) * (u.at<float>(i + 1, j) - u.at<float>(i, j)) +
                                    psi2.at<float>(i, j) * (u.at<float>(i - 1, j) - u.at<float>(i, j)) +
                                    psi3.at<float>(i, j) * (u.at<float>(i, j + 1) - u.at<float>(i, j)) +
                                    psi4.at<float>(i, j) * (u.at<float>(i, j - 1) - u.at<float>(i, j));
            div_v.at<float>(i, j) = psi1.at<float>(i, j) * (v.at<float>(i + 1, j) - v.at<float>(i, j)) +
                                    psi2.at<float>(i, j) * (v.at<float>(i - 1, j) - v.at<float>(i, j)) +
                                    psi3.at<float>(i, j) * (v.at<float>(i, j + 1) - v.at<float>(i, j)) +
                                    psi4.at<float>(i, j) * (v.at<float>(i, j - 1) - v.at<float>(i, j));
        }
    }

    // first and last rows
    for (int i = 1; i < u.cols - 1; ++i) {
        div_u.at<float>(0, i) = psi1.at<float>(0, i) * (u.at<float>(1, i) - u.at<float>(0, i)) +
                                psi3.at<float>(0, i) * (u.at<float>(0, i + 1) - u.at<float>(0, i)) +
                                psi4.at<float>(0, i) * (u.at<float>(0, i - 1) - u.at<float>(0, i));
        div_v.at<float>(0, i) = psi1.at<float>(0, i) * (v.at<float>(1, i) - v.at<float>(0, i)) +
                                psi3.at<float>(0, i) * (v.at<float>(0, i + 1) - v.at<float>(0, i)) +
                                psi4.at<float>(0, i) * (v.at<float>(0, i - 1) - v.at<float>(0, i));
    }

    // first and last columns
    for (int i = 1; i < u.rows - 1; ++i) {
        div_u.at<float>(i, 0) = psi1.at<float>(i, 0) * (u.at<float>(i + 1, 0) - u.at<float>(i, 0)) +
                                psi2.at<float>(i, 0) * (u.at<float>(i - 1, 0) - u.at<float>(i, 0)) +
                                psi3.at<float>(i, 0) * (u.at<float>(i, 1) - u.at<float>(i, 0));
        div_v.at<float>(i, 0) = psi1.at<float>(i, 0) * (v.at<float>(i + 1, 0) - v.at<float>(i, 0)) +
                                psi2.at<float>(i, 0) * (v.at<float>(i - 1, 0) - v.at<float>(i, 0)) +
                                psi3.at<float>(i, 0) * (v.at<float>(i, 1) - v.at<float>(i, 0));
    }

    // top-left corner
    div_u.at<float>(0, 0) = psi1.at<float>(0, 0) * (u.at<float>(1, 0) - u.at<float>(0, 0)) +
                            psi3.at<float>(0, 0) * (u.at<float>(0, 1) - u.at<float>(0, 0));
    div_v.at<float>(0, 0) = psi1.at<float>(0, 0) * (v.at<float>(1, 0) - v.at<float>(0, 0)) +
                            psi3.at<float>(0, 0) * (v.at<float>(0, 1) - v.at<float>(0, 0));

    // top-right corner
    div_u.at<float>(0, u.cols - 1) =
            psi1.at<float>(0, u.cols - 1) * (u.at<float>(1, u.cols - 1) - u.at<float>(0, u.cols - 1)) +
            psi4.at<float>(0, 0) * (u.at<float>(0, u.cols - 2) - u.at<float>(0, u.cols - 1));
    div_v.at<float>(0, u.cols - 1) =
            psi1.at<float>(0, u.cols - 1) * (v.at<float>(1, u.cols - 1) - v.at<float>(0, u.cols - 1)) +
            psi4.at<float>(0, 0) * (v.at<float>(0, u.cols - 2) - v.at<float>(0, u.cols - 1));

    // bottom-left corner
    div_u.at<float>(u.rows - 1, 0) =
            psi2.at<float>(u.rows - 1, 0) * (u.at<float>(u.rows - 2, 0) - u.at<float>(u.rows - 1, 0)) +
            psi3.at<float>(u.rows - 1, 0) * (u.at<float>(u.rows - 1, 1) - u.at<float>(u.rows - 1, 0));
    div_v.at<float>(u.rows - 1, 0) =
            psi2.at<float>(u.rows - 1, 0) * (v.at<float>(u.rows - 2, 0) - v.at<float>(u.rows - 1, 0)) +
            psi3.at<float>(u.rows - 1, 0) * (v.at<float>(u.rows - 1, 1) - v.at<float>(u.rows - 1, 0));

    // bottom-right corner
    div_u.at<float>(u.rows - 1, u.cols - 1) =
            psi2.at<float>(u.rows - 1, u.cols - 1) *
            (u.at<float>(u.rows - 2, u.cols - 1) - u.at<float>(u.rows - 1, u.cols - 1)) +
            psi4.at<float>(u.rows - 1, u.cols - 1) *
            (u.at<float>(u.rows - 1, u.cols - 2) - u.at<float>(u.rows - 1, u.cols - 1));
    div_v.at<float>(u.rows - 1, u.cols - 1) =
            psi2.at<float>(u.rows - 1, u.cols - 1) *
            (v.at<float>(u.rows - 2, u.cols - 1) - v.at<float>(u.rows - 1, u.cols - 1)) +
            psi4.at<float>(u.rows - 1, u.cols - 1) *
            (v.at<float>(u.rows - 1, u.cols - 2) - v.at<float>(u.rows - 1, u.cols - 1));
}

void calculate_psi_data(Mat_<float> &I1, Mat_<float> &I2, Mat_<float> &I2x, Mat_<float> &I2y, Mat_<float> &du,
                        Mat_<float> &dv, Mat_<float> &psi_data) {
    // calculate Ψ′D = Ψ′((I2 - I1_warped + I2x*du + I2y*dv)^2)
    psi_data = Mat::zeros(I1.rows, I1.cols, CV_32F);
    for (int i = 0; i < I1.rows; ++i) {
        for (int j = 0; j < I1.cols; ++j) {
            float dI = I2.at<float>(i, j) - I1.at<float>(i, j) + I2x.at<float>(i, j) * du.at<float>(i, j) +
                       I2y.at<float>(i, j) * dv.at<float>(i, j);
            psi_data.at<float>(i, j) = 1.0f / sqrt(dI * dI + EPSILON * EPSILON);
        }
    }
}

void calculate_psi_gradient(Mat_<float> &I1x, Mat_<float> &I1y, Mat_<float> &I2x, Mat_<float> &I2y, Mat_<float> &I2xx,
                            Mat_<float> &I2yy, Mat_<float> &I2xy, Mat_<float> &du, Mat_<float> &dv,
                            Mat_<float> &psi_gradient) {
    // calculate Ψ′G = Ψ′(|∇I(x+w) - ∇I(x)|^2)
    psi_gradient = Mat::zeros(I1x.rows, I1x.cols, CV_32F);
    for (int i = 0; i < I1x.rows; ++i) {
        for (int j = 0; j < I1x.cols; ++j) {
            float dIx = I2x.at<float>(i, j) - I1x.at<float>(i, j) + I2xx.at<float>(i, j) * du.at<float>(i, j) +
                        I2xy.at<float>(i, j) * dv.at<float>(i, j);
            float dIy = I2y.at<float>(i, j) - I1y.at<float>(i, j) + I2xy.at<float>(i, j) * dv.at<float>(i, j) +
                        I2yy.at<float>(i, j) * dv.at<float>(i, j);
            float dI = dIx * dIx + dIy * dIy;
            psi_gradient.at<float>(i, j) = 1.0f / sqrt(dI + EPSILON * EPSILON);
        }
    }
}

void sor_iterate(Mat_<float> &Au, Mat_<float> &Av, Mat_<float> &Du, Mat_<float> &Dv, Mat_<float> &D, Mat_<float> &du,
                  Mat_<float> &dv,
                  float &alpha, Mat_<float> &psi1, Mat_<float> &psi2, Mat_<float> &psi3, Mat_<float> &psi4,
                  float tolerance) {

    float error = FLT_MAX;
    int iter = 0;
    float sor_param = SOR_PARAM;

//    for (int i = 0; i < du.rows; ++i) {
//        for (int j = 0; j < du.cols; ++j) {
//            if (du.at<float>(i, j) != 0) {
//                std::cout << "Here " << du.at<float>(i, j) << "\n";
//            }
//        }
//    }

    while (error > tolerance && iter < MAX_SOR_ITER) {
        error = 0;
//        for (int i = 0; i < du.rows; ++i) {
//            for (int j = 0; j < du.cols; ++j) {
//                if (du.at<float>(i, j) != 0) {
//                    std::cout << "Here " << du.at<float>(i, j) << "\n";
//                }
//            }
//        }
        // center of image
        for (int i = 1; i < Au.rows - 1; ++i) {
            for (int j = 1; j < Au.cols - 1; ++j) {
                float div_du =
                        psi1.at<float>(i, j) * du.at<float>(i + 1, j) + psi2.at<float>(i, j) * du.at<float>(i - 1, j) +
                        psi3.at<float>(i, j) * du.at<float>(i, j + 1) + psi4.at<float>(i, j) * du.at<float>(i, j - 1);
                float div_dv =
                        psi1.at<float>(i, j) * dv.at<float>(i + 1, j) + psi2.at<float>(i, j) * du.at<float>(i - 1, j) +
                        psi3.at<float>(i, j) * dv.at<float>(i, j + 1) + psi4.at<float>(i, j) * dv.at<float>(i, j - 1);
//                std::cout<<div_du<<"\n";
//                std::cout<<psi1.at<float>(i,j)<<" "<<psi2.at<float>(i,j)<<" "<<psi3.at<float>(i,j)<<" "<<psi4.at<float>(i,j)<<" ";
//                std::cout<<du.at<float>(i+1,j)<<" "<<du.at<float>(i-1,j)<<" "<<du.at<float>(i,j+1)<<" "<<du.at<float>(i,j-1)<<" \n";
                float du_last = du.at<float>(i, j);
                float dv_last = dv.at<float>(i, j);

                du.at<float>(i, j) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(i, j) -
                                                                                 D.at<float>(i, j) *
                                                                                 dv.at<float>(i, j) +
                                                                                 alpha * div_du) / Du.at<float>(i, j);
                dv.at<float>(i, j) = (1.0f - sor_param) * dv_last + sor_param * (Av.at<float>(i, j) -
                                                                                 D.at<float>(i, j) *
                                                                                 du.at<float>(i, j) +
                                                                                 alpha * div_dv) / Dv.at<float>(i, j);

//                std::cout<<du.at<float>(i,j)<<"\n";

                error += ((du.at<float>(i, j) - du_last) * (du.at<float>(i, j) - du_last) +
                          (dv.at<float>(i, j) - dv_last) * (dv.at<float>(i, j) - dv_last));
            }
        }
//        for (int i = 0; i < 50; ++i) {
//            for (int j = 0; j < 50; ++j) {
//                std::cout<<du.at<float>(i,j)<<"\n";
//            }
//        }
//        std::cout << du.at<float>(100, 100) << "\n";

        // first and last rows
        for (int i = 1; i < Au.cols - 1; ++i) {
            // first row
            float div_du =
                    psi1.at<float>(0, i) * du.at<float>(1, i) + psi2.at<float>(0, i) * du.at<float>(0, i) +
                    psi3.at<float>(0, i) * du.at<float>(0, i + 1) + psi4.at<float>(0, i) * du.at<float>(0, i - 1);
            float div_dv =
                    psi1.at<float>(0, i) * dv.at<float>(1, i) + psi2.at<float>(0, i) * dv.at<float>(0, i) +
                    psi3.at<float>(0, i) * dv.at<float>(0, i + 1) + psi4.at<float>(0, i) * dv.at<float>(0, i - 1);
            float du_last = du.at<float>(0, i);
            float dv_last = dv.at<float>(0, i);

            du.at<float>(0, i) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(0, i) -
                                                                             D.at<float>(0, i) *
                                                                             dv.at<float>(0, i) +
                                                                             alpha * div_du) / Du.at<float>(0, i);
            dv.at<float>(0, i) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(0, i) -
                                                                             D.at<float>(0, i) *
                                                                             du.at<float>(0, i) +
                                                                             alpha * div_dv) / Dv.at<float>(0, i);
            error += ((du.at<float>(0, i) - du_last) * (du.at<float>(0, i) - du_last) +
                      (dv.at<float>(0, i) - dv_last) * (dv.at<float>(0, i) - dv_last));

            // last row
            div_du =
                    psi1.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i) +
                    psi2.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 2, i) +
                    psi3.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i + 1) +
                    psi4.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i - 1);
            div_dv =
                    psi1.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i) +
                    psi2.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 2, i) +
                    psi3.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i + 1) +
                    psi4.at<float>(Au.rows - 1, i) * du.at<float>(Au.rows - 1, i - 1);
            du_last = du.at<float>(Au.rows - 1, i);
            dv_last = dv.at<float>(Au.rows - 1, i);
            du.at<float>(Au.rows - 1, i) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(Au.rows - 1, i) -
                                                                                       D.at<float>(Au.rows - 1, i) *
                                                                                       dv.at<float>(Au.rows - 1, i) +
                                                                                       alpha * div_du) /
                                                                          Du.at<float>(Au.rows - 1, i);
            dv.at<float>(Au.rows - 1, i) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(Au.rows - 1, i) -
                                                                                       D.at<float>(Au.rows - 1, i) *
                                                                                       du.at<float>(Au.rows - 1, i) +
                                                                                       alpha * div_dv) /
                                                                          Dv.at<float>(Au.rows - 1, i);
            error += ((du.at<float>(Au.rows - 1, i) - du_last) * (du.at<float>(Au.rows - 1, i) - du_last) +
                      (dv.at<float>(Au.rows - 1, i) - dv_last) * (dv.at<float>(Au.rows - 1, i) - dv_last));
        }

        // first and last column
        for (int i = 1; i < Au.rows - 1; ++i) {
            // first column
            float div_du =
                    psi1.at<float>(i, 0) * du.at<float>(i + 1, 0) + psi2.at<float>(i, 0) * du.at<float>(i - 1, 0) +
                    psi3.at<float>(i, 0) * du.at<float>(i, 1) + psi4.at<float>(i, 0) * du.at<float>(i, 0);
            float div_dv =
                    psi1.at<float>(i, 0) * dv.at<float>(i + 1, 0) + psi2.at<float>(i, 0) * dv.at<float>(i - 1, 0) +
                    psi3.at<float>(i, 0) * dv.at<float>(i, 1) + psi4.at<float>(i, 0) * dv.at<float>(i, 0);
            float du_last = du.at<float>(i, 0);
            float dv_last = dv.at<float>(i, 0);

            du.at<float>(i, 0) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(i, 0) -
                                                                             D.at<float>(i, 0) *
                                                                             dv.at<float>(i, 0) +
                                                                             alpha * div_du) / Du.at<float>(i, 0);
            dv.at<float>(i, 0) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(i, 0) -
                                                                             D.at<float>(i, 0) *
                                                                             du.at<float>(i, 0) +
                                                                             alpha * div_dv) / Dv.at<float>(i, 0);
            error += ((du.at<float>(i, 0) - du_last) * (du.at<float>(i, 0) - du_last) +
                      (dv.at<float>(i, 0) - dv_last) * (dv.at<float>(i, 0) - dv_last));

            // last column
            div_du =
                    psi1.at<float>(i, Au.cols - 1) * du.at<float>(i + 1, Au.cols - 1) +
                    psi2.at<float>(i, Au.cols - 1) * du.at<float>(i - 1, Au.cols - 1) +
                    psi3.at<float>(i, Au.cols - 1) * du.at<float>(i, Au.cols - 1) +
                    psi4.at<float>(i, Au.cols - 1) * du.at<float>(i, Au.cols - 2);
            div_dv =
                    psi1.at<float>(i, Au.cols - 1) * dv.at<float>(i + 1, Au.cols - 1) +
                    psi2.at<float>(i, Au.cols - 1) * dv.at<float>(i - 1, Au.cols - 1) +
                    psi3.at<float>(i, Au.cols - 1) * dv.at<float>(i, Au.cols - 1) +
                    psi4.at<float>(i, Au.cols - 1) * dv.at<float>(i, Au.cols - 2);
            du_last = du.at<float>(i, Au.cols - 1);
            dv_last = dv.at<float>(i, Au.cols - 1);

            du.at<float>(i, Au.cols - 1) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(i, Au.cols - 1) -
                                                                                       D.at<float>(i, Au.cols - 1) *
                                                                                       dv.at<float>(i, Au.cols - 1) +
                                                                                       alpha * div_du) /
                                                                          Du.at<float>(i, 0);
            dv.at<float>(i, Au.cols - 1) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(i, Au.cols - 1) -
                                                                                       D.at<float>(i, Au.cols - 1) *
                                                                                       du.at<float>(i, Au.cols - 1) +
                                                                                       alpha * div_dv) /
                                                                          Dv.at<float>(i, Au.cols - 1);
            error += ((du.at<float>(i, Au.cols - 1) - du_last) * (du.at<float>(i, Au.cols - 1) - du_last) +
                      (dv.at<float>(i, Au.cols - 1) - dv_last) * (dv.at<float>(i, Au.cols - 1) - dv_last));
        }

        // top-left corner
        float div_du =
                psi1.at<float>(0, 0) * du.at<float>(1, 0) + psi2.at<float>(0, 0) * du.at<float>(0, 0) +
                psi3.at<float>(0, 0) * du.at<float>(0, 1) + psi4.at<float>(0, 0) * du.at<float>(0, 0);
        float div_dv =
                psi1.at<float>(0, 0) * dv.at<float>(1, 0) + psi2.at<float>(0, 0) * dv.at<float>(0, 0) +
                psi3.at<float>(0, 0) * dv.at<float>(0, 1) + psi4.at<float>(0, 0) * dv.at<float>(0, 0);
        float du_last = du.at<float>(0, 0);
        float dv_last = dv.at<float>(0, 0);

        du.at<float>(0, 0) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(0, 0) -
                                                                         D.at<float>(0, 0) *
                                                                         dv.at<float>(0, 0) +
                                                                         alpha * div_du) / Du.at<float>(0, 0);
        dv.at<float>(0, 0) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(0, 0) -
                                                                         D.at<float>(0, 0) *
                                                                         du.at<float>(0, 0) +
                                                                         alpha * div_dv) / Dv.at<float>(0, 0);
        error += ((du.at<float>(0, 0) - du_last) * (du.at<float>(0, 0) - du_last) +
                  (dv.at<float>(0, 0) - dv_last) * (dv.at<float>(0, 0) - dv_last));

        // top-right corner
        div_du =
                psi1.at<float>(0, Au.cols - 1) * du.at<float>(1, Au.cols - 1) +
                psi2.at<float>(0, Au.cols - 1) * du.at<float>(0, Au.cols - 1) +
                psi3.at<float>(0, Au.cols - 1) * du.at<float>(0, Au.cols - 1) +
                psi4.at<float>(0, Au.cols - 1) * du.at<float>(0, Au.cols - 2);
        div_dv =
                psi1.at<float>(0, Au.cols - 1) * dv.at<float>(1, Au.cols - 1) +
                psi2.at<float>(0, Au.cols - 1) * dv.at<float>(0, Au.cols - 1) +
                psi3.at<float>(0, Au.cols - 1) * dv.at<float>(0, Au.cols - 1) +
                psi4.at<float>(0, Au.cols - 1) * dv.at<float>(0, Au.cols - 2);
        du_last = du.at<float>(0, Au.cols - 1);
        dv_last = dv.at<float>(0, Au.cols - 1);

        du.at<float>(0, Au.cols - 1) = (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(0, Au.cols - 1) -
                                                                                   D.at<float>(0, Au.cols - 1) *
                                                                                   dv.at<float>(0, Au.cols - 1) +
                                                                                   alpha * div_du) /
                                                                      Du.at<float>(0, Au.cols - 1);
        dv.at<float>(0, Au.cols - 1) = (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(0, Au.cols - 1) -
                                                                                   D.at<float>(0, Au.cols - 1) *
                                                                                   du.at<float>(0, Au.cols - 1) +
                                                                                   alpha * div_dv) /
                                                                      Dv.at<float>(0, Au.cols - 1);
        error += ((du.at<float>(0, Au.cols - 1) - du_last) * (du.at<float>(0, Au.cols - 1) - du_last) +
                  (dv.at<float>(0, Au.cols - 1) - dv_last) * (dv.at<float>(0, Au.cols - 1) - dv_last));

        // bottom-left corner
        div_du =
                psi1.at<float>(Au.rows - 1, 0) * du.at<float>(Au.rows - 1, 0) +
                psi2.at<float>(Au.rows - 1, 0) * du.at<float>(Au.rows - 2, 0) +
                psi3.at<float>(Au.rows - 1, 0) * du.at<float>(Au.rows - 1, 1) +
                psi4.at<float>(Au.rows - 1, 0) * du.at<float>(Au.rows - 1, 0);
        div_dv =
                psi1.at<float>(Au.rows - 1, 0) * dv.at<float>(Au.rows - 1, 0) +
                psi2.at<float>(Au.rows - 1, 0) * dv.at<float>(Au.rows - 2, 0) +
                psi3.at<float>(Au.rows - 1, 0) * dv.at<float>(Au.rows - 1, 1) +
                psi4.at<float>(Au.rows - 1, 0) * dv.at<float>(Au.rows - 1, 0);
        du_last = du.at<float>(Au.rows - 1, 0);
        dv_last = dv.at<float>(Au.rows - 1, 0);

        du.at<float>(Au.rows - 1, 0) =
                (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(Au.rows - 1, 0) -
                                                            D.at<float>(Au.rows - 1, 0) *
                                                            dv.at<float>(Au.rows - 1, 0) +
                                                            alpha * div_du) /
                                               Du.at<float>(Au.rows - 1, 0);
        dv.at<float>(Au.rows - 1, 0) =
                (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(Au.rows - 1, 0) -
                                                            D.at<float>(Au.rows - 1, 0) *
                                                            du.at<float>(Au.rows - 1, 0) +
                                                            alpha * div_dv) /
                                               Dv.at<float>(Au.rows - 1, 0);
        error += ((du.at<float>(Au.rows - 1, 0) - du_last) *
                  (du.at<float>(Au.rows - 1, 0) - du_last) +
                  (dv.at<float>(Au.rows - 1, 0) - dv_last) *
                  (dv.at<float>(Au.rows - 1, 0) - dv_last));

        // bottom-right corner
        div_du =
                psi1.at<float>(Au.rows - 1, Au.cols - 1) * du.at<float>(Au.rows - 1, Au.cols - 1) +
                psi2.at<float>(Au.rows - 1, Au.cols - 1) * du.at<float>(Au.rows - 2, Au.cols - 1) +
                psi3.at<float>(Au.rows - 1, Au.cols - 1) * du.at<float>(Au.rows - 1, Au.cols - 1) +
                psi4.at<float>(Au.rows - 1, Au.cols - 1) * du.at<float>(Au.rows - 1, Au.cols - 2);
        div_dv =
                psi1.at<float>(Au.rows - 1, Au.cols - 1) * dv.at<float>(Au.rows - 1, Au.cols - 1) +
                psi2.at<float>(Au.rows - 1, Au.cols - 1) * dv.at<float>(Au.rows - 2, Au.cols - 1) +
                psi3.at<float>(Au.rows - 1, Au.cols - 1) * dv.at<float>(Au.rows - 1, Au.cols - 1) +
                psi4.at<float>(Au.rows - 1, Au.cols - 1) * dv.at<float>(Au.rows - 1, Au.cols - 2);
        du_last = du.at<float>(Au.rows - 1, Au.cols - 1);
        dv_last = dv.at<float>(Au.rows - 1, Au.cols - 1);

        du.at<float>(Au.rows - 1, Au.cols - 1) =
                (1.0f - sor_param) * du_last + sor_param * (Au.at<float>(Au.rows - 1, Au.cols - 1) -
                                                            D.at<float>(Au.rows - 1, Au.cols - 1) *
                                                            dv.at<float>(Au.rows - 1, Au.cols - 1) +
                                                            alpha * div_du) /
                                               Du.at<float>(Au.rows - 1, Au.cols - 1);
        dv.at<float>(Au.rows - 1, Au.cols - 1) =
                (1.0f - sor_param) * du_last + sor_param * (Av.at<float>(Au.rows - 1, Au.cols - 1) -
                                                            D.at<float>(Au.rows - 1, Au.cols - 1) *
                                                            du.at<float>(Au.rows - 1, Au.cols - 1) +
                                                            alpha * div_dv) /
                                               Dv.at<float>(Au.rows - 1, Au.cols - 1);
        error += ((du.at<float>(Au.rows - 1, Au.cols - 1) - du_last) *
                  (du.at<float>(Au.rows - 1, Au.cols - 1) - du_last) +
                  (dv.at<float>(Au.rows - 1, Au.cols - 1) - dv_last) *
                  (dv.at<float>(Au.rows - 1, Au.cols - 1) - dv_last));

        error = sqrt(error / float(Au.rows * Au.cols));
        iter++;
    }
}

void
calculateLevelOpticalFlow(Mat_<float> &I1, Mat_<float> &I2, Mat_<float> &u, Mat_<float> &v, float alpha, float gamma,
                          float tolerance) {

    Mat_<float> I1x, I1y;
    Mat_<float> I2x, I2y;
//    Mat_<float> I1xx, I1yy, I1xy;
    Mat_<float> I2xx, I2yy, I2xy, I2yx;
    Mat_<float> I1_warped, I1x_warped, I1y_warped, I1xx_warped, I1yy_warped, I1xy_warped;
    Mat_<float> ux, uy, vx, vy;
    Mat_<float> psi_smooth, psi_data, psi_gradient;
    Mat_<float> psi1, psi2, psi3, psi4;
    Mat_<float> div_u, div_v, div_d;
    Mat_<float> div_du, div_dv;
    Mat_<float> du, dv;
    Mat_<float> Au, Av, Du, Dv, D;
    Au = Mat::zeros(I1.rows, I1.cols, CV_32F);
    Av = Mat::zeros(I1.rows, I1.cols, CV_32F);
    Du = Mat::zeros(I1.rows, I1.cols, CV_32F);
    Dv = Mat::zeros(I1.rows, I1.cols, CV_32F);
    D = Mat::zeros(I1.rows, I1.cols, CV_32F);

    // calculate gradients
    calculateGradients(I1, I1x, I1y);
    calculateGradients(I2, I2x, I2y);

    // calculate second order partial derivatives
    calculateSecondOrderGradients(I2, I2xx, I2yy, I2xy, I2yx);

    // outer iterations
    for (int i = 0; i < OUTER_LOOP_COUNT; ++i) {
        // warp images and derivatives
        Mat_<float> xmap = Mat::zeros(u.rows, u.cols, CV_32F);
        Mat_<float> ymap = Mat::zeros(u.rows, u.cols, CV_32F);
        // generate mapping functions
        for (int j = 0; j < u.rows; ++j) {
            for (int k = 0; k < u.cols; ++k) {
                xmap.at<float>(j, k) = (float) k + u.at<float>(j, k);
                ymap.at<float>(j, k) = (float) j + v.at<float>(j, k);
            }
        }
        remap(I1, I1_warped, xmap, ymap, INTER_CUBIC);
//        std::cout<<"I1 "<<I1.rows<<" "<<I1.cols<<" I1warped "<<I1_warped.rows<<" "<<I1_warped.cols<<"\n";
        remap(I1x, I1x_warped, xmap, ymap, INTER_CUBIC);
        remap(I1y, I1y_warped, xmap, ymap, INTER_CUBIC);
//        remap(I1xx, I1xx_warped, xmap, ymap, INTER_CUBIC);
//        remap(I1yy, I1yy_warped, xmap, ymap, INTER_CUBIC);
//        remap(I1xy, I1xy_warped, xmap, ymap, INTER_CUBIC);

//        for (int k = 0; k < I1.rows; ++k) {
//            for (int l = 0; l < I1.cols; ++l) {
//                std::cout << "I1 " << I1.at<float>(k, l) << " I2 " << I2.at<float>(k, l) << " I1_warped "
//                          << I1_warped.at<float>(k, l) << "\n";
//            }
//        }

        // calculate u, v gradients

        calculateGradients(u, ux, uy);
        calculateGradients(v, vx, vy);

//        for (int k = 0; k < I1.rows; ++k) {
//            for (int l = 0; l < I1.cols; ++l) {
//                std::cout << "ux " << ux.at<float>(k, l) << "\n";
//            }
//        }

        // Note: Ψ'(s^2) = 1.0/√(s^2 + ϵ^2)

        // calculate Ψ's = Ψ'(|∇u|*|∇u| + |∇v|*|∇v|)
        calculate_psi_smooth(ux, uy, vx, vy, psi_smooth);

        // div(Ψ's)k,l.∇(u(k,l) + du(k,l+1)) = div_u + (div_du - div_d.du(k, l+1))
        // coefficients for the terms are
        // (Ψ's(i+1,j) + Ψ's(i,j))/2 ; (Ψ's(i-1,j) + Ψ's(i,j))/2
        // (Ψ's(i,j+1) + Ψ's(i,j))/2 ; (Ψ's(i,j-1) + Ψ's(i,j))/2

        // calculate coefficients for div(((Ψ's)k,l).∇(u(k,l) + du(k,l+1)))) discretization
        calculate_psi_divergence_coeff(psi_smooth, psi1, psi2, psi3, psi4);

        // calculate div_u & div_v
        calculate_divergence(u, v, psi1, psi2, psi3, psi4, div_u, div_v);

        // calculate alpha*div_d and initialize du and dv
        div_d = Mat::zeros(u.rows, u.cols, CV_32F);
        du = Mat::zeros(u.rows, u.cols, CV_32F);
        dv = Mat::zeros(u.rows, u.cols, CV_32F);
        for (int j = 0; j < u.rows; ++j) {
            for (int k = 0; k < u.cols; ++k) {
                div_d.at<float>(j, k) = alpha * (psi1.at<float>(i, j) + psi2.at<float>(i, j) + psi3.at<float>(i, j) +
                                                 psi4.at<float>(i, j));
            }
        }

        // inner iterations
        for (int j = 0; j < INNER_LOOP_COUNT; ++j) {
            // calculate Ψ′D = Ψ′((I2 - I1_warped + I2x*du + I2y*dv)^2)
//            for (int k = 0; k < I1.rows; ++k) {
//                for (int l = 0; l < I1.cols; ++l) {
//                    std::cout << "du " << du.at<float>(k, l) << " dv " << dv.at<float>(k, l) << " \n";
//                }
//            }
            calculate_psi_data(I1_warped, I2, I2x, I2y, du, dv, psi_data);

//            for (int k = 0; k < I1.rows; ++k) {
//                for (int l = 0; l < I1.cols; ++l) {
//                    std::cout<<psi_data.at<float>(k,l)<<"\n";
//                }
//            }

            // calculate Ψ′G = Ψ′(|∇I(x+w) - ∇I(x)|^2)
            calculate_psi_gradient(I1x_warped, I1y_warped, I2x, I2y, I2xx, I2yy, I2xy, du, dv, psi_gradient);

//            for (int k = 0; k < I1.rows; ++k) {
//                for (int l = 0; l < I1.cols; ++l) {
//                    std::cout << psi_gradient.at<float>(k, l) << "\n";
//                }
//            }

            for (int k = 0; k < I1.rows; ++k) {
                for (int l = 0; l < I1.cols; ++l) {
                    {
                        // brightness constancy
                        float diff = I2.at<float>(k, l) - I1_warped.at<float>(k, l);
                        float BAu = -psi_data.at<float>(k, l) * diff + I2x.at<float>(k, l);
                        float BAv = -psi_data.at<float>(k, l) * diff + I2y.at<float>(k, l);
                        float BDu = psi_data.at<float>(k, l) * I2x.at<float>(k, l) * I2x.at<float>(k, l);
                        float BDv = psi_data.at<float>(k, l) * I2y.at<float>(k, l) * I2y.at<float>(k, l);

                        // gradient constancy
                        float dx = I2x.at<float>(k, l) - I1x_warped.at<float>(k, l);
                        float dy = I2y.at<float>(k, l) - I1y_warped.at<float>(k, l);
                        float GAu = -gamma * psi_gradient.at<float>(k, l) *
                                    (dx * I2xx.at<float>(k, l) + dy * I2yx.at<float>(k, l));
                        float GAv = -gamma * psi_gradient.at<float>(k, l) *
                                    (dx * I2xy.at<float>(k, l) + dy * I2yy.at<float>(k, l));
                        float GDu = gamma * psi_gradient.at<float>(k, l) *
                                    (I2xx.at<float>(k, l) * I2xx.at<float>(k, l) +
                                     I2xy.at<float>(k, l) * I2xy.at<float>(k, l));
                        float GDv = gamma * psi_gradient.at<float>(k, l) *
                                    (I2yy.at<float>(k, l) * I2yy.at<float>(k, l) +
                                     I2xy.at<float>(k, l) * I2xy.at<float>(k, l));
                        float Duv = psi_data.at<float>(k, l) * I2y.at<float>(k, l) * I2x.at<float>(k, l) +
                                    (gamma * psi_gradient.at<float>(k, l) * I2xy.at<float>(k, l) *
                                     (I2xx.at<float>(k, l) + I2yy.at<float>(k, l)));

                        Au.at<float>(k, l) = BAu + GAu + alpha * div_u.at<float>(k, l);
                        Av.at<float>(k, l) = BAv + GAv + alpha * div_v.at<float>(k, l);
                        Du.at<float>(k, l) = BDu + GDu + div_d.at<float>(k, l);
                        Dv.at<float>(k, l) = BDv + GDv + div_d.at<float>(k, l);
                        D.at<float>(k, l) = Duv;
//                        std::cout << "Au = " << Au.at<float>(k, l) << " Av = " << Av.at<float>(k, l) << " Du = "
//                                  << Du.at<float>(k, l) << " Dv = " << Dv.at<float>(k, l) << " D = "
//                                  << D.at<float>(k, l)<<"\n\n";
                    }
                }
            }

//            for (int i = 0; i < I1.rows; ++i) {
//                for (int j = 0; j < I1.cols; ++j) {
////                    if (Dv.at<float>(i, j) == 0) {
//                        std::cout << "Here " << du.at<float>(i, j) << "\n";
////                    }
//                }
//            }

            sor_iterate(Au, Av, Du, Dv, D, du, dv, alpha, psi1, psi2, psi3, psi4, tolerance);
        }

        for (int k = 0; k < u.rows; ++k) {
            for (int l = 0; l < u.cols; ++l) {
                u.at<float>(k, l) += du.at<float>(k, l);
                v.at<float>(k, l) += dv.at<float>(k, l);
            }
        }

    }
}

void
calculateOpticalFlow(Mat_<float> &I1, Mat_<float> &I2, Mat_<float> &u, Mat_<float> &v, float alpha, float gamma,
                     int pyramidLevel, float pyramidFactor, float tolerance) {
    std::vector<Mat_<float>> I1_Pyramid;
    std::vector<Mat_<float>> I2_Pyramid;
    std::vector<Mat_<float>> u_Pyramid;
    std::vector<Mat_<float>> v_Pyramid;

    I1_Pyramid.push_back(I1);
    I2_Pyramid.push_back(I2);
    u_Pyramid.push_back(u);
    v_Pyramid.push_back(v);

    pyramidLevel = 1;
    // create pyramid
    for (int i = 1; i < pyramidLevel; ++i) {
        Mat_<float> I1_R;
        Mat_<float> I2_R;
        Mat_<float> u_R;
        Mat_<float> v_R;

        resize(I1_Pyramid[i - 1], I1_R, Size(), pyramidFactor, pyramidFactor, INTER_CUBIC);
        resize(I2_Pyramid[i - 1], I2_R, Size(), pyramidFactor, pyramidFactor, INTER_CUBIC);
        resize(u_Pyramid[i - 1], u_R, Size(), pyramidFactor, pyramidFactor, INTER_CUBIC);
        resize(v_Pyramid[i - 1], v_R, Size(), pyramidFactor, pyramidFactor, INTER_CUBIC);

        I1_Pyramid.push_back(I1_R);
        I2_Pyramid.push_back(I2_R);
        u_Pyramid.push_back(u_R);
        v_Pyramid.push_back(v_R);
    }
    std::cout << "Pyramid Level = " << pyramidLevel << "\n";
    for (int i = pyramidLevel - 1; i >= 0; i--) {
//        std::cout<<"I1_Pyramid "<<I1_Pyramid[i].rows<<" "<<I1_Pyramid[i].cols<<"    ";
//        std::cout<<"I2_Pyramid "<<I2_Pyramid[i].rows<<" "<<I2_Pyramid[i].cols<<"\n";
        //calculate optical flow
        calculateLevelOpticalFlow(I1_Pyramid[i], I2_Pyramid[i], u_Pyramid[i], v_Pyramid[i], alpha, gamma,
                                  tolerance);
        if (i != 0) {
            Size2i size2I(u_Pyramid[i-1].cols, u_Pyramid[i-1].rows);
            resize(u_Pyramid[i], u_Pyramid[i - 1], size2I, 0, 0, INTER_CUBIC);
            resize(v_Pyramid[i], v_Pyramid[i - 1], size2I, 0, 0, INTER_CUBIC);
        }
    }

//    for (int i = 0; i < u_Pyramid[0].rows; ++i) {
//        for (int j = 0; j < u_Pyramid[0].rows; ++j) {
//            if (u_Pyramid[0].at<float>(i, j) != 0) {
//                std::cout << "Here " << u_Pyramid[0].at<float>(i, j) << "\n";
//            }
//        }
//    }

    u = u_Pyramid[0];
    v = v_Pyramid[0];
}

