#pragma once
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

class LaplacianZC {

private:
    cv::Mat laplace;
    int aperture;

public:
    LaplacianZC() : aperture(3) {}

    void setAperture(int a) {
        aperture = a;
    }

    int getAperture() const {
        return aperture;
    }

    cv::Mat computeLaplacian(const cv::Mat& image) {
        cv::Laplacian(image, laplace, CV_32F, aperture);
        return laplace;
    }

    cv::Mat getLaplacianImage(double scale = -1.0) {
        if (scale < 0) {
            double lapmin, lapmax;
            cv::minMaxLoc(laplace, &lapmin, &lapmax);
            scale = 127 / std::max(-lapmin, lapmax);
        }

        cv::Mat laplaceImage;
        laplace.convertTo(laplaceImage, CV_8U, scale, 128);
        return laplaceImage;
    }

    cv::Mat getZeroCrossings(cv::Mat laplace) {

        cv::Mat signImage;
        cv::threshold(laplace, signImage, 0, 255, cv::THRESH_BINARY);

        cv::Mat binary;
        signImage.convertTo(binary, CV_8U);

        cv::Mat dilated;
        cv::dilate(binary, dilated, cv::Mat());

        return dilated - binary;
    }

};