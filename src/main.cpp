#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat cameraMatrix[2], distCoeffs[2], leftCamMap[2], rightCamMap[2];
cv::Mat R1, R2, P1, P2, Q;
cv::Mat R, T, E, F;


//cv::StereoBM matcher(cv::StereoBM::BASIC_PRESET, 80,5);
cv::StereoSGBM matcher;

void onMouseMove(int event, int x, int y, int flags, void* userdata)
{
    //hope no memory leaks are in here
    cv::Mat pic = cv::Mat::zeros(250, 250, CV_32FC3);
    cv::Mat* depthPicPtr = (cv::Mat*)userdata;
    std :: string text = "depth = " + std :: to_string(depthPicPtr->at<float>(x, y));
    cv::putText(
            pic,
            text,
            cv::Point(50,50),
            CV_FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            1,
            8,
            false
    );
    cv::imshow("Text", pic);
    //std :: cout << "x = " << x << " y = " << y  << std :: endl;
}

inline static void init_matcher()
{
    matcher.minDisparity = 0;
    matcher.SADWindowSize = 5;
    matcher.numberOfDisparities = 192;
    matcher.preFilterCap = 4;
    matcher.uniquenessRatio = 10;
    matcher.speckleWindowSize = 50;
    matcher.speckleRange = 1;
    matcher.disp12MaxDiff = 0;
    matcher.fullDP = false;
    matcher.P1 = 8 * 3 * matcher.SADWindowSize * matcher.SADWindowSize;
    matcher.P2 = 32 * 3 * matcher.SADWindowSize * matcher.SADWindowSize;

//    matcher.state->SADWindowSize = 9;
//    matcher.state->numberOfDisparities = 112;
//    matcher.state->preFilterSize = 5;
//    matcher.state->preFilterCap = 31;
//    matcher.state->minDisparity = 0;
//    matcher.state->textureThreshold = 10;
//    matcher.state->uniquenessRatio = 15;
//    matcher.state->speckleWindowSize = 100;
//    matcher.state->speckleRange = 32;
//    matcher.state->disp12MaxDiff = 1;
}


int main()
{
    init_matcher();

    cv::VideoCapture captureLeft(0);
    cv::VideoCapture captureRight(1);

    cv::namedWindow("left");
    cv::namedWindow("right");
    cv::namedWindow("depth");
    cv::namedWindow("disp");

    cv::Size boardSize(9, 6);
    float squareSize = 0.0215f;
    size_t imagesCount = 10;
//    std::string imageListFn = "./../../config/images.xml";
//    //std::vector<std::string> images = readImgList(imageListFn);
//
//    //stereo_calibrate(images, boardSize, squareSize);


    std::vector<std::vector<cv::Point2f>> imagePoints[2];
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Size imageSize(640, 480);

    //ten images should be enough for calibration
    imagePoints[0].resize(imagesCount);
    imagePoints[1].resize(imagesCount);
    objectPoints.resize(imagesCount);
    int j = 0;

    while (true)
    {
        cv::Mat leftImage;
        cv::Mat rightImage;
        cv::Mat greyLeftImage;
        cv::Mat greyRightImage;
        captureLeft >> leftImage;
        captureRight >> rightImage;

        cv::cvtColor(leftImage, greyLeftImage, CV_BGR2GRAY);
        cv::cvtColor(rightImage, greyRightImage, CV_BGR2GRAY);

        while (std::any_of(imagePoints[0].begin(), imagePoints[0].end(), [](std::vector<cv::Point2f>& vec){return vec.empty();}))
        {
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(
                    greyLeftImage,
                    boardSize,
                    corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
            );
            if (found)
            {
                cv::cornerSubPix(greyLeftImage, corners, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                              30, 0.01));
                imagePoints[0][j] = corners;
                j++;
            }
        }

        j = 0;
        while (std::any_of(imagePoints[1].begin(), imagePoints[1].end(), [](std::vector<cv::Point2f>& vec){return vec.empty();}))
        {
            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(
                    greyRightImage,
                    boardSize,
                    corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
            );
            if (found)
            {
                cornerSubPix(greyRightImage, corners, cv::Size(11,11), cv::Size(-1,-1),
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                              30, 0.01));
                imagePoints[1][j] = corners;
                j++;
            }
        }

        while (cameraMatrix[0].empty() && cameraMatrix[1].empty())
        {
            for (int i = 0; i < imagesCount; ++i)
            {
                for (int j = 0; j < boardSize.height; ++j)
                {
                    for (int k = 0; k < boardSize.width; ++k)
                    {
                        objectPoints[i].push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
                    }
                }
            }

            std::cout << "Running stereo calibration ..." << std::endl;


            cameraMatrix[0] = cv::initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
            cameraMatrix[1] = cv::initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);

            double rms = cv::stereoCalibrate(
                    objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-5),
                    CV_CALIB_FIX_ASPECT_RATIO +
                    CV_CALIB_ZERO_TANGENT_DIST +
                    CV_CALIB_USE_INTRINSIC_GUESS +
                    CV_CALIB_SAME_FOCAL_LENGTH +
                    CV_CALIB_RATIONAL_MODEL +
                    CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5
            );



            std::cout << "done with RMS error=" << rms << std::endl;

            cv::FileStorage fs("../../config/intrinsics.yml", CV_STORAGE_WRITE);
            if( fs.isOpened() )
            {
                fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
                "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
                fs.release();
            }
            else
            {
                std::cout << "Error: can not save the intrinsic parameters" << std::endl;
            }

            cv::Rect validRoi[2];

            stereoRectify(cameraMatrix[0], distCoeffs[0],
                          cameraMatrix[1], distCoeffs[1],
                          imageSize, R, T, R1, R2, P1, P2, Q,
                          CV_CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

            //due to a bug in OpenCV
            Q.convertTo(Q, CV_32FC1);
            Q.at<float>(3, 3) = -Q.at<float>(3, 3);

            fs.open("../../config/extrinsics.yml", CV_STORAGE_WRITE);
            if( fs.isOpened() )
            {
                fs << "R" << R << "T" << T << "R1" << R1 << "R2"
                << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
                fs.release();
            }
            else
            {
                std::cout << "Error: can not save the extrinsic parameters" << std::endl;
            }

            cv::initUndistortRectifyMap(
                    cameraMatrix[0],
                    distCoeffs[0],
                    R1,
                    cameraMatrix[0],
                    imageSize,
                    CV_32FC1,
                    leftCamMap[0],
                    leftCamMap[1]
            );
            std::cout << "Computed rectification map for left camera!" << std::endl;

            cv::initUndistortRectifyMap(
                    cameraMatrix[1],
                    distCoeffs[1],
                    R2,
                    cameraMatrix[1],
                    imageSize,
                    CV_32FC1,
                    rightCamMap[0],
                    rightCamMap[1]
            );
            std::cout << "Computed rectification map for right camera!" << std::endl;

        }

        cv::Mat mappedLeftImage, mappedRightImage;
        cv::remap(
                leftImage,
                mappedLeftImage,
                leftCamMap[0],
                leftCamMap[1],
                cv::INTER_LINEAR
        );

        cv::remap(
                rightImage,
                mappedRightImage,
                rightCamMap[0],
                rightCamMap[1],
                cv::INTER_LINEAR
        );


        cv::Mat disp;
        //matcher(greyLeftImage, greyRightImage, disp, CV_32F);
        matcher(greyLeftImage, greyRightImage, disp);
        cv::Mat disp8;
        cv::normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
        cv::imshow("disp", disp8);

        cv::Mat depthMap3D(disp.size(), CV_32FC1);
        cv::reprojectImageTo3D(disp, depthMap3D, Q, true);

        cv::Point2f leftCamPrincipalPoint(
                cameraMatrix[0].at<float>(0, 2),
                cameraMatrix[0].at<float>(1, 2)
        );
        cv::Point2f rightCamPrincipalPoint(
                cameraMatrix[1].at<float>(0, 2),
                cameraMatrix[1].at<float>(1, 2)
        );

        cv::Point2f midPoint = (leftCamPrincipalPoint + rightCamPrincipalPoint) * .5;
        cv::Point3f midPoint3D(midPoint.x, midPoint.y, 0);

        cv::Mat depthMap(depthMap3D.size(), CV_32FC1);

        for (int i = 0; i < depthMap.rows; ++i)
        {
            for (int j = 0; j < depthMap.cols; ++j)
            {
                cv::Point3f pixel =  depthMap3D.at<cv::Point3f>(i, j);
                depthMap.at<float>(i, j) = pixel.z;
                //depthMap.at<float>(i, j) = (float) cv::norm(pixel - midPoint3D);
            }
        }

        cv::imshow("left", leftImage);
        cv::imshow("right", rightImage);
        cv::imshow("depth", depthMap);
        cv::setMouseCallback("depth", onMouseMove, (void*)&depthMap);

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}