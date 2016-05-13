#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat cameraMatrix[2], distCoeffs[2], leftCamMap[2], rightCamMap[2];
cv::Mat R1, R2, P1, P2, Q;
cv::Mat R, T, E, F;


cv::StereoBM matcher(cv::StereoBM::BASIC_PRESET, 80,5);
// cv::StereoSGBM matcher;

void onMouseMove(int event, int x, int y, int flags, void* userdata)
{
    //hope no memory leaks are in here
    cv::Mat pic(250, 250, CV_8UC3);
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
    // matcher.SADWindowSize = 5;
    // matcher.numberOfDisparities = 192;
    // matcher.preFilterCap = 4;
    // matcher.minDisparity = -64;
    // matcher.uniquenessRatio = 1;
    // matcher.speckleWindowSize = 150;
    // matcher.speckleRange = 2;
    // matcher.disp12MaxDiff = 10;
    // matcher.fullDP = false;
    // matcher.P1 = 600;
    // matcher.P2 = 2400;

    matcher.state->SADWindowSize = 9;
    matcher.state->numberOfDisparities = 112;
    matcher.state->preFilterSize = 5;
    matcher.state->preFilterCap = 31;
    matcher.state->minDisparity = 0;
    matcher.state->textureThreshold = 10;
    matcher.state->uniquenessRatio = 15;
    matcher.state->speckleWindowSize = 100;
    matcher.state->speckleRange = 32;
    matcher.state->disp12MaxDiff = 1;
}

static void stereo_calibrate(std::vector<std::string> images, cv::Size boardSize, float squareSize)
{
    std::vector<std::vector<cv::Point2f>> imagePoints[2];
    std::vector<std::vector<cv::Point3f>> objectPoints;
    cv::Size imageSize;

    int imagesCount = (int) (images.size() / 2);

    imagePoints[0].resize((unsigned long) imagesCount);
    imagePoints[1].resize((unsigned long) imagesCount);
    objectPoints.resize((unsigned long) imagesCount);

    for (int i = 0, j = 0; i < imagesCount; ++i)
    {
        int k;
        for(k = 0; k < 2; k++)
        {
            const std::string& filename = images[i * 2 + k];
            cv::Mat img = cv::imread(filename, 0);

            if (imageSize == cv::Size())
            {
                imageSize = img.size();
            }

            std::vector<cv::Point2f>& corners = imagePoints[k][j];
            bool found = findChessboardCorners(
                    img,
                    boardSize,
                    corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE
            );
            cornerSubPix(img, corners, cv::Size(11,11), cv::Size(-1,-1),
                         cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                      30, 0.01));
        }
    }


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


    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);

    double rms = stereoCalibrate(
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

    //due to a bug in OpenCV
    Q.convertTo(Q, CV_32FC1);
    Q.at<float>(3, 3) = -Q.at<float>(3, 3);

    std::cout << "done with RMS error=" << rms << std::endl;

    cv::FileStorage fs("../config/intrinsics.yml", CV_STORAGE_WRITE);
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

    fs.open("../config/extrinsics.yml", CV_STORAGE_WRITE);
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

std::vector<std::string> readImgList(const std::string& filename)
{
    std::vector<std::string> ret;
    cv::FileStorage fs(filename, CV_STORAGE_READ);
    cv::FileNode node = fs.getFirstTopLevelNode();
    for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it)
    {
        ret.push_back((std::string)(*it));
    }
    return ret;
}

int main()
{
    cv::VideoCapture captureLeft(0);
    cv::VideoCapture captureRight(1);

    cv::namedWindow("left");
    cv::namedWindow("right");
    cv::namedWindow("depth");

    cv::Size board_size(9, 6);
    float square_size = 0.215f;
    std::string imageListFn = "./../../config/images.xml";
    std::vector<std::string> images = readImgList(imageListFn);

    stereo_calibrate(images, board_size, square_size);
    init_matcher();

    while (true)
    {
        cv::Mat leftImage;
        cv::Mat rightImage;
        captureLeft >> leftImage;
        captureRight >> rightImage;
//
//        cv::Mat mappedLeftImage, mappedRightImage;
//        cv::remap(
//                leftImage,
//                mappedLeftImage,
//                leftCamMap[0],
//                leftCamMap[1],
//                cv::INTER_LINEAR
//        );
//
//        cv::remap(
//                rightImage,
//                mappedRightImage,
//                rightCamMap[0],
//                rightCamMap[1],
//                cv::INTER_LINEAR
//        );
//
//        cv::Mat greyLeftImage, greyRightImage;
//        cv::cvtColor(mappedLeftImage, greyLeftImage, cv::COLOR_BGR2GRAY);
//        cv::cvtColor(mappedRightImage, greyRightImage, cv::COLOR_BGR2GRAY);
//
//        cv::Mat disp;
//        matcher(greyLeftImage, greyRightImage, disp, CV_32F);
//
//        cv::Mat depthMap3D(disp.size(), CV_32FC1);
//        cv::reprojectImageTo3D(disp, depthMap3D, Q, true);
//
//        cv::Point2f leftCamPrincipalPoint(
//                cameraMatrix[0].at<float>(0, 2),
//                cameraMatrix[0].at<float>(1, 2)
//        );
//        cv::Point2f rightCamPrincipalPoint(
//                cameraMatrix[0].at<float>(0, 2),
//                cameraMatrix[0].at<float>(1, 2)
//        );
//
//        cv::Point2f midPoint = (leftCamPrincipalPoint + rightCamPrincipalPoint) * .5;
//        cv::Point3f midPoint3D(midPoint.x, midPoint.y, 0);
//
//        cv::Mat depthMap(depthMap3D.size(), CV_32FC1);
//
//        for (int i = 0; i < depthMap.rows; ++i)
//        {
//            for (int j = 0; j < depthMap.cols; ++j)
//            {
//                cv::Point3f pixel =  depthMap3D.at<cv::Point3f>(i, j);
//                depthMap.at<float>(i, j) = (float) cv::norm(pixel - midPoint3D);
//            }
//        }

        cv::imshow("left", leftImage);
        cv::imshow("right", rightImage);
        //cv::imshow("depth", depthMap);

        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}