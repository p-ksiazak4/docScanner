#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

// apect ratio = 10:7
constexpr unsigned int RATIO_HEIGHT = 10;
constexpr unsigned int RATIO_WIDTH = 7;

// Size of the document to extract.
int doc_width = 500;
int doc_height = (doc_width * RATIO_HEIGHT) / RATIO_WIDTH;

constexpr unsigned int SPACE = 32;
constexpr unsigned int X = 120;
constexpr unsigned int ESC = 27;

// The ratio can be specified manually in the code.
// Program works in the following manner:
// 1) I'm converting document image to grayscale, and performing thresholding,
//      along with morphological opening to remove as much noise as possible.
//      But probably there will be some noise left if the background has some light
//      even if the background is dark.
// 2) So to remove this noise but to have the document properly identified with white color,
//      I decided next to perform finding contours and then select the contour with the
//      biggest arc, because this is the contour of the document, another contours are dropped.
//      It could have been done also with contour area, but I decided to choose arcLength.
// 3) Having the proper contour, I'm performing approximation by approxPolyDP function,
//      the goal is to have 4 corners.
// 4) Next, I'm drawing the lines and circles on the image to identify the document.
// 5) Finally, we perform homography by using findHomography and warpPerspecive.
int main()
{
    cv::String winName{ "Document Scanner" };
    cv::Mat image = cv::imread("scanned-form.jpg", cv::IMREAD_COLOR);
    cv::Mat imageClone = image.clone();

    int width = image.size().width;
    int height = image.size().height;

    std::cout << "Size of the image with document: " << width << "x" << height << '\n';

    // Initial operations on the image, thresholding and removing as much noise as possible.
    cv::Mat imageGray, imageThresh;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    cv::threshold(imageGray, imageThresh, 200, 255, cv::THRESH_BINARY);
    cv::Mat structElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{5, 5});
    cv::morphologyEx(imageThresh, imageThresh, cv::MORPH_OPEN, structElement);

    // Finding contours of the document.
    std::vector< std::vector<cv::Point> > contours, contourProper;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imageThresh, contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cout << "No. of contours: " << contours.size() << '\n';

    unsigned int max_id = 0;
    double max_arc = 0.0, arc;
    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        arc = cv::arcLength(contours[i], true);
        cout << "Arc length: " << arc << "\n";
        if (arc > max_arc)
        {
            max_id = i;
            max_arc = arc;
        }
    }
    cout << "The largest arc is " << max_arc << '\n';

    //*************
    cout << "Press SPACE to automatically select the document.\n";
    cv::putText(image, "Press SPACE to auto-select the document.", cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(250, 0, 100), 2);
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::imshow(winName, image);

    int key = 0;
    while (key != SPACE)
    {
        key = cv::waitKey();
        if (key == ESC)
            return -1;
    }


    cv::Mat imageCorners = imageClone.clone();
    contourProper.push_back(contours[max_id]);

    //*//////////////////////////////////////////*//
    // Approximating the corners of the document. //
    // Finding the proper epsilon, to get the approx of exactly 4 corners.
    std::vector< cv::Point > docCorners;
    double eps = 1.0;
    while (true)
    {
        cv::approxPolyDP(contourProper[0], docCorners, eps, true);
        cout << "docCorners.size() = " << docCorners.size() << '\n';
        if (docCorners.size() == 4)
            break;
        else
            eps = eps + 0.1;
    }

    // Drawing the lines and circles to identify the document.
    for (size_t i = 0; i < docCorners.size(); ++i)
    {
        cv::circle(imageCorners, docCorners[i], 20, cv::Scalar{100, 0, 255}, -1);
        cv::line(imageCorners, docCorners[i], docCorners[(i + 1) % docCorners.size()],
                cv::Scalar{100, 0, 255}, 3);
    }

    // Extracting the document.
    cv::putText(imageCorners, "Press X to extract the document!", cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(250, 0, 200), 2);
    cv::imshow(winName, imageCorners);
    while (key != X)
    {
        key = cv::waitKey();
        if (key == ESC)
            return -1;
    }


    // Mat to extract the image.
    cv::Mat imageExtracted = cv::Mat::zeros(doc_height, doc_width, CV_8UC3);
    std::vector<cv::Point> dstPoints {cv::Point{doc_width, 0},
                                      cv::Point{0, 0},
                                      cv::Point{0, doc_height},
                                      cv::Point{doc_width, doc_height} };

    //*//////////////////*//
    // Finding homography //
    image = imageClone.clone();
    cv::Mat h = cv::findHomography(docCorners, dstPoints);
    cout << h << '\n';
    cv::warpPerspective(image, imageExtracted, h, cv::Size{doc_width, doc_height});

    // Displaying extracted document.
    cv::namedWindow("Document extracted", cv::WINDOW_AUTOSIZE);
    cv::imshow("Document extracted", imageExtracted);
    cv::waitKey(0);

    std::cout << "Size of the image with extracted document: "
              << doc_width << "x" << doc_height << '\n';

    cv::destroyAllWindows();
    return 0;
}
