"""
Author: Piotr Książak
docScanner.py
"""

import numpy as np
import cv2 as cv

# apect ratio = 10:7
RATIO_HEIGHT = 10
RATIO_WIDTH = 7

# Size of the document to extract.
doc_width = 500;
doc_height = int((doc_width * RATIO_HEIGHT) / RATIO_WIDTH)

SPACE = 32
X = 120
ESC = 27

# The ratio can be specified manually in the code.
# Program works in the following manner:
# 1) I'm converting document image to grayscale, and performing thresholding,
#      along with morphological opening to remove as much noise as possible.
#      But probably there will be some noise left if the background has
#      some light even if the background is dark.
# 2) So to remove this noise but to have the document properly identified 
#      with white color, I decided next to perform finding contours and then
#      select the contour with the biggest arc, because this is the contour of
#      the document, another contours are dropped.
#      It could have been done also with contour area,
#      but I decided to choose arcLength.
# 3) Having the proper contour, I'm performing approximation
#      by approxPolyDP function, the goal is to have 4 corners.
# 4) Next, I'm drawing the lines and circles on the image
#      to identify the document.
# 5) Finally, we perform homography by using findHomography and warpPerspecive.

def main():
    winName = "Document Scanner"
    image = cv.imread("scanned-form.jpg", cv.IMREAD_COLOR);
    imageCopy = image.copy();

    width = image.shape[0];
    height = image.shape[1];

    print("Size of the image with document: {}x{}".format(width, height))

    # Initial operations on the image, thresholding and removing as much noise as possible.
    imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, imageThresh = cv.threshold(imageGray, 200, 255, cv.THRESH_BINARY)
    structElement = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    imageThresh = cv.morphologyEx(imageThresh, cv.MORPH_OPEN, structElement)

    # Finding contours of the document.
    contours, hierarchy = cv.findContours(imageThresh,
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("No. of contours: {}".format(len(contours)))

    max_id = 0
    max_arc = 0.0
    for i in range(len(contours)):
        arc = cv.arcLength(contours[i], True)
        print("Arc length: {}".format(arc))
        if arc > max_arc:
            max_id = i
            max_arc = arc
    print("The largest arc is {}".format(max_arc))

    #*************
    print("Press SPACE to automatically select the document.")
    cv.putText(image, "Press SPACE to auto-select the document.", (20, 50),
                cv.FONT_HERSHEY_COMPLEX, 1, (250, 0, 100), 2)
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.imshow(winName, image)

    key = 0
    while key != SPACE:
        key = cv.waitKey()
        if key == ESC:
            return -1

    imageCorners = imageCopy.copy();
    contourProper = []
    contourProper.append(contours[max_id]);

    #*//////////////////////////////////////////*#
    # Approximating the corners of the document. #
    # Finding the proper epsilon, to get the approx of exactly 4 corners.
    eps = 1.0
    while True:
        docCorners = cv.approxPolyDP(contourProper[0], eps, True)
        print("len(docCorners) = {}".format(len(docCorners)))
        if len(docCorners) == 4:
            break
        else:
            eps = eps + 0.1

    # Drawing the lines and circles to identify the document.
    for i in range(len(docCorners)):
        point1 = (docCorners[i][0][0], docCorners[i][0][1])
        point2 = (docCorners[(i + 1) % len(docCorners)][0][0],
                    docCorners[(i + 1) % len(docCorners)][0][1])
        cv.circle(imageCorners, point1, 20, (100, 0, 255), -1)
        cv.line(imageCorners, point1, point2, (100, 0, 255), 3)

    # Extracting the document.
    cv.putText(imageCorners, "Press X to extract the document!",
        (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (250, 0, 200), 2)
    cv.imshow(winName, imageCorners)
    while key != X:
        key = cv.waitKey();
        if key == ESC:
            return -1

    # ndarray to extract the image.
    imageExtracted = np.zeros((doc_height, doc_width))
    dstPoints = np.array([[doc_width, 0], [0, 0],
                    [0, doc_height], [doc_width, doc_height]])

    #*//////////////////*#
    # Finding homography #
    image = imageCopy.copy();
    h, status = cv.findHomography(docCorners, dstPoints)
    print(h)
    imageExtracted = cv.warpPerspective(image, h, (doc_width, doc_height));

    # Displaying extracted document.
    cv.namedWindow("Document extracted", cv.WINDOW_AUTOSIZE);
    cv.imshow("Document extracted", imageExtracted);
    cv.waitKey(0);

    print("Size of the image with extracted document: {}x{}".format(doc_width, doc_height))

    cv.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()
