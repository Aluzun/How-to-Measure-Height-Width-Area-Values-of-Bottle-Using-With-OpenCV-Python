import cv2
import numpy as np

frameWidth = 1280  # Width of the frame
frameHeight = 720  # Heigth of the frame
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            par = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * par, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # Dimensions depends on the environmental conditions. That's why,
            # it can change with test environment.
            if h > 100 and h < 185:
                print('can', h, w, area)
            elif h > 185 and h < 370:
                print('not can', h, w, area)


            # cv2.putText(imgContour,"Points: " + str(len(approx)), (x + w + 20, y+20), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0),2)
            #cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
while True:
    succes, img = cap.read()
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    imGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        imGray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU)[1]
    imgCanny = cv2.Canny(thresh, 120, 150)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, img)
    # cropped = img[100:450,0:180]  If you want to monitor a limited area you can use crop method
    # cv2.imshow("Result",cropped)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
