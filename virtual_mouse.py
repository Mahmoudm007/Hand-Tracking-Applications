import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import autopy
import hand_tracking_module as htm

wCam, hCam = 640 , 480
pTime = 0  # prev. time
cTime = 0  # current time

detector = htm.handDetector(maxHands=1)

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


while True:
    # 1) find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)
    # 2) get hte yip of the index and middle fingers
    # 3) check which finger are up
    # 4) only index finger : moving mode
    # 5) convert coordinates
    # 6) smooth the values
    # 7) move the mouse
    # 8) when middle and index up : clicking mode
    # 9) find the distance between the fingers
    # 10) click the miouse when the distance is short

    # 11) frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # 12) display the camera port
    cv.imshow("Image", img)
    cv.waitKey(1)
