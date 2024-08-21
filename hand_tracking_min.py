import os
import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
""""
false : means that the video not always tracking the hands make it (TRUE) make the project slow
min_tracking_confidence : the threshold of the tracking
"""
# calculating the frame rate
pTime = 0  # prev. time
cTime = 0  # current time
while True:
    success , img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # make he process and give as the frame rate
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # get the landmarks information and the finger number or id
            for id, lm in enumerate(handLms.landmark):
                # print the coordinates of each landmark in the hand in decimal before convert it to pixels
                print(id, lm)
                # convert from decimal to pixels
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 12:
                    # draw a circle on the index 12
                    # cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)

                cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # handLms to make the landmarks on one hand

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv.imshow("Image", img)
    cv.waitKey(1)
