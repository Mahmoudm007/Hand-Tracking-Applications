import math
import time
import mediapipe as mp
import cv2 as cv
import numpy as np
import hand_tracking_module as htm
import pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)
pTime = 0
vol = 0
volBar = 400
volPercentage = 0
detector = htm.handDetector(detectionCon= 0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# -65 : the minimum value
# zero : is the maximum value of sound
minVol = volRange[0]
maxVol = volRange[1]


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)

    cTime = time.time()
    fps = 1/ (cTime-pTime)
    pTime = cTime

    if len(lmlist) !=0 :
        x1 , y1 = lmlist[4][1],lmlist[4][2]
        x2 , y2 = lmlist[8][1],lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1,y1) , (x2,y2), (255,0,255), 3)
        cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # hand range (50 - 300)
        # Volume range (-65 - 0)

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPercentage = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
    cv.putText(img, f'FBS:{int(volPercentage)}%', (40, 450),
               cv.FONT_HERSHEY_COMPLEX,
               1, (255, 0, 0,), 3)

    cv.putText(img, f'FBS:{int(fps)}', (40,50),
               cv.FONT_HERSHEY_PLAIN,
               1, (255, 0, 0,), 3)

    cv.imshow("Image", img)
    cv.waitKey(1)
