import math

import cv2
import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode,
                                        max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=trackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipTds = [4,8,12,16,20]

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum = 0, draw=True):
        xlist = []
        ylist = []
        bbox = []
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 7, (0, 0, 255), cv.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
        return self.lmlist, bbox

    def fingerUp(self):
        fingers = []
        # Thumb
        if self.lmlist[[self.tipTds[0]][1]> self.lmlist[self.tipTds[0]-1][1]]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if self.lmlist[self.tipTds[id]][2] < self.lmlist[self.tipTds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def fingerDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r,(255, 0, 255),cv.FILLED)
            cv.circle(img, (x2, y2), r,(255, 0, 255),cv.FILLED)
            cv.circle(img, (cx, cy), r,(0, 0, 255),cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=False)

        if len(lmlist) != 0:
            # print(lmlist[4])
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fing = detector.fingerUp()
            print(fing)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
