# import math
# import os
# import cv2 as cv
# import mediapipe as mp
# import time
# import hand_tracking_module as htm
#
# folderPath = "images"
# myList = os.listdir(folderPath)
# print(myList)
# ovewLayList = []
# for imPath in myList:
#     image = cv.imread(f'{folderPath}/{imPath}')
#     ovewLayList.append(image)
#
# header = ovewLayList[0]
#
# cap = cv.VideoCapture(0)
# cap.set(3, 1000)
# cap.set(4, 720)
# detect = htm.handDetector(detectionCon= 0.5)
# while True:
#     success, img = cap.read()
#     img = cv.flip(img, 1)
#
#     img = detect.findHands(img)
#     lmList = detect.findPosition(img, draw=False)
#
#     if len(lmList) != 0:
#         print(lmList)
#         # tip of index and ,middle finger
#         x1, y1 = lmList[8][1:]  # from 1 till the end
#         x2, y2 = lmList[12][1:]  # from 1 till the end
#         # check which finger is up
#         fing = detect.fingerUp()
#         print(fing)
#
#         img[0:80, 0:892] = header
#
#     cv.imshow("Image", img)
#     cv.waitKey(1)

# -- coding: utf-8 --
# @Author: charliegallentine
# @Date:   2020-06-25 22:39:04
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2020-06-29 10:27:26

import cv2
import numpy as np
from time import sleep


# Run edge detection on image
def img_laplacian(img):
    return cv2.Canny(img, 30, 150)


# Normalize array to values from 0-1
def norm_0_1(arr):
    maximum = np.amax(arr)

    if maximum == 0:
        return arr

    return arr / np.amax(arr)


def add_border(arr):
    return np.pad(arr, pad_width=2, mode='constant', constant_values=255)


class ContourPoint:
    def _init_(self, row, col):
        self.row = row
        self.col = col

        # Must combine energies so that:
        # 	contour grows/shrinks
        # 	contour points remain equidistant
        # 	contour points prioritize lines

        # Total Energies
        self.energies = np.empty((7, 7), dtype=float)

    # Draws a single point in the contour
    def draw_point(self, img, val=255, w1=15, w2=4):
        '''Draws a cross on image at point in image'''
        img[self.row - w1 // 2:self.row + w1 // 2 + 1, self.col - w2 // 2:self.col + w2 // 2 + 1] = val
        img[self.row - w2 // 2:self.row + w2 // 2 + 1, self.col - w1 // 2:self.col + w1 // 2 + 1] = val

    # Calculates energies to move points away from each other
    # 	Moves to point which is furthest from all contour points
    def calc_energy_distance(self, contour_r, contour_c, shrink=False):
        r = self.row
        c = self.col

        energy_distance = np.zeros((7, 7), dtype=float)

        for i in range(-3, 4):
            for j in range(-3, 4):
                # (x2 - x1)^2 + (y2 - y1)^2
                energy_distance[i + 3, j + 3] = np.sum(np.square((r + i) - contour_r) + np.square((c + j) - contour_c))

        if shrink:
            self.energies = np.dstack((self.energies, norm_0_1(energy_distance)))
        else:
            self.energies = np.dstack((self.energies, 1.0 - norm_0_1(energy_distance)))

    # Attempts to move points to center of neighbors
    def calc_energy_deviation(self, prior_point, next_point):
        r = self.row
        c = self.col

        energy_deviation = np.zeros((7, 7), dtype=float)

        for i in range(-3, 4):
            for j in range(-3, 4):
                d2n = np.square(r + i - next_point.row) + np.square(c + j - next_point.col)
                d2p = np.square(r + i - prior_point.row) + np.square(c + j - prior_point.col)

                energy_deviation[i + 3, j + 3] = np.power(np.absolute(d2n - d2p), 2)

        self.energies = np.dstack((self.energies, norm_0_1(energy_deviation)), )

    # Pulls contour to higher values on grayscale image
    def calc_energy_gradient(self, img, to_low=False):
        r = self.row
        c = self.col

        energy_gradient = np.zeros((7, 7), dtype=float)

        for i in range(-3, 4):
            for j in range(-3, 4):
                energy_gradient[i + 3, j + 3] = np.square(img[r + i, c + j])

        if to_low:
            self.energies = np.dstack((self.energies, norm_0_1(energy_gradient)))
        else:
            self.energies = np.dstack((self.energies, 1.0 - norm_0_1(energy_gradient)))

    # add all energies in contour
    def add_energies(self):
        self.energies = np.sum(self.energies, axis=2)

    def adjust_point(self):
        minimum_energy = np.argmin(self.energies)

        # 0,0 : 0,1 : 0,2 : 0,3 : 0,4 : 0,5 : 0,6
        # 1,0 : 1,1 : 1,2 : 1,3 : 1,4 : 1,5 : 1,6
        # 2,0 : 2,1 : 2,2 : 2,3 : 2,4 : 2,5 : 2,6
        # 3,0 : 3,1 : 3,2 : 3,3 : 3,4 : 3,5 : 3,6
        # 4,0 : 4,1 : 4,2 : 4,3 : 4,4 : 4,5 : 4,6
        # 5,0 : 5,1 : 5,2 : 5,3 : 5,4 : 5,5 : 5,6
        # 6,0 : 6,1 : 6,2 : 6,3 : 6,4 : 6,5 : 6,6

        # Add row adjustment, is shift from center, 3,3 on 7x7 kernel
        self.row += minimum_energy // 7 - 3
        # Add column adjustment, same deal as row adjust
        self.col += minimum_energy % 7 - 3


class Contour:
    def _init_(self, contour=None):
        self.contour = []
        self.contour_r = []
        self.contour_c = []
        self.average_distance = 0.0

        if contour != None:
            for point in contour:
                self.contour_r.append(point[0])
                self.contour_c.append(point[1])
                self.contour.append(ContourPoint(point[0], point[1]))

            self.contour_r = np.array(self.contour_r)
            self.contour_c = np.array(self.contour_c)
            self.contour = np.array(self.contour)

            tmp_r = np.roll(np.copy(self.contour_r), 1)
            tmp_c = np.roll(np.copy(self.contour_c), 1)

            self.average_distance = np.average(
                np.sqrt(np.power(self.contour_r - tmp_r, 2) + np.power(self.contour_c - tmp_c, 2)))

    def draw_contour(self, img, val=255, w1=15, w2=2):
        '''Draws each point in contour and line between them'''
        for i in range(len(self.contour)):
            self.contour[i].draw_point(img)

            cv2.line(
                img,
                (self.contour_c[i], self.contour_r[i]),
                (self.contour_c[(i + 1) % len(self.contour_c)], self.contour_r[(i + 1) % len(self.contour_r)]),
                val, 1)

    def calc_energies(self, img):
        for i, point in enumerate(self.contour):
            next_point = self.contour[(i + 1) % len(self.contour)]

            if i == 0:
                prior_point = self.contour[-1]
            else:
                prior_point = self.contour[i - 1]

            point.energies = np.zeros((7, 7))

            point.calc_energy_distance(self.contour_r, self.contour_c)
            point.calc_energy_deviation(prior_point, next_point)
            point.calc_energy_gradient(img)

            point.add_energies()

    def update_points(self):
        self.contour_r = []
        self.contour_c = []

        for point in self.contour:
            point.adjust_point()

            self.contour_r.append(point.row)
            self.contour_c.append(point.col)

        self.contour_r = np.array(self.contour_r)
        self.contour_c = np.array(self.contour_c)
        self.contour = np.array(self.contour)

        tmp_r = np.roll(np.copy(self.contour_r), 1)
        tmp_c = np.roll(np.copy(self.contour_c), 1)

        self.average_distance = np.average(
            np.sqrt(np.power(self.contour_r - tmp_r, 2) + np.power(self.contour_c - tmp_c, 2)))

    def insert_points(self):
        tmp_contour = []

        for i, point in enumerate(self.contour):
            next_row = (point.row + (self.contour[(i + 1) % len(self.contour)].row)) // 2
            next_col = (point.col + (self.contour[(i + 1) % len(self.contour)].col)) // 2

            tmp_contour.append(point)
            tmp_contour.append(ContourPoint(next_row, next_col))

        self.contour = tmp_contour
        self.contour_r = []
        self.contour_c = []

        for point in self.contour:
            self.contour_r.append(point.row)
            self.contour_c.append(point.col)

        self.contour_r = np.array(self.contour_r)
        self.contour_c = np.array(self.contour_c)
        self.contour = np.array(self.contour)

        tmp_r = np.roll(np.copy(self.contour_r), 1)
        tmp_c = np.roll(np.copy(self.contour_c), 1)

        self.average_distance = np.average(
            np.sqrt(np.power(self.contour_r - tmp_r, 2) + np.power(self.contour_c - tmp_c, 2)))








# -- coding: utf-8 --
# @Author: Charlie Gallentine
# @Date:   2020-06-28 19:52:40
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2020-06-28 21:32:12


# Row/Column format
contour_points = [(88,156),(144,151),(170,110),(205,155),(241,155),(189,224),(135,225)]

img = np.array(cv2.imread("car.jpg", 0))
lap = np.array(img_laplacian(img))
lap = add_border(lap)

contour = Contour(contour_points)

# Add a ton more points
contour.insert_points()
contour.insert_points()
contour.insert_points()
contour.insert_points()
contour.insert_points()

# Create series of images fitting contour
allimgs = []
for i in range(100):
	lapcpy = np.copy(lap)

	contour.calc_energies(lapcpy)

	contour.update_points()

	contour.draw_contour(lapcpy)

	allimgs.append(lapcpy)

	# cv2.imwrite('./imgs/%d.png' % i,lapcpy)

# Creates a short animation of curve fitting
breakon = False
while True:
	for img in allimgs:
		img = cv2.resize(img, (550,550))
		cv2.imshow('Image', img)

		sleep(0.01)

		k = cv2.waitKey(33)
		if k==27:    # Esc key to stop
			breakon = True

		if breakon:
			break

	if breakon:
			break