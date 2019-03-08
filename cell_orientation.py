from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi


def find_biggest_contour(image):

    # Copy to prevent modification
    image = image.copy()
    image = cv.medianBlur(image, 9)
    _, contours, hierarchy = cv.findContours(image, cv.RETR_TREE,
                                             cv.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv.contourArea(contour), contour)
                     for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask, contours


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) +
                      (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1,
            cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1,
            cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1,
            cv.LINE_AA)


def getOrientation(pts, img):

    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0, 1],
                  eigenvectors[0, 0])  # orientation in radians

    return angle


src = cv.imread('./patterned_ethched.jpg')
# Check if image is loaded successfully
if src is None:
    print('Could not open or find the image')
    exit(0)
src = cv.GaussianBlur(src, (13, 13), 5)
# src = cv.medianBlur(src, 3)

# Convert image to grayscale
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
_, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# big_c, im, contours = find_biggest_contour(bw)
# area = cv.contourArea(big_c)
# if area < 1e2 or 1e5 < area:
#     # continue
#     # Draw each contour only for visualisation purposes
#     cv.drawContours(src, big_c, 0, (0, 0, 255), -1)
#     # Find the orientation of each shape
#     getOrientation(big_c, src)

for i, c in enumerate(contours):
    # Calculate the area of each contour
    area = cv.contourArea(c)
    # Ignore contours that are too small or too large
    if area < 1e2 or 1e5 < area:
        continue
    # Draw each contour only for visualisation purposes
    cv.drawContours(src, contours, i, (0, 0, 255), 1)
    # Find the orientation of each shape
    getOrientation(c, src)
cv.imshow('output', src)
cv.waitKey(0)
""" Comment this part of the code in to run a live PCA to get object orientation"""
# cap = cv.VideoCapture(0)
# while 1:
#     ret, src = cap.read()

#     # Check if image is loaded successfully
#     if src is None:
#         print('Could not open or find the image)
#         exit(0)
#         src = cv.GaussianBlur(src, (55, 55), 55)

#     # Convert image to grayscale
#     gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     # Convert image to binary
#     _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

#     # big_c, im, contours = find_biggest_contour(bw)
#     # area = cv.contourArea(big_c)
#     # if area < 1e2 or 1e5 < area:
#     #     # continue
#     #     # Draw each contour only for visualisation purposes
#     #     cv.drawContours(src, big_c, 0, (0, 0, 255), -1)
#     #     # Find the orientation of each shape
#     #     getOrientation(big_c, src)

#     for i, c in enumerate(contours):
#         # Calculate the area of each contour
#         area = cv.contourArea(c)
#         # Ignore contours that are too small or too large
#         if area < 1e2 or 1e5 < area:
#             continue
#         # Draw each contour only for visualisation purposes
#         cv.drawContours(src, contours, i, (0, 0, 255), -1)
#         # Find the orientation of each shape
#         getOrientation(c, src)
#     cv.imshow('output', src)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()
