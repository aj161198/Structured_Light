import glob
import cv2
import numpy as np


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob('*.jpg')
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
for fname in images:
    img = cv2.imread(fname)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayscale, 15, 255, cv2.THRESH_BINARY)
    _, contours, h = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
    approx = None
    for cnt in cnts:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
    world=np.array([[[800.,0.]],[[0.,0.]],[[0.,600.]],[[800.,600.]]])
    H, _ = cv2.findHomography(approx, world, cv2.RANSAC, 3.)
    dst = cv2.warpPerspective(grayscale, H, (800,600))
    cv2.imwrite(fname[:-4]+"p.jpg", dst)
    ret, corners = cv2.findChessboardCorners(dst, (9,7),None)
    corners = cv2.cornerSubPix(dst, corners, (11, 11), (-1, -1), criteria)
    img = cv2.drawChessboardCorners(dst, (9,7), corners, ret)
    cv2.imshow('img', dst)
    cv2.waitKey(500)
    objpoints.append(objp)
    imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, dst.shape[::-1],None,None)
print(mtx, dist)
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints)
    tot_error += error
print("total error: ", tot_error / len(objpoints))

